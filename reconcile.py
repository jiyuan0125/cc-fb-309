#!/usr/bin/env python3
"""订单表与支付表的日粒度实收 GMV 对账。

口径要点：
* 订单表 ``updated_at`` 是运营系统的北京时间（无时区标记），直接取当地日历日。
* 支付表 ``paid_at`` 是 UTC，先换算成 Asia/Shanghai 再取日历日。
* 同一订单在订单表可能有多行状态记录，偶有乱序，最终状态按状态机定：
  成交 > 改价 > 下单 > 取消；只有最终状态为「成交」的订单计入实收，
  金额取该行金额（即改价后的订单净额）。
* 差异行为「日 × 订单」粒度，每行恰好归入四类之一。

用法::

    python3 reconcile.py
    python3 reconcile.py --orders data/orders.csv --payments data/payments.csv \\
        --outdir out
"""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("reconcile")

BUSINESS_TIMEZONE = "Asia/Shanghai"

ORDER_COLUMNS = ("order_id", "status", "amount", "updated_at")
PAYMENT_COLUMNS = ("pay_id", "order_id", "amount", "paid_at")

# 状态机优先级：成交优先于改价，改价优先于下单，取消最低。
STATUS_PRIORITY: dict[str, int] = {"成交": 3, "改价": 2, "下单": 1, "取消": 0}
REVENUE_STATUS = "成交"

CATEGORY_TIMEZONE = "时区窗口差"
CATEGORY_REFUND = "退款或负向冲抵"
CATEGORY_DUPLICATE = "重复或多状态行"
CATEGORY_UNMATCHED = "其余未对上"
CATEGORIES: tuple[str, ...] = (
    CATEGORY_TIMEZONE,
    CATEGORY_REFUND,
    CATEGORY_DUPLICATE,
    CATEGORY_UNMATCHED,
)

DIFF_COLUMNS = (
    "date",
    "order_id",
    "order_amount",
    "payment_amount",
    "diff_amount",
    "category",
)
DAILY_COLUMNS = ("date", "gmv", "mom_amount", "mom_pct")


@dataclass(frozen=True)
class ReconcileResult:
    """一次对账的完整结果，便于测试与调用方复用。"""

    orders_daily: pd.DataFrame
    payments_daily: pd.DataFrame
    diff: pd.DataFrame
    category_stats: pd.DataFrame
    order_total_cents: int
    payment_total_cents: int
    diff_net_cents: int
    diff_abs_cents: int
    summary: str
    orders_daily_path: Path
    payments_daily_path: Path
    diff_path: Path
    summary_path: Path


# ---------------------------------------------------------------------------
# 读取与校验
# ---------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], path: Path) -> None:
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} 缺少必需列: {', '.join(missing)}")


def _parse_amount_cents(series: pd.Series, path: Path) -> pd.Series:
    """把金额列解析成「分」，用整数计算避免浮点误差。"""
    amount = pd.to_numeric(series, errors="coerce")
    bad = amount.isna()
    if bad.any():
        # CSV 行号 = DataFrame 行号 + 2（表头占 1 行）
        sample = ", ".join(str(pos + 2) for pos in amount.index[bad][:5])
        raise ValueError(f"{path} 存在无法解析的金额，CSV 行号示例: {sample}")
    return (amount * 100).round().astype("int64")


def load_orders(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    _require_columns(frame, ORDER_COLUMNS, path)

    frame["order_id"] = frame["order_id"].str.strip()
    frame["status"] = frame["status"].str.strip()
    unknown = sorted(set(frame["status"]) - set(STATUS_PRIORITY))
    if unknown:
        raise ValueError(f"{path} 出现状态机未定义的状态: {', '.join(unknown)}")
    if (frame["order_id"] == "").any():
        raise ValueError(f"{path} 存在空 order_id")

    frame["amount_cents"] = _parse_amount_cents(frame["amount"], path)
    updated_at = pd.to_datetime(frame["updated_at"], errors="coerce")
    if updated_at.isna().any():
        sample = ", ".join(
            str(pos + 2) for pos in updated_at.index[updated_at.isna()][:5]
        )
        raise ValueError(f"{path} 存在无法解析的 updated_at，CSV 行号示例: {sample}")
    frame["updated_at"] = updated_at
    frame["order_date"] = updated_at.dt.date
    LOGGER.info(
        "读取订单表 %s：%d 行，%d 个订单", path, len(frame), frame["order_id"].nunique()
    )
    return frame


def load_payments(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    _require_columns(frame, PAYMENT_COLUMNS, path)

    frame["order_id"] = frame["order_id"].str.strip()
    frame["pay_id"] = frame["pay_id"].str.strip()
    if (frame["order_id"] == "").any():
        raise ValueError(f"{path} 存在空 order_id")
    if (frame["pay_id"] == "").any():
        raise ValueError(f"{path} 存在空 pay_id")

    frame["amount_cents"] = _parse_amount_cents(frame["amount"], path)
    paid_utc = pd.to_datetime(frame["paid_at"], utc=True, errors="coerce")
    if paid_utc.isna().any():
        sample = ", ".join(str(pos + 2) for pos in paid_utc.index[paid_utc.isna()][:5])
        raise ValueError(f"{path} 存在无法解析的 paid_at，CSV 行号示例: {sample}")
    frame["paid_utc"] = paid_utc
    frame["paid_utc_date"] = paid_utc.dt.date
    frame["paid_date"] = paid_utc.dt.tz_convert(BUSINESS_TIMEZONE).dt.date
    LOGGER.info(
        "读取支付表 %s：%d 行，%d 笔，%d 个订单",
        path,
        len(frame),
        frame["pay_id"].nunique(),
        frame["order_id"].nunique(),
    )
    return frame


# ---------------------------------------------------------------------------
# 订单侧：状态机定最终状态
# ---------------------------------------------------------------------------


def final_status_rows(orders: pd.DataFrame) -> pd.DataFrame:
    """按状态机优先级取每个订单的最终状态行。

    时间戳只用于同优先级下的兜底排序，跨状态不参与判断，因此「乱序」不影响结果。
    只有最终状态为「成交」的行才计入订单侧实收。
    """
    ranked = orders.assign(_priority=orders["status"].map(STATUS_PRIORITY))
    ranked = ranked.sort_values(
        ["_priority", "updated_at"], ascending=[False, False], kind="stable"
    )
    final = ranked.groupby("order_id", sort=False).head(1)
    return final.drop(columns=["_priority"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 日粒度 GMV 与环比
# ---------------------------------------------------------------------------


def _daily_series(amount_cents_by_date: pd.Series, calendar: list[date]) -> pd.DataFrame:
    gmv_cents = (
        amount_cents_by_date.reindex(calendar).fillna(0).astype("int64").sort_index()
    )
    previous = gmv_cents.shift(1)
    mom_cents = gmv_cents - previous

    table = pd.DataFrame(
        {
            "date": gmv_cents.index,
            "gmv_cents": gmv_cents.to_numpy(),
            "mom_cents": mom_cents.to_numpy(),
            "mom_pct": float("nan"),
        }
    )
    # 环比百分比：分母取绝对值，基期为 0 或首日时留空（无法定义）。
    mask = (previous.notna() & (previous != 0)).to_numpy()
    if mask.any():
        values = mom_cents.to_numpy()[mask] / abs(previous.to_numpy()[mask]) * 100
        table.loc[mask, "mom_pct"] = values.round(2)
    return table


def build_daily_tables(
    revenue: pd.DataFrame, payments: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[date]]:
    order_by_date = revenue.groupby("order_date")["amount_cents"].sum()
    payment_by_date = payments.groupby("paid_date")["amount_cents"].sum()

    all_dates = sorted(set(order_by_date.index) | set(payment_by_date.index))
    if not all_dates:
        raise ValueError("订单表与支付表都没有可用日期，无法生成日粒度结果")
    calendar = list(pd.date_range(all_dates[0], all_dates[-1], freq="D").date)

    orders_daily = _daily_series(order_by_date, calendar)
    payments_daily = _daily_series(payment_by_date, calendar)
    return orders_daily, payments_daily, calendar


# ---------------------------------------------------------------------------
# 对账：日 × 订单
# ---------------------------------------------------------------------------


def build_cells(revenue: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    """把两侧数据聚合成「日 × 订单」对账单元。"""
    order_side = (
        revenue.groupby(["order_date", "order_id"], as_index=False)["amount_cents"]
        .sum()
        .rename(columns={"order_date": "date", "amount_cents": "order_cents"})
    )
    payment_side = (
        payments.groupby(["paid_date", "order_id"], as_index=False)["amount_cents"]
        .sum()
        .rename(columns={"paid_date": "date", "amount_cents": "pay_cents"})
    )

    cells = order_side.merge(payment_side, on=["date", "order_id"], how="outer")
    cells["date"] = pd.to_datetime(cells["date"]).dt.date
    cells["order_cents"] = cells["order_cents"].fillna(0).astype("int64")
    cells["pay_cents"] = cells["pay_cents"].fillna(0).astype("int64")
    cells["diff_cents"] = cells["order_cents"] - cells["pay_cents"]
    cells = cells.sort_values(["date", "order_id"]).reset_index(drop=True)
    return cells


def classify_cells(
    cells: pd.DataFrame, orders: pd.DataFrame, payments: pd.DataFrame
) -> pd.DataFrame:
    """给每条差异行归入唯一分类，返回带 category 列的差异明细。

    判定顺序即互斥依据：时区窗口差 -> 退款或负向冲抵 -> 重复或多状态行 -> 其余未对上。
    """
    status_rows = orders.groupby("order_id")["status"]
    row_count = status_rows.size().to_dict()
    has_deal = status_rows.agg(
        lambda items: bool((items == REVENUE_STATUS).any())
    ).to_dict()

    pay_utc_dates = (
        payments.groupby(["paid_date", "order_id"])["paid_utc_date"]
        .agg(lambda items: frozenset(items))
        .to_dict()
    )
    duplicate_keys = _duplicate_payment_keys(payments)

    cell_lookup = {
        (row.date, row.order_id): idx
        for idx, row in cells.iterrows()
    }

    diff_cells = cells[cells["diff_cents"] != 0]
    categories: dict[int, str] = {}

    # 1) 时区窗口差：支付行按 UTC→北京换算后跨日，且相邻日存在金额相同的订单侧记录。
    timezone_matched: set[int] = set()
    for idx, cell in diff_cells.iterrows():
        if cell["pay_cents"] <= 0:
            continue
        utc_dates = pay_utc_dates.get((cell["date"], cell["order_id"]))
        if utc_dates is None or len(utc_dates) != 1:
            continue
        utc_date = next(iter(utc_dates))
        if utc_date == cell["date"]:
            continue  # 换算没有换日，不属于时区窗口
        peer_idx = cell_lookup.get((utc_date, cell["order_id"]))
        if peer_idx is None or peer_idx == idx:
            continue
        peer = cells.loc[peer_idx]
        if peer["pay_cents"] != 0:
            continue
        if peer["order_cents"] != cell["pay_cents"]:
            continue
        if peer["diff_cents"] != -cell["diff_cents"]:
            continue
        timezone_matched.update({idx, peer_idx})

    for idx, cell in diff_cells.iterrows():
        order_id = cell["order_id"]
        if idx in timezone_matched:
            categories[idx] = CATEGORY_TIMEZONE
        elif cell["order_cents"] < 0 or cell["pay_cents"] < 0:
            categories[idx] = CATEGORY_REFUND
        elif (cell["date"], order_id) in duplicate_keys:
            categories[idx] = CATEGORY_DUPLICATE
        elif row_count.get(order_id, 0) > 1 and not has_deal.get(order_id, False):
            # 多状态行：订单有下单/改价记录但从未成交，支付找不到应收金额。
            categories[idx] = CATEGORY_DUPLICATE
        else:
            categories[idx] = CATEGORY_UNMATCHED

    missing = sorted(set(diff_cells.index) - set(categories))
    if missing:
        raise AssertionError(f"存在未归类的差异行: {missing}")

    detail = diff_cells.copy()
    detail["category"] = pd.Series(categories, dtype="object").loc[detail.index]
    unknown = set(detail["category"]) - set(CATEGORIES)
    if unknown:
        raise AssertionError(f"出现四类之外的分类: {sorted(unknown)}")

    columns = ["date", "order_id", "order_cents", "pay_cents", "diff_cents", "category"]
    return detail[columns].reset_index(drop=True)


def to_diff_table(detail: pd.DataFrame) -> pd.DataFrame:
    """把内部分单位的差异明细转成对外输出的元金额表。"""
    table = pd.DataFrame(
        {
            "date": detail["date"],
            "order_id": detail["order_id"],
            "order_amount": (detail["order_cents"] / 100).round(2),
            "payment_amount": (detail["pay_cents"] / 100).round(2),
            "diff_amount": (detail["diff_cents"] / 100).round(2),
            "category": detail["category"],
        }
    )
    return table[list(DIFF_COLUMNS)]


def _duplicate_payment_keys(payments: pd.DataFrame) -> set[tuple[date, str]]:
    """重复支付：pay_id 重复，或同一订单同一时刻同一金额出现多行。"""
    duplicated = payments[
        payments.duplicated(subset=["pay_id"], keep=False)
        | payments.duplicated(subset=["order_id", "paid_at", "amount"], keep=False)
    ]
    return set(zip(duplicated["paid_date"], duplicated["order_id"]))


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def build_category_stats(detail: pd.DataFrame) -> pd.DataFrame:
    """按四类汇总笔数、差异净额与差异绝对值（单位：分）。"""
    rows = []
    for category in CATEGORIES:
        subset = detail[detail["category"] == category]
        rows.append(
            {
                "category": category,
                "count": len(subset),
                "net_cents": int(subset["diff_cents"].sum()),
                "abs_cents": int(subset["diff_cents"].abs().sum()),
            }
        )
    return pd.DataFrame(rows, columns=["category", "count", "net_cents", "abs_cents"])


def _yuan(cents: float) -> str:
    return f"{cents / 100:.2f}"


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(width - _display_width(text), 0)


def render_summary(
    cells: pd.DataFrame, category_stats: pd.DataFrame, calendar: list[date]
) -> str:
    order_total = int(cells["order_cents"].sum())
    payment_total = int(cells["pay_cents"].sum())
    diff_net = order_total - payment_total
    diff_abs = int(cells["diff_cents"].abs().sum())
    diff_rows = int((cells["diff_cents"] != 0).sum())
    start, end = calendar[0], calendar[-1]

    lines = [
        "订单与支付双表对账汇总",
        "=" * 48,
        f"对账区间: {start} ~ {end}（北京时间日粒度，共 {len(calendar)} 天）",
        f"订单侧实收GMV(成交订单净额): {_yuan(order_total)} 元",
        f"支付侧实收GMV: {_yuan(payment_total)} 元",
        f"对账差异净额(订单-支付): {_yuan(diff_net)} 元",
        f"对账差异绝对值合计: {_yuan(diff_abs)} 元",
        f"对不上记录数: {diff_rows} 条（日 × 订单）",
        "",
        "分类明细（四类互斥，每行恰好一类）:",
    ]

    headers = ("分类", "笔数", "差异净额(元)", "差异绝对值(元)")
    widths = (18, 8, 16, 18)

    def row_text(values: tuple[str, ...]) -> str:
        padded = (_pad(value, width) for value, width in zip(values, widths, strict=True))
        return "  ".join(padded).rstrip()

    lines.append(row_text(headers))
    lines.append("-" * (_display_width(row_text(headers)) + 2))
    for stat in category_stats.itertuples(index=False):
        lines.append(
            row_text(
                (
                    stat.category,
                    str(stat.count),
                    _yuan(stat.net_cents),
                    _yuan(stat.abs_cents),
                )
            )
        )
    lines.append(
        row_text(
            (
                "合计",
                str(int(category_stats["count"].sum())),
                _yuan(int(category_stats["net_cents"].sum())),
                _yuan(int(category_stats["abs_cents"].sum())),
            )
        )
    )
    lines.extend(
        [
            "",
            (
                "每日实收GMV与环比见 orders_daily.csv / payments_daily.csv，"
                "差异明细见 diff.csv。"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.2f")
    LOGGER.info("已写出 %s（%d 行）", path, len(frame))


def run(orders_path: Path, payments_path: Path, outdir: Path) -> ReconcileResult:
    """读取两表、算日粒度实收与环比、对账并落盘。"""
    for path in (orders_path, payments_path):
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在: {path}")

    outdir.mkdir(parents=True, exist_ok=True)

    orders = load_orders(orders_path)
    payments = load_payments(payments_path)

    final = final_status_rows(orders)
    LOGGER.info("订单最终状态分布: %s", final["status"].value_counts().to_dict())
    revenue = final[final["status"] == REVENUE_STATUS].copy()
    LOGGER.info("计入实收的成交订单: %d 个", len(revenue))

    cells = build_cells(revenue, payments)
    detail = classify_cells(cells, orders, payments)
    diff = to_diff_table(detail)
    category_stats = build_category_stats(detail)

    orders_daily, payments_daily, calendar = build_daily_tables(revenue, payments)

    order_total = int(cells["order_cents"].sum())
    payment_total = int(cells["pay_cents"].sum())
    if int(orders_daily["gmv_cents"].sum()) != order_total:
        raise AssertionError("订单侧日粒度合计与对账单元合计不一致")
    if int(payments_daily["gmv_cents"].sum()) != payment_total:
        raise AssertionError("支付侧日粒度合计与对账单元合计不一致")

    summary = render_summary(cells, category_stats, calendar)

    orders_daily_out = _daily_output(orders_daily)
    payments_daily_out = _daily_output(payments_daily)

    orders_daily_path = outdir / "orders_daily.csv"
    payments_daily_path = outdir / "payments_daily.csv"
    diff_path = outdir / "diff.csv"
    summary_path = outdir / "summary.txt"

    _write_csv(orders_daily_out, orders_daily_path)
    _write_csv(payments_daily_out, payments_daily_path)
    _write_csv(diff, diff_path)
    summary_path.write_text(summary, encoding="utf-8")
    LOGGER.info("已写出 %s", summary_path)

    return ReconcileResult(
        orders_daily=orders_daily_out,
        payments_daily=payments_daily_out,
        diff=diff,
        category_stats=category_stats,
        order_total_cents=order_total,
        payment_total_cents=payment_total,
        diff_net_cents=order_total - payment_total,
        diff_abs_cents=int(detail["diff_cents"].abs().sum()),
        summary=summary,
        orders_daily_path=orders_daily_path,
        payments_daily_path=payments_daily_path,
        diff_path=diff_path,
        summary_path=summary_path,
    )


def _daily_output(daily: pd.DataFrame) -> pd.DataFrame:
    """把以「分」计的日粒度结果转成对外输出的元金额表。"""
    table = pd.DataFrame(
        {
            "date": daily["date"],
            "gmv": (daily["gmv_cents"] / 100).round(2),
            "mom_amount": (daily["mom_cents"] / 100).round(2),
            "mom_pct": daily["mom_pct"],
        }
    )
    return table[list(DAILY_COLUMNS)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="订单表与支付表按日对账：实收 GMV、日环比与差异归因。"
    )
    parser.add_argument(
        "--orders",
        type=Path,
        default=Path("data/orders.csv"),
        help="订单表 CSV 路径（默认 data/orders.csv）",
    )
    parser.add_argument(
        "--payments",
        type=Path,
        default=Path("data/payments.csv"),
        help="支付表 CSV 路径（默认 data/payments.csv）",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="输出目录（默认当前目录）",
    )
    parser.add_argument("--quiet", action="store_true", help="只输出汇总，不打印日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = run(args.orders, args.payments, args.outdir)
    except (OSError, ValueError) as error:
        LOGGER.error("对账失败: %s", error)
        return 1
    sys.stdout.write(result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
