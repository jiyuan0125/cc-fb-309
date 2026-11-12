#!/usr/bin/env python3
"""订单表 / 支付表日粒度实收 GMV 对账。

口径摘要：
- 订单表时间戳为运营系统北京时间（无时区标记）；支付表时间戳为 UTC。
  两侧统一换算成北京时间的自然日再聚合。
- 同一订单有多行状态记录，终态按状态机取：
  成交 > 改价 > 下单 > 取消；同优先级时取 updated_at 最新的一行。
- 实收 = 订单终态净额（取消为 0，负数表示退款 / 冲抵）。
- 对账最小单元为 (北京自然日, 订单号)，差异 = 订单口径 - 支付口径。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("reconcile")

CN_TZ = "Asia/Shanghai"
STATUS_PRIORITY: dict[str, int] = {"成交": 4, "改价": 3, "下单": 2, "取消": 1}
STATUS_CANCELLED = "取消"
ORDER_REQUIRED_COLUMNS = ("order_id", "status", "amount", "updated_at")
PAYMENT_REQUIRED_COLUMNS = ("pay_id", "order_id", "amount", "paid_at")

# 金额精度 1 分，比较时用半分容差；低于此值视为对平。
EPS = 0.005

CATEGORY_TZ = "时区窗口差"
CATEGORY_REFUND = "退款或负向冲抵"
CATEGORY_MULTI = "重复或多状态行"
CATEGORY_OTHER = "其余未对上"
CATEGORIES = (CATEGORY_TZ, CATEGORY_REFUND, CATEGORY_MULTI, CATEGORY_OTHER)

DIFF_COLUMNS = (
    "date",
    "order_id",
    "order_status",
    "order_amount",
    "payment_amount",
    "diff_amount",
    "category",
    "detail",
)


@dataclass(frozen=True)
class ReconcileResult:
    """一次对账的完整结果。"""

    orders_daily: pd.DataFrame
    payments_daily: pd.DataFrame
    diff: pd.DataFrame
    category_stats: pd.DataFrame
    order_total: float
    payment_total: float
    diff_total: float
    order_row_count: int
    payment_row_count: int
    order_count: int
    payment_order_count: int
    window_start: date
    window_end: date


def _require_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} 缺少必需列 {missing}；实际列为 {list(df.columns)}")


def _read_csv(path: Path, required: Sequence[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label}文件不存在: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    _require_columns(df, required, label)
    if df.empty:
        LOGGER.warning("%s为空: %s", label, path)
    return df


def _parse_amount(series: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        samples = list(series[bad].head(5))
        raise ValueError(f"{label} 有 {int(bad.sum())} 行金额无法解析，示例: {samples}")
    return parsed.astype(float)


def _parse_order_timestamps(series: pd.Series, label: str) -> pd.Series:
    """解析订单表时间戳，返回无时区的北京时间。

    订单表的裸时间戳本身就是运营系统的北京时间；若后续数据带时区标记，
    按标记换算到北京时间，避免二次偏移。
    """
    parsed = pd.to_datetime(series, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        LOGGER.info("%s 时间戳带时区，已换算为 %s 本地时间", label, CN_TZ)
        parsed = parsed.dt.tz_convert(CN_TZ).dt.tz_localize(None)
    bad = parsed.isna()
    if bad.any():
        samples = list(series[bad].head(5))
        raise ValueError(
            f"{label} 有 {int(bad.sum())} 行时间戳无法解析，示例: {samples}"
        )
    return parsed


def load_orders(path: Path) -> pd.DataFrame:
    """读取订单状态记录，校验字段、时间与状态枚举。"""
    df = _read_csv(path, ORDER_REQUIRED_COLUMNS, "订单表")
    df["amount"] = _parse_amount(df["amount"], "订单表")
    df["updated_at"] = _parse_order_timestamps(df["updated_at"], "订单表")
    unknown = sorted(set(df["status"]) - set(STATUS_PRIORITY))
    if unknown:
        raise ValueError(f"订单表出现状态机外的状态: {unknown}")
    LOGGER.info(
        "订单表载入 %d 行，%d 个订单，时间范围 %s ~ %s（北京时间）",
        len(df),
        df["order_id"].nunique(),
        df["updated_at"].min(),
        df["updated_at"].max(),
    )
    return df


def load_payments(path: Path) -> pd.DataFrame:
    """读取支付流水，换算为北京时间并派生日。"""
    df = _read_csv(path, PAYMENT_REQUIRED_COLUMNS, "支付表")
    df["amount"] = _parse_amount(df["amount"], "支付表")
    df["paid_at_utc"] = pd.to_datetime(df["paid_at"], errors="coerce", utc=True)
    bad = df["paid_at_utc"].isna()
    if bad.any():
        samples = list(df.loc[bad, "paid_at"].head(5))
        raise ValueError(f"支付表有 {int(bad.sum())} 行时间戳无法解析，示例: {samples}")
    df["paid_at_bj"] = df["paid_at_utc"].dt.tz_convert(CN_TZ).dt.tz_localize(None)
    df["local_date"] = df["paid_at_bj"].dt.normalize()
    if df["pay_id"].duplicated().any():
        duplicated = df.loc[df["pay_id"].duplicated(), "pay_id"].head(5).tolist()
        LOGGER.warning("支付表存在重复 pay_id（按流水行处理）: %s", duplicated)
    LOGGER.info(
        "支付表载入 %d 笔流水，%d 个订单，时间范围 %s ~ %s（北京时间）",
        len(df),
        df["order_id"].nunique(),
        df["paid_at_bj"].min(),
        df["paid_at_bj"].max(),
    )
    return df


def build_final_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """按状态机优先级选出每个订单的终态行。

    优先级 成交 > 改价 > 下单 > 取消；同优先级取 updated_at 最新；
    时间戳相同则取文件中靠后的一行，保证结果可重复。
    """
    df = orders.copy()
    df["_priority"] = df["status"].map(STATUS_PRIORITY)
    df["_seq"] = range(len(df))
    row_counts = df.groupby("order_id").size().rename("order_row_count")
    df = df.sort_values(
        ["order_id", "_priority", "updated_at", "_seq"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    final = df.drop_duplicates(subset="order_id", keep="first").copy()
    final["order_row_count"] = final["order_id"].map(row_counts).astype(int)
    final["local_date"] = final["updated_at"].dt.normalize()
    LOGGER.info(
        "终态订单 %d 个（原始 %d 行），终态分布: %s",
        len(final),
        len(orders),
        final["status"].value_counts().to_dict(),
    )
    return final[
        ["order_id", "status", "amount", "updated_at", "local_date", "order_row_count"]
    ].reset_index(drop=True)


def _daily_gmv(
    grouped: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """把按日聚合的 GMV 补齐日历日并计算环比（百分比，前一日为 0 时留空）。"""
    calendar = pd.date_range(start, end, freq="D")
    gmv = grouped.reindex(calendar, fill_value=0.0).astype(float)
    previous = gmv.shift(1)
    momentum = (gmv / previous - 1.0) * 100.0
    momentum = momentum.where(previous.abs() > EPS)
    frame = pd.DataFrame(
        {
            "date": calendar.strftime("%Y-%m-%d"),
            "gmv": gmv.round(2).to_numpy(),
            "prev_gmv": previous.round(2).to_numpy(),
            "mom_pct": momentum.round(2).to_numpy(),
        }
    )
    return frame


def _unique_days(series: pd.Series) -> tuple[date, ...]:
    return tuple(sorted({timestamp.date() for timestamp in series}))


def _any_negative(series: pd.Series) -> bool:
    return bool((series < -EPS).any())


def _classify_row(row: Mapping[str, Any]) -> tuple[str, str]:
    """对一条 (日, 订单) 差异行归因，四类互斥。

    归因看的是订单级上下文（订单是否存在、是否存在支付、各有几行），
    而不是当天的单边存在性，因此跨日的一对差异行会得到同一个分类。
    判定优先级：时区窗口差 > 退款或负向冲抵 > 重复或多状态行 > 其余未对上。
    """
    has_order = pd.notna(row["order_total"])
    has_payment = pd.notna(row["pay_total"])
    pay_days = row["pay_days"] if isinstance(row["pay_days"], tuple) else ()
    order_day = row["order_day"].date() if has_order else None

    # 1. 时区窗口差：同一笔金额，订单与支付落在相邻的北京时间自然日。
    if (
        has_order
        and has_payment
        and len(pay_days) == 1
        and abs(row["order_total"] - row["pay_total"]) <= EPS
    ):
        gap = (pay_days[0] - order_day).days
        if abs(gap) == 1:
            return CATEGORY_TZ, (
                f"订单终态日 {order_day}，支付换算北京时间后归属 {pay_days[0]}，"
                f"相差 {abs(gap)} 天且金额一致，为日窗口错位"
            )

    # 2. 退款或负向冲抵：取消单仍有收款，或任一侧出现负金额。
    refund_reasons: list[str] = []
    if has_order and has_payment and row["order_status"] == STATUS_CANCELLED:
        refund_reasons.append(
            f"终态为取消但支付表仍有 {int(row['pay_rows'])} 笔收款，订单净额按 0 计"
        )
    if has_order and row["order_total"] < -EPS:
        refund_reasons.append(
            f"订单终态净额为负（{row['order_total']:.2f}），按退款 / 冲抵处理"
        )
    if has_payment and bool(row["pay_has_negative"]):
        refund_reasons.append("支付流水含负金额（退款 / 冲抵）")
    if refund_reasons:
        return CATEGORY_REFUND, "；".join(refund_reasons)

    # 3. 重复或多状态行：订单多状态行金额不一致，或同一订单拆成多笔支付。
    multi_reasons: list[str] = []
    if has_order and int(row["order_row_total"]) > 1:
        multi_reasons.append(
            f"订单存在 {int(row['order_row_total'])} 行状态记录"
            f"（终态 {row['order_status']}），终态金额与支付口径不一致"
        )
    if has_payment and int(row["pay_rows"]) > 1:
        span = f"，分布于 {len(pay_days)} 个自然日" if len(pay_days) > 1 else ""
        multi_reasons.append(f"同一订单拆成 {int(row['pay_rows'])} 笔支付{span}")
    if multi_reasons:
        return CATEGORY_MULTI, "；".join(multi_reasons)

    # 4. 其余未对上：单边缺失或无法归入前三类。
    if not has_order:
        return CATEGORY_OTHER, "支付表有流水，订单表无对应订单"
    if not has_payment:
        return CATEGORY_OTHER, "订单表有终态，支付表无对应流水"
    return CATEGORY_OTHER, "两侧均有记录，但金额差异无法归入前三类"


def _order_context(final_orders: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    """按订单汇总对账所需的上下文（订单总额、支付总额 / 笔数 / 日集合）。"""
    orders_ctx = final_orders.set_index("order_id")[
        ["status", "amount", "local_date", "order_row_count"]
    ].rename(
        columns={
            "status": "order_status",
            "amount": "order_total",
            "local_date": "order_day",
            "order_row_count": "order_row_total",
        }
    )
    payments_ctx = payments.groupby("order_id").agg(
        pay_total=("amount", "sum"),
        pay_rows=("amount", "size"),
        pay_days=("local_date", _unique_days),
        pay_has_negative=("amount", _any_negative),
    )
    return orders_ctx.join(payments_ctx, how="outer")


def reconcile(final_orders: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    """按 (北京日, 订单) 对账，返回差异明细（已归因、按日期与订单排序）。"""
    order_side = final_orders[["order_id", "local_date", "status", "amount"]].rename(
        columns={"amount": "order_amount"}
    )
    payment_side = payments.groupby(["order_id", "local_date"], as_index=False).agg(
        payment_amount=("amount", "sum"), payment_row_count=("amount", "size")
    )
    merged = order_side.merge(payment_side, on=["order_id", "local_date"], how="outer")
    merged["order_amount"] = merged["order_amount"].fillna(0.0).astype(float)
    merged["payment_amount"] = merged["payment_amount"].fillna(0.0).astype(float)
    merged["diff_amount"] = merged["order_amount"] - merged["payment_amount"]

    context = _order_context(final_orders, payments)
    merged = merged.merge(context, left_on="order_id", right_index=True, how="left")

    diff = merged[merged["diff_amount"].abs() > EPS].copy()
    if diff.empty:
        LOGGER.info("对账无差异：两侧逐日逐单完全对上")
        diff = diff.assign(
            category=pd.Series(dtype="object"), detail=pd.Series(dtype="object")
        )
        return diff
    labels = diff.apply(_classify_row, axis=1)
    diff["category"] = [label[0] for label in labels]
    diff["detail"] = [label[1] for label in labels]
    diff = diff.sort_values(["local_date", "order_id"], kind="mergesort")
    LOGGER.info(
        "对账差异 %d 行，分类分布: %s",
        len(diff),
        diff["category"].value_counts().to_dict(),
    )
    return diff


def _category_stats(diff: pd.DataFrame) -> pd.DataFrame:
    if diff.empty:
        stats = pd.DataFrame(
            {"rows": 0, "net_amount": 0.0, "abs_amount": 0.0}, index=CATEGORIES
        )
    else:
        stats = diff.groupby("category").agg(
            rows=("diff_amount", "size"),
            net_amount=("diff_amount", "sum"),
            abs_amount=("diff_amount", lambda values: values.abs().sum()),
        )
        stats = stats.reindex(CATEGORIES, fill_value=0)
    stats["rows"] = stats["rows"].astype(int)
    stats["net_amount"] = stats["net_amount"].round(2)
    stats["abs_amount"] = stats["abs_amount"].round(2)
    return stats


def _format_diff(diff: pd.DataFrame) -> pd.DataFrame:
    if diff.empty:
        return pd.DataFrame(columns=DIFF_COLUMNS)
    output = pd.DataFrame(
        {
            "date": diff["local_date"].dt.strftime("%Y-%m-%d"),
            "order_id": diff["order_id"],
            "order_status": diff["order_status"].fillna(""),
            "order_amount": diff["order_amount"].round(2),
            "payment_amount": diff["payment_amount"].round(2),
            "diff_amount": diff["diff_amount"].round(2),
            "category": diff["category"],
            "detail": diff["detail"],
        }
    )
    return output[list(DIFF_COLUMNS)].reset_index(drop=True)


def run(orders_path: Path, payments_path: Path, out_dir: Path) -> ReconcileResult:
    """执行完整对账流程并把结果写入 out_dir。"""
    orders = load_orders(orders_path)
    payments = load_payments(payments_path)
    if orders.empty or payments.empty:
        raise ValueError(
            f"输入数据为空：订单表 {len(orders)} 行，支付表 {len(payments)} 行，"
            "无法确定对账窗口"
        )
    final_orders = build_final_orders(orders)

    window_start = min(final_orders["local_date"].min(), payments["local_date"].min())
    window_end = max(final_orders["local_date"].max(), payments["local_date"].max())
    if window_start > window_end:
        raise ValueError(
            f"数据窗口异常: 起始 {window_start} 晚于结束 {window_end}，请检查输入"
        )

    orders_daily = _daily_gmv(
        final_orders.groupby("local_date")["amount"].sum(), window_start, window_end
    )
    payments_daily = _daily_gmv(
        payments.groupby("local_date")["amount"].sum(), window_start, window_end
    )

    diff = reconcile(final_orders, payments)
    stats = _category_stats(diff)

    order_total = round(float(final_orders["amount"].sum()), 2)
    payment_total = round(float(payments["amount"].sum()), 2)
    diff_total = round(order_total - payment_total, 2)
    row_total = round(float(diff["diff_amount"].sum()) if not diff.empty else 0.0, 2)
    if abs(row_total - diff_total) > 0.01:
        raise RuntimeError(
            "差异明细汇总与两侧总额不一致，说明归集逻辑有漏项："
            f"明细合计 {row_total:.2f}，总额差 {diff_total:.2f}"
        )
    daily_total = round(float(orders_daily["gmv"].sum()), 2)
    if abs(daily_total - order_total) > 0.01:
        raise RuntimeError(
            f"订单日粒度合计 {daily_total:.2f} 与订单总额 {order_total:.2f} 不一致"
        )

    result = ReconcileResult(
        orders_daily=orders_daily,
        payments_daily=payments_daily,
        diff=diff,
        category_stats=stats,
        order_total=order_total,
        payment_total=payment_total,
        diff_total=diff_total,
        order_row_count=len(orders),
        payment_row_count=len(payments),
        order_count=len(final_orders),
        payment_order_count=int(payments["order_id"].nunique()),
        window_start=window_start.date(),
        window_end=window_end.date(),
    )
    _write_outputs(result, out_dir, orders_path, payments_path)
    return result


def _build_summary(
    result: ReconcileResult, orders_path: Path, payments_path: Path
) -> str:
    days = len(result.orders_daily)
    diff_days = (
        int(result.diff["local_date"].dt.normalize().nunique())
        if not result.diff.empty
        else 0
    )
    sign_note = "支付侧多" if result.diff_total < 0 else "订单侧多"
    lines = [
        "=" * 68,
        "订单 / 支付 日粒度实收对账报告",
        "=" * 68,
        (
            f"数据窗口(北京时间)   : {result.window_start} ~ {result.window_end}"
            f"（{days} 天）"
        ),
        (
            f"订单表               : {orders_path}"
            f"（{result.order_row_count} 行状态记录，{result.order_count} 个订单终态）"
        ),
        (
            f"支付表               : {payments_path}"
            f"（{result.payment_row_count} 笔流水，{result.payment_order_count} 个订单）"
        ),
        (
            "口径                 : 实收 = 订单终态净额；"
            "支付 UTC→北京时间归日；差异 = 订单口径 − 支付口径"
        ),
        "",
        "一、两边口径总额",
        f"  订单口径实收GMV     : {result.order_total:,.2f}",
        f"  支付口径实收GMV     : {result.payment_total:,.2f}",
        (
            f"  对账差异总额        : {result.diff_total:,.2f}"
            f"（{sign_note}，绝对额 {abs(result.diff_total):,.2f}）"
        ),
        f"  差异明细行数        : {len(result.diff)}",
        f"  出现差异的日数      : {diff_days} / {days}",
        "",
        "二、差异归因（四类互斥，每行仅一类）",
        f"  {'分类':<12}{'笔数':>8}{'差异净额':>16}{'绝对差异':>16}",
        "  " + "-" * 50,
    ]
    for category, row in result.category_stats.iterrows():
        lines.append(
            f"  {category:<12}{int(row['rows']):>8d}"
            f"{row['net_amount']:>16,.2f}{row['abs_amount']:>16,.2f}"
        )
    lines.append("  " + "-" * 50)
    lines.append(
        f"  {'合计':<12}{int(result.category_stats['rows'].sum()):>8d}"
        f"{result.category_stats['net_amount'].sum():>16,.2f}"
        f"{result.category_stats['abs_amount'].sum():>16,.2f}"
    )
    lines.extend(
        [
            "",
            "说明：",
            (
                "  1. 时区窗口差为同一笔金额在相邻两日各记一次，跨日净额相互抵消，"
                "故类别净额为 0；按日看两侧仍存在时点错位。"
            ),
            "  2. 退款或负向冲抵包含：终态取消但支付表仍有收款、订单/支付任一侧出现负金额。",
            (
                "  3. 重复或多状态行包含：订单多状态行导致终态金额与支付不一致、"
                "同一订单拆成多笔支付。"
            ),
            "  4. 其余未对上为单边缺失（有单无款 / 有款无单）等无法归入前三类的差异。",
            "  5. 差异净额符号 = 订单口径 − 支付口径；负值表示支付侧金额更大。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(
    result: ReconcileResult,
    out_dir: Path,
    orders_path: Path,
    payments_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    orders_daily_path = out_dir / "orders_daily.csv"
    payments_daily_path = out_dir / "payments_daily.csv"
    diff_path = out_dir / "diff.csv"
    summary_path = out_dir / "summary.txt"

    result.orders_daily.to_csv(
        orders_daily_path, index=False, encoding="utf-8-sig", float_format="%.2f"
    )
    result.payments_daily.to_csv(
        payments_daily_path, index=False, encoding="utf-8-sig", float_format="%.2f"
    )
    _format_diff(result.diff).to_csv(
        diff_path, index=False, encoding="utf-8-sig", float_format="%.2f"
    )
    summary_path.write_text(
        _build_summary(result, orders_path, payments_path), encoding="utf-8"
    )
    LOGGER.info(
        "结果已写出: %s, %s, %s, %s",
        orders_daily_path,
        payments_daily_path,
        diff_path,
        summary_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="订单/支付日粒度实收 GMV 对账")
    parser.add_argument(
        "--orders",
        type=Path,
        default=Path("data/orders.csv"),
        help="订单表路径（默认 data/orders.csv）",
    )
    parser.add_argument(
        "--payments",
        type=Path,
        default=Path("data/payments.csv"),
        help="支付表路径（默认 data/payments.csv）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="输出目录（默认仓库根目录）",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志级别",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        result = run(args.orders, args.payments, args.out_dir)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("对账失败: %s", exc)
        return 2
    except Exception:
        LOGGER.exception("对账过程中出现未预期错误")
        return 1

    print(f"订单口径实收GMV: {result.order_total:,.2f}")
    print(f"支付口径实收GMV: {result.payment_total:,.2f}")
    print(f"对账差异总额(订单-支付): {result.diff_total:,.2f}")
    print(f"差异明细 {len(result.diff)} 行，已写出 diff.csv / summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
