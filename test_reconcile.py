"""reconcile.py 的测试：状态机、时区换算、日环比与四类归因。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import reconcile

DATA_DIR = Path(__file__).resolve().parent / "data"

ORDER_HEADER = "order_id,status,amount,updated_at\n"
PAYMENT_HEADER = "pay_id,order_id,amount,paid_at\n"


def write_inputs(
    tmp_path: Path, order_rows: list[str], payment_rows: list[str]
) -> tuple[Path, Path]:
    orders_path = tmp_path / "orders.csv"
    payments_path = tmp_path / "payments.csv"
    orders_path.write_text(
        ORDER_HEADER + "\n".join(order_rows) + "\n", encoding="utf-8"
    )
    payments_path.write_text(
        PAYMENT_HEADER + "\n".join(payment_rows) + "\n", encoding="utf-8"
    )
    return orders_path, payments_path


def run_case(
    tmp_path: Path, order_rows: list[str], payment_rows: list[str]
) -> reconcile.ReconcileResult:
    orders_path, payments_path = write_inputs(tmp_path, order_rows, payment_rows)
    return reconcile.run(orders_path, payments_path, tmp_path / "out")


def assert_single_category_per_row(diff: pd.DataFrame) -> None:
    assert list(diff.columns) == list(reconcile.DIFF_COLUMNS)
    assert diff["category"].notna().all()
    assert set(diff["category"]) <= set(reconcile.CATEGORIES)
    assert not diff.duplicated(subset=["date", "order_id"]).any()
    assert (diff["diff_amount"] != 0).all()


# ---------------------------------------------------------------------------
# 订单侧：状态机
# ---------------------------------------------------------------------------


def test_final_status_follows_priority_not_timestamp(tmp_path: Path) -> None:
    """乱序时（下单时间晚于成交）仍应按状态机取成交。"""
    rows = [
        "O1,下单,120.00,2026-10-01 18:00:00",
        "O1,成交,100.00,2026-10-01 10:00:00",
        "O1,改价,100.00,2026-10-01 09:00:00",
        "O2,下单,80.00,2026-10-02 09:00:00",
        "O2,改价,70.00,2026-10-02 10:00:00",
        "O3,取消,0.00,2026-10-03 09:00:00",
        "O4,成交,50.00,2026-10-04 09:00:00",
        "O4,取消,0.00,2026-10-04 12:00:00",
    ]
    orders_path, _ = write_inputs(tmp_path, rows, [])
    final = reconcile.final_status_rows(reconcile.load_orders(orders_path))
    final = final.set_index("order_id")

    assert final.loc["O1", "status"] == "成交"
    assert final.loc["O1", "amount_cents"] == 10000
    assert final.loc["O2", "status"] == "改价"
    assert final.loc["O3", "status"] == "取消"
    # 成交优先于取消：取消不能抹掉已成交订单的实收
    assert final.loc["O4", "status"] == "成交"
    assert final.loc["O4", "amount_cents"] == 5000


def test_unknown_status_is_rejected(tmp_path: Path) -> None:
    orders_path, _ = write_inputs(
        tmp_path, ["O1,已发货,10.00,2026-10-01 09:00:00"], []
    )
    with pytest.raises(ValueError, match="未定义的状态"):
        reconcile.load_orders(orders_path)


# ---------------------------------------------------------------------------
# 日粒度 GMV 与环比
# ---------------------------------------------------------------------------


def test_daily_gmv_and_mom(tmp_path: Path) -> None:
    """两天都有实收：环比 = 差额，百分比以昨日为基期。"""
    result = run_case(
        tmp_path,
        [
            "O1,成交,100.00,2026-10-01 12:00:00",
            "O2,成交,150.00,2026-10-02 12:00:00",
        ],
        [
            "P1,O1,100.00,2026-10-01 02:00:00+00:00",
            "P2,O2,150.00,2026-10-02 02:00:00+00:00",
        ],
    )
    orders_daily = result.orders_daily
    assert list(orders_daily.columns) == list(reconcile.DAILY_COLUMNS)
    assert list(orders_daily["gmv"]) == [100.0, 150.0]
    assert result.orders_daily["mom_amount"].iloc[0] != result.orders_daily[
        "mom_amount"
    ].iloc[0]  # 首日为 NaN
    assert orders_daily["mom_amount"].iloc[1] == 50.0
    assert orders_daily["mom_pct"].iloc[1] == 50.0
    assert result.payments_daily["gmv"].tolist() == [100.0, 150.0]
    assert result.diff.empty


def test_mom_pct_blank_when_previous_day_is_zero(tmp_path: Path) -> None:
    """日历由两侧数据共同决定：订单侧首日为 0，环比百分比留空。"""
    result = run_case(
        tmp_path,
        ["O1,成交,100.00,2026-10-02 12:00:00"],
        [
            "P9,O999,30.00,2026-10-01 02:00:00+00:00",
            "P1,O1,100.00,2026-10-02 02:00:00+00:00",
        ],
    )
    assert result.orders_daily["gmv"].tolist() == [0.0, 100.0]
    assert pd.isna(result.orders_daily["mom_pct"].iloc[0])
    assert result.orders_daily["mom_amount"].iloc[1] == 100.0
    assert pd.isna(result.orders_daily["mom_pct"].iloc[1])


def test_mom_pct_uses_absolute_base_for_negative_gmv(tmp_path: Path) -> None:
    result = run_case(
        tmp_path,
        ["O1,成交,100.00,2026-10-01 12:00:00"],
        [
            "P1,O1,-100.00,2026-10-01 02:00:00+00:00",
            "P2,O1,-50.00,2026-10-02 02:00:00+00:00",
        ],
    )
    payments_daily = result.payments_daily
    assert payments_daily["gmv"].tolist() == [-100.0, -50.0]
    assert payments_daily["mom_amount"].iloc[1] == 50.0
    assert payments_daily["mom_pct"].iloc[1] == 50.0


# ---------------------------------------------------------------------------
# 归因：四类互斥
# ---------------------------------------------------------------------------


def test_timezone_window_pair_is_timezone_category(tmp_path: Path) -> None:
    """支付 UTC 时刻落在次日北京：两侧各一条差异，归入时区窗口差且净额为 0。"""
    result = run_case(
        tmp_path,
        ["O1,成交,100.00,2026-10-01 17:00:00"],
        ["P1,O1,100.00,2026-10-01 20:00:00+00:00"],
    )
    diff = result.diff
    assert_single_category_per_row(diff)
    assert len(diff) == 2
    assert set(diff["category"]) == {reconcile.CATEGORY_TIMEZONE}
    assert diff["diff_amount"].tolist() == [100.0, -100.0]
    assert result.diff_net_cents == result.order_total_cents - result.payment_total_cents
    assert sum(
        stat["net_cents"] for stat in result.category_stats.to_dict("records")
        if stat["category"] == reconcile.CATEGORY_TIMEZONE
    ) == 0


def test_adjacent_day_payment_without_timezone_shift_is_unmatched(
    tmp_path: Path,
) -> None:
    """跨日但未发生时区换日（UTC 早时段）：不得误判为时区窗口差。"""
    result = run_case(
        tmp_path,
        ["O1,成交,100.00,2026-10-01 12:00:00"],
        ["P1,O1,100.00,2026-10-02 02:00:00+00:00"],
    )
    diff = result.diff
    assert_single_category_per_row(diff)
    assert set(diff["category"]) == {reconcile.CATEGORY_UNMATCHED}


def test_refund_sign_conflict_is_refund_category(tmp_path: Path) -> None:
    """订单侧记了负向冲抵、支付侧仍是正向收款：归入退款或负向冲抵。"""
    result = run_case(
        tmp_path,
        ["O1,成交,-50.00,2026-10-01 12:00:00"],
        ["P1,O1,50.00,2026-10-01 02:00:00+00:00"],
    )
    diff = result.diff
    assert_single_category_per_row(diff)
    assert diff["category"].tolist() == [reconcile.CATEGORY_REFUND]
    assert diff["diff_amount"].tolist() == [-100.0]


def test_mirrored_negative_amounts_reconcile_without_diff(tmp_path: Path) -> None:
    """两表都记负向冲抵：金额一致，不产生差异行。"""
    result = run_case(
        tmp_path,
        ["O1,成交,-50.00,2026-10-01 12:00:00"],
        ["P1,O1,-50.00,2026-10-01 02:00:00+00:00"],
    )
    assert result.diff.empty
    assert result.order_total_cents == result.payment_total_cents == -5000


def test_duplicate_payment_rows_are_flagged(tmp_path: Path) -> None:
    result = run_case(
        tmp_path,
        ["O1,成交,100.00,2026-10-01 12:00:00"],
        [
            "P1,O1,100.00,2026-10-01 02:00:00+00:00",
            "P1,O1,100.00,2026-10-01 02:00:00+00:00",
        ],
    )
    diff = result.diff
    assert_single_category_per_row(diff)
    assert set(diff["category"]) == {reconcile.CATEGORY_DUPLICATE}
    assert diff["diff_amount"].tolist() == [-100.0]


def test_multi_status_order_without_deal_is_flagged(tmp_path: Path) -> None:
    """有下单/改价但没有成交的订单：订单侧不计实收，支付侧有款，归入多状态行。"""
    result = run_case(
        tmp_path,
        [
            "O1,下单,100.00,2026-10-01 09:00:00",
            "O1,改价,90.00,2026-10-01 10:00:00",
        ],
        ["P1,O1,100.00,2026-10-01 02:00:00+00:00"],
    )
    diff = result.diff
    assert_single_category_per_row(diff)
    assert diff["category"].tolist() == [reconcile.CATEGORY_DUPLICATE]
    assert result.order_total_cents == 0
    assert diff["diff_amount"].tolist() == [-100.0]


def test_orphan_payment_and_missing_payment_are_unmatched(tmp_path: Path) -> None:
    result = run_case(
        tmp_path,
        ["O1,成交,100.00,2026-10-01 12:00:00"],
        ["P9,O999,80.00,2026-10-01 02:00:00+00:00"],
    )
    diff = result.diff
    assert_single_category_per_row(diff)
    rows = diff.set_index("order_id")
    assert rows.loc["O1", "category"] == reconcile.CATEGORY_UNMATCHED
    assert rows.loc["O999", "category"] == reconcile.CATEGORY_UNMATCHED
    assert rows.loc["O1", "diff_amount"] == 100.0
    assert rows.loc["O999", "diff_amount"] == -80.0


def test_split_payment_across_days_is_unmatched(tmp_path: Path) -> None:
    """分笔支付跨多日：两行差异都不能归入时区窗口差。"""
    result = run_case(
        tmp_path,
        ["O1,成交,100.00,2026-10-01 12:00:00"],
        [
            "P1,O1,60.00,2026-10-01 02:00:00+00:00",
            "P2,O1,40.00,2026-10-05 02:00:00+00:00",
        ],
    )
    diff = result.diff
    assert_single_category_per_row(diff)
    assert set(diff["category"]) == {reconcile.CATEGORY_UNMATCHED}
    assert diff["diff_amount"].tolist() == [40.0, -40.0]


def test_categories_are_mutually_exclusive_in_one_case(tmp_path: Path) -> None:
    """同一份数据里四类各自成行，且每行只有一个分类。"""
    result = run_case(
        tmp_path,
        [
            "O1,成交,100.00,2026-10-01 17:00:00",
            "O2,成交,-50.00,2026-10-02 12:00:00",
            "O3,下单,90.00,2026-10-03 09:00:00",
            "O3,改价,80.00,2026-10-03 10:00:00",
            "O4,成交,60.00,2026-10-04 12:00:00",
        ],
        [
            "P1,O1,100.00,2026-10-01 20:00:00+00:00",
            "P2,O2,50.00,2026-10-02 02:00:00+00:00",
            "P3,O3,90.00,2026-10-03 02:00:00+00:00",
            "P4,O999,30.00,2026-10-04 02:00:00+00:00",
        ],
    )
    diff = result.diff
    assert_single_category_per_row(diff)
    assert set(diff["category"]) == set(reconcile.CATEGORIES)
    assert result.category_stats["count"].sum() == len(diff)
    assert result.diff_net_cents == result.order_total_cents - result.payment_total_cents
    assert result.diff_abs_cents == int(diff["diff_amount"].abs().sum() * 100)


# ---------------------------------------------------------------------------
# 端到端：样例数据与 CLI
# ---------------------------------------------------------------------------


def test_sample_data_end_to_end(tmp_path: Path) -> None:
    result = reconcile.run(
        DATA_DIR / "orders.csv", DATA_DIR / "payments.csv", tmp_path / "out"
    )
    diff = result.diff
    assert_single_category_per_row(diff)

    assert len(result.orders_daily) == 30
    assert result.orders_daily["date"].iloc[0] == pd.Timestamp("2026-10-01").date()
    assert pd.isna(result.orders_daily["mom_pct"].iloc[0])
    assert result.orders_daily["gmv"].sum() == pytest.approx(
        result.order_total_cents / 100
    )
    assert result.payments_daily["gmv"].sum() == pytest.approx(
        result.payment_total_cents / 100
    )
    assert result.diff_net_cents == (
        result.order_total_cents - result.payment_total_cents
    )
    assert result.category_stats["count"].sum() == len(diff)
    assert set(diff["category"]) == set(reconcile.CATEGORIES)
    assert (diff["order_amount"] - diff["payment_amount"]).round(2).equals(
        diff["diff_amount"]
    )

    for path in (
        result.diff_path,
        result.summary_path,
        result.orders_daily_path,
        result.payments_daily_path,
    ):
        assert path.is_file() and path.stat().st_size > 0
    assert "对账差异净额" in result.summary


def test_main_writes_outputs_and_returns_zero(tmp_path: Path, capsys) -> None:
    exit_code = reconcile.main(
        [
            "--orders",
            str(DATA_DIR / "orders.csv"),
            "--payments",
            str(DATA_DIR / "payments.csv"),
            "--outdir",
            str(tmp_path / "cli-out"),
            "--quiet",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "订单侧实收GMV" in captured
    assert (tmp_path / "cli-out" / "diff.csv").is_file()
    assert (tmp_path / "cli-out" / "summary.txt").is_file()


def test_main_returns_error_on_missing_input(tmp_path: Path) -> None:
    exit_code = reconcile.main(
        [
            "--orders",
            str(tmp_path / "nope.csv"),
            "--payments",
            str(DATA_DIR / "payments.csv"),
            "--outdir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 1


def test_bad_amount_is_rejected(tmp_path: Path) -> None:
    orders_path, _ = write_inputs(
        tmp_path, ["O1,成交,abc,2026-10-01 09:00:00"], []
    )
    with pytest.raises(ValueError, match="无法解析的金额"):
        reconcile.load_orders(orders_path)
