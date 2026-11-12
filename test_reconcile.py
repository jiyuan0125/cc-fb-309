"""reconcile.py 的单元测试与端到端测试。

覆盖：状态机终态选择、UTC→北京时间归日、日粒度 GMV 与环比、
四类差异互斥归因、总额勾稽，以及样例数据的端到端冒烟。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reconcile

REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_ORDERS = REPO_ROOT / "data" / "orders.csv"
SAMPLE_PAYMENTS = REPO_ROOT / "data" / "payments.csv"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def _run(
    workspace: Path,
    order_rows: list[dict[str, str]],
    payment_rows: list[dict[str, str]],
) -> reconcile.ReconcileResult:
    workspace.mkdir(parents=True, exist_ok=True)
    orders_path = _write_csv(workspace / "orders.csv", order_rows)
    payments_path = _write_csv(workspace / "payments.csv", payment_rows)
    out_dir = workspace / "out"
    return reconcile.run(orders_path, payments_path, out_dir)


def _order(
    order_id: str, status: str, amount: float, updated_at: str
) -> dict[str, str]:
    return {
        "order_id": order_id,
        "status": status,
        "amount": f"{amount:.2f}",
        "updated_at": updated_at,
    }


def _payment(pay_id: str, order_id: str, amount: float, paid_at: str) -> dict[str, str]:
    return {
        "pay_id": pay_id,
        "order_id": order_id,
        "amount": f"{amount:.2f}",
        "paid_at": paid_at,
    }


def _diff_row(result: reconcile.ReconcileResult, order_id: str, date: str) -> pd.Series:
    picked = result.diff[
        (result.diff["order_id"] == order_id)
        & (result.diff["local_date"].dt.strftime("%Y-%m-%d") == date)
    ]
    assert len(picked) == 1, f"{date} {order_id} 应恰好一行差异，实际 {len(picked)} 行"
    return picked.iloc[0]


class TestFinalStatus:
    """状态机优先级与乱序容错。"""

    def test_priority_beats_timestamp(self, workspace: Path) -> None:
        # 成交之后又落了取消，但成交优先级更高，终态仍为成交。
        result = _run(
            workspace,
            [
                _order("O1", "成交", 100.0, "2026-10-01 09:00:00"),
                _order("O1", "取消", 0.0, "2026-10-01 20:00:00"),
            ],
            [_payment("P1", "O1", 100.0, "2026-10-01 01:00:00+00:00")],
        )
        assert result.order_total == pytest.approx(100.0)
        assert result.diff.empty

    def test_repricing_outranks_order_creation(self, workspace: Path) -> None:
        result = _run(
            workspace,
            [
                _order("O1", "下单", 88.0, "2026-10-01 10:00:00"),
                _order("O1", "改价", 99.0, "2026-10-01 14:00:00"),
            ],
            [_payment("P1", "O1", 99.0, "2026-10-01 02:00:00+00:00")],
        )
        assert result.order_total == pytest.approx(99.0)

    def test_duplicate_status_uses_latest_row(self, workspace: Path) -> None:
        result = _run(
            workspace,
            [
                _order("O1", "成交", 120.0, "2026-10-01 09:00:00"),
                _order("O1", "成交", 130.0, "2026-10-01 15:00:00"),
            ],
            [_payment("P1", "O1", 130.0, "2026-10-01 03:00:00+00:00")],
        )
        assert result.order_total == pytest.approx(130.0)

    def test_out_of_order_rows_do_not_change_result(self, workspace: Path) -> None:
        ordered = [
            _order("O1", "下单", 100.0, "2026-10-01 09:00:00"),
            _order("O1", "成交", 100.0, "2026-10-01 11:00:00"),
        ]
        shuffled = [ordered[1], ordered[0]]
        pays = [_payment("P1", "O1", 100.0, "2026-10-01 03:00:00+00:00")]
        first = _run(workspace / "a", ordered, pays)
        second = _run(workspace / "b", shuffled, pays)
        pd.testing.assert_frame_equal(first.orders_daily, second.orders_daily)
        pd.testing.assert_frame_equal(first.diff, second.diff)

    def test_unknown_status_is_rejected(self, workspace: Path) -> None:
        orders_path = _write_csv(
            workspace / "orders.csv",
            [_order("O1", "已发货", 100.0, "2026-10-01 09:00:00")],
        )
        payments_path = _write_csv(
            workspace / "payments.csv",
            [_payment("P1", "O1", 100.0, "2026-10-01 01:00:00+00:00")],
        )
        with pytest.raises(ValueError, match="状态机外"):
            reconcile.run(orders_path, payments_path, workspace / "out")


class TestTimezone:
    """支付 UTC 时间换算为北京时间后归日。"""

    def test_utc_payment_crossing_midnight_is_next_beijing_day(
        self, workspace: Path
    ) -> None:
        # 10-01 16:30 UTC == 10-02 00:30 北京时间。
        result = _run(
            workspace,
            [_order("O1", "成交", 100.0, "2026-10-01 20:00:00")],
            [_payment("P1", "O1", 100.0, "2026-10-01 16:30:00+00:00")],
        )
        payments_daily = result.payments_daily.set_index("date")["gmv"]
        assert payments_daily["2026-10-01"] == pytest.approx(0.0)
        assert payments_daily["2026-10-02"] == pytest.approx(100.0)
        assert set(result.diff["category"]) == {reconcile.CATEGORY_TZ}

    def test_order_timestamp_is_already_beijing(self, workspace: Path) -> None:
        # 订单 10-01 09:30 北京时间，不应被当成 UTC 再偏移 8 小时。
        result = _run(
            workspace,
            [_order("O1", "成交", 100.0, "2026-10-01 09:30:00")],
            [_payment("P1", "O1", 100.0, "2026-10-01 01:30:00+00:00")],
        )
        orders_daily = result.orders_daily.set_index("date")["gmv"]
        assert orders_daily["2026-10-01"] == pytest.approx(100.0)
        assert result.diff.empty


class TestDailyGmv:
    """日粒度 GMV 与环比。"""

    def test_missing_day_is_filled_with_zero(self, workspace: Path) -> None:
        result = _run(
            workspace,
            [_order("O1", "成交", 100.0, "2026-10-01 10:00:00")],
            [_payment("P1", "O1", 100.0, "2026-10-01 02:00:00+00:00")],
        )
        assert list(result.orders_daily["date"]) == ["2026-10-01"]
        assert len(result.orders_daily) == 1

    def test_momentum_uses_previous_day(self, workspace: Path) -> None:
        result = _run(
            workspace,
            [
                _order("O1", "成交", 100.0, "2026-10-01 10:00:00"),
                _order("O2", "成交", 150.0, "2026-10-02 10:00:00"),
            ],
            [
                _payment("P1", "O1", 100.0, "2026-10-01 02:00:00+00:00"),
                _payment("P2", "O2", 150.0, "2026-10-02 02:00:00+00:00"),
            ],
        )
        daily = result.orders_daily.set_index("date")
        assert pd.isna(daily.loc["2026-10-01", "mom_pct"])
        assert daily.loc["2026-10-02", "mom_pct"] == pytest.approx(50.0)
        assert daily.loc["2026-10-02", "prev_gmv"] == pytest.approx(100.0)

    def test_momentum_blank_when_previous_day_is_zero(self, workspace: Path) -> None:
        result = _run(
            workspace,
            [
                _order("O1", "成交", 100.0, "2026-10-01 10:00:00"),
                _order("O2", "成交", 50.0, "2026-10-03 10:00:00"),
            ],
            [
                _payment("P1", "O1", 100.0, "2026-10-01 02:00:00+00:00"),
                _payment("P2", "O2", 50.0, "2026-10-02 02:00:00+00:00"),
            ],
        )
        daily = result.orders_daily.set_index("date")
        assert list(daily.index) == ["2026-10-01", "2026-10-02", "2026-10-03"]
        assert daily.loc["2026-10-03", "prev_gmv"] == pytest.approx(0.0)
        assert pd.isna(daily.loc["2026-10-03", "mom_pct"])


class TestClassification:
    """四类差异互斥归因。"""

    def test_refund_by_negative_order_amount(self, workspace: Path) -> None:
        result = _run(
            workspace,
            [_order("O1", "成交", -80.0, "2026-10-01 10:00:00")],
            [_payment("P1", "O1", 80.0, "2026-10-01 02:00:00+00:00")],
        )
        row = _diff_row(result, "O1", "2026-10-01")
        assert row["category"] == reconcile.CATEGORY_REFUND

    def test_refund_by_negative_payment_amount(self, workspace: Path) -> None:
        # 订单 100，支付一笔退款 −50 加一笔 100，净 50，差异 50 且支付侧含负数。
        result = _run(
            workspace,
            [_order("O1", "成交", 100.0, "2026-10-01 10:00:00")],
            [
                _payment("P1", "O1", -50.0, "2026-10-01 02:00:00+00:00"),
                _payment("P2", "O1", 100.0, "2026-10-01 05:00:00+00:00"),
            ],
        )
        row = _diff_row(result, "O1", "2026-10-01")
        assert row["category"] == reconcile.CATEGORY_REFUND
        assert "负金额" in row["detail"]

    def test_cancelled_with_payment_is_refund(self, workspace: Path) -> None:
        result = _run(
            workspace,
            [_order("O1", "取消", 0.0, "2026-10-01 12:00:00")],
            [_payment("P1", "O1", 100.0, "2026-10-01 02:00:00+00:00")],
        )
        row = _diff_row(result, "O1", "2026-10-01")
        assert row["category"] == reconcile.CATEGORY_REFUND
        assert "取消" in row["detail"]

    def test_split_payment_is_multi_row(self, workspace: Path) -> None:
        result = _run(
            workspace,
            [_order("O1", "成交", 100.0, "2026-10-01 10:00:00")],
            [
                _payment("P1", "O1", 60.0, "2026-10-01 01:00:00+00:00"),
                _payment("P2", "O1", 30.0, "2026-10-01 04:00:00+00:00"),
            ],
        )
        row = _diff_row(result, "O1", "2026-10-01")
        assert row["category"] == reconcile.CATEGORY_MULTI

    def test_repricing_mismatch_is_multi_row(self, workspace: Path) -> None:
        result = _run(
            workspace,
            [
                _order("O1", "下单", 90.0, "2026-10-01 09:00:00"),
                _order("O1", "改价", 110.0, "2026-10-01 12:00:00"),
            ],
            [_payment("P1", "O1", 90.0, "2026-10-01 02:00:00+00:00")],
        )
        row = _diff_row(result, "O1", "2026-10-01")
        assert row["category"] == reconcile.CATEGORY_MULTI

    def test_orphan_payment_and_orphan_order_are_other(self, workspace: Path) -> None:
        result = _run(
            workspace,
            [_order("O1", "成交", 100.0, "2026-10-01 10:00:00")],
            [_payment("P1", "O2", 70.0, "2026-10-01 02:00:00+00:00")],
        )
        assert _diff_row(result, "O1", "2026-10-01")["category"] == (
            reconcile.CATEGORY_OTHER
        )
        assert _diff_row(result, "O2", "2026-10-01")["category"] == (
            reconcile.CATEGORY_OTHER
        )

    def test_categories_are_mutually_exclusive(self, workspace: Path) -> None:
        # 同一行同时有多笔支付且金额为负，只能落在一个类别里。
        result = _run(
            workspace,
            [
                _order("O1", "下单", 100.0, "2026-10-01 09:00:00"),
                _order("O1", "改价", -20.0, "2026-10-01 12:00:00"),
            ],
            [
                _payment("P1", "O1", 80.0, "2026-10-01 02:00:00+00:00"),
                _payment("P2", "O1", 20.0, "2026-10-01 05:00:00+00:00"),
            ],
        )
        assert len(result.diff) == 1
        row = _diff_row(result, "O1", "2026-10-01")
        assert row["category"] == reconcile.CATEGORY_REFUND
        assert result.category_stats["rows"].sum() == len(result.diff)


class TestTotals:
    """总额勾稽与输出文件。"""

    def test_diff_detail_sums_to_gap(self, workspace: Path) -> None:
        result = _run(
            workspace,
            [
                _order("O1", "成交", 100.0, "2026-10-01 10:00:00"),
                _order("O2", "成交", 200.0, "2026-10-01 12:00:00"),
                _order("O3", "取消", 0.0, "2026-10-02 12:00:00"),
            ],
            [
                _payment("P1", "O1", 100.0, "2026-10-01 02:00:00+00:00"),
                _payment("P2", "O2", 190.0, "2026-10-01 05:00:00+00:00"),
                _payment("P3", "O3", 50.0, "2026-10-02 02:00:00+00:00"),
                _payment("P4", "O9", 30.0, "2026-10-02 03:00:00+00:00"),
            ],
        )
        detail_sum = round(float(result.diff["diff_amount"].sum()), 2)
        assert result.diff_total == pytest.approx(-70.0)
        assert detail_sum == pytest.approx(result.diff_total)

    def test_outputs_written(self, workspace: Path) -> None:
        result = _run(
            workspace,
            [_order("O1", "成交", 100.0, "2026-10-01 10:00:00")],
            [_payment("P1", "O1", 90.0, "2026-10-01 02:00:00+00:00")],
        )
        out = workspace / "out"
        for name in ("orders_daily.csv", "payments_daily.csv", "diff.csv"):
            assert (out / name).is_file()
        summary = (out / "summary.txt").read_text(encoding="utf-8")
        assert "订单口径实收GMV" in summary
        assert "对账差异总额" in summary
        for category in reconcile.CATEGORIES:
            assert category in summary
        assert result.diff_total == pytest.approx(10.0)

    def test_empty_inputs_are_rejected(self, workspace: Path) -> None:
        orders_path = workspace / "orders.csv"
        payments_path = workspace / "payments.csv"
        orders_path.write_text("order_id,status,amount,updated_at\n", encoding="utf-8")
        payments_path.write_text("pay_id,order_id,amount,paid_at\n", encoding="utf-8")
        with pytest.raises(ValueError, match="输入数据为空"):
            reconcile.run(orders_path, payments_path, workspace / "out")


@pytest.mark.skipif(
    not (SAMPLE_ORDERS.is_file() and SAMPLE_PAYMENTS.is_file()),
    reason="样例数据不存在",
)
class TestSampleData:
    """样例数据端到端回归，锁定当前口径下的结果。"""

    @pytest.fixture(scope="class")
    def result(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> reconcile.ReconcileResult:
        out_dir = tmp_path_factory.mktemp("sample_out")
        return reconcile.run(SAMPLE_ORDERS, SAMPLE_PAYMENTS, out_dir)

    def test_totals(self, result: reconcile.ReconcileResult) -> None:
        assert result.order_total == pytest.approx(152059.41)
        assert result.payment_total == pytest.approx(157248.00)
        assert result.diff_total == pytest.approx(-5188.59)

    def test_window_is_thirty_days(self, result: reconcile.ReconcileResult) -> None:
        assert len(result.orders_daily) == 30
        assert len(result.payments_daily) == 30

    def test_daily_sums_match_totals(self, result: reconcile.ReconcileResult) -> None:
        assert result.orders_daily["gmv"].sum() == pytest.approx(
            result.order_total, abs=0.01
        )
        assert result.payments_daily["gmv"].sum() == pytest.approx(
            result.payment_total, abs=0.01
        )

    def test_every_diff_row_has_exactly_one_category(
        self, result: reconcile.ReconcileResult
    ) -> None:
        assert set(result.diff["category"]) <= set(reconcile.CATEGORIES)
        assert result.diff["category"].notna().all()
        assert result.category_stats["rows"].sum() == len(result.diff)

    def test_momentum_rows_consistent(self, result: reconcile.ReconcileResult) -> None:
        daily = result.orders_daily.dropna(subset=["prev_gmv"])
        recomputed = (daily["gmv"] / daily["prev_gmv"] - 1.0) * 100.0
        assert (recomputed.round(2) - daily["mom_pct"]).abs().max() < 0.02
