"""
財務與績效評估指標模組 (Metrics Module)
計算淨獲利金額、TWR 報酬、CAGR 年化複合增長、最大歷史回撤 MDD、XIRR 內部報酬率。
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class PerformanceMetrics:
    """績效計算與統計分析器。"""

    @staticmethod
    def calculate_xirr(cashflows: List[float], dates: List[pd.Timestamp]) -> float:
        """二分法穩健求解 XIRR (年化內部報酬率)。"""
        try:
            flows = np.asarray(cashflows, dtype=float).reshape(-1)
            ts_dates = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce"))
        except Exception:
            return np.nan

        if len(flows) < 2 or len(flows) != len(ts_dates) or ts_dates.isna().any():
            return np.nan

        df_cf = pd.DataFrame({"flow": flows, "date": ts_dates}).groupby("date", as_index=False).sum()
        flows = df_cf["flow"].to_numpy(dtype=float)
        ts_dates = pd.DatetimeIndex(df_cf["date"])

        if len(flows) < 2 or not (np.any(flows < 0) and np.any(flows > 0)):
            return np.nan

        days = (ts_dates - ts_dates[0]).days.to_numpy(dtype=float)
        years = days / 365.25
        if years[-1] <= 0:
            return np.nan

        def npv(r: float) -> float:
            try:
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    val = float(np.sum(flows / np.power(1.0 + r, years)))
                return val if np.isfinite(val) else np.nan
            except Exception:
                return np.nan

        low, high = -0.9999, 10.0
        f_low, f_high = npv(low), npv(high)

        attempts = 0
        while np.isfinite(f_low) and np.isfinite(f_high) and np.sign(f_low) == np.sign(f_high) and attempts < 30:
            high = high * 1.5 + 1.0
            f_high = npv(high)
            attempts += 1

        if not np.isfinite(f_low) or not np.isfinite(f_high) or np.sign(f_low) == np.sign(f_high):
            return np.nan

        for _ in range(100):
            mid = (low + high) / 2.0
            f_mid = npv(mid)
            if not np.isfinite(f_mid):
                high = mid
                continue
            if abs(f_mid) < 1e-7 or abs(high - low) < 1e-9:
                return mid
            if np.sign(f_low) != np.sign(f_mid):
                high, f_high = mid, f_mid
            else:
                low, f_low = mid, f_mid

        return (low + high) / 2.0

    @classmethod
    def evaluate(cls, df: pd.DataFrame, curr_symbol: str = "NT$", initial_capital: float = 0.0) -> Dict[str, Any]:
        if df is None or len(df) < 2 or "portfolio_value" not in df.columns:
            return {}

        final_value = float(df["portfolio_value"].iloc[-1])
        total_invested = float(df["total_invested"].iloc[-1])
        net_profit = final_value - total_invested
        roi = (net_profit / total_invested) if total_invested > 0 else 0.0

        twr_series = df["equity_net"] if "equity_net" in df.columns else (df["portfolio_value"] / df["portfolio_value"].iloc[0])
        total_twr = float(twr_series.iloc[-1] - 1.0)

        elapsed_days = max(1, (pd.Timestamp(df.index[-1]) - pd.Timestamp(df.index[0])).days)
        years = elapsed_days / 365.25
        ending_index = max(float(twr_series.iloc[-1]), 0.0)
        cagr = (ending_index ** (1.0 / years) - 1.0) if ending_index > 0 and years > 0.05 else total_twr

        running_peak = twr_series.cummax()
        drawdown_series = (twr_series - running_peak) / running_peak.replace(0, np.nan)
        mdd = float(drawdown_series.min()) if not drawdown_series.empty else 0.0

        cashflows = [-float(initial_capital if initial_capital > 0 else df["total_invested"].iloc[0])]
        dates = [pd.Timestamp(df.index[0])]
        if "external_flow" in df.columns:
            for d, flow in df["external_flow"].iloc[1:].items():
                if float(flow) > 0:
                    cashflows.append(-float(flow))
                    dates.append(pd.Timestamp(d))
        cashflows.append(final_value)
        dates.append(pd.Timestamp(df.index[-1]))
        xirr = cls.calculate_xirr(cashflows, dates)

        trades = int(df["trades"].iloc[-1]) if "trades" in df.columns and isinstance(df["trades"].iloc[-1], (int, np.integer)) else (
            int(df["trades"].iloc[-1]) if "trades" in df.columns and pd.notna(df["trades"].iloc[-1]) else 0
        )

        return {
            "累積投入本金": f"{curr_symbol} {total_invested:,.0f}",
            "最終總資產": f"{curr_symbol} {final_value:,.0f}",
            "實賺獲利金額": f"{curr_symbol} {net_profit:+,.0f}",
            "淨獲利報酬率": f"{roi:+.2%}",
            "TWR 總報酬率": f"{total_twr:+.2%}",
            "CAGR 年化報酬": f"{cagr:+.2%}",
            "XIRR 資金加權": f"{xirr:+.2%}" if np.isfinite(xirr) else "-",
            "最大歷史回撤 (MDD)": f"{mdd:.2%}",
            "累積交易次數": f"{trades} 次",
            "_raw_final": final_value,
            "_raw_invested": total_invested,
            "_raw_net_profit": net_profit,
            "_raw_roi": roi,
            "_raw_twr": total_twr,
            "_raw_cagr": cagr,
            "_raw_mdd": mdd,
            "_raw_xirr": xirr,
            "_raw_trades": trades
        }
