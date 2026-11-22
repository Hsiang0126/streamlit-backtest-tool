"""
財務與績效評估指標模組 (Metrics Module)
計算淨獲利金額、TWR 報酬、CAGR 年化複合增長、XIRR 內部報酬率、
年化波動度、Sharpe、Sortino、Calmar 比率、最長套牢天數、水下回撤序列與月度報酬矩陣。
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class PerformanceMetrics:
    """績效計算與統計分析器。"""

    MONTH_NAMES = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]

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

    @staticmethod
    def calculate_drawdown_series(df: pd.DataFrame) -> pd.Series:
        """計算水下百分比回撤序列（負值，如 -0.15 代表 -15%）。"""
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if "equity_net" in df.columns:
            twr = pd.to_numeric(df["equity_net"], errors="coerce").ffill().fillna(1.0)
        elif "portfolio_value" in df.columns:
            pv = pd.to_numeric(df["portfolio_value"], errors="coerce")
            base = float(pv.iloc[0]) if len(pv) > 0 and pv.iloc[0] > 0 else 1.0
            twr = pv / base
        else:
            return pd.Series(dtype=float)

        running_peak = twr.cummax().replace(0.0, np.nan)
        dd = ((twr - running_peak) / running_peak).fillna(0.0)
        return dd

    @classmethod
    def calculate_monthly_returns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """基於 TWR 淨值曲線計算 Year × (1月~12月 + 全年) 月度報酬矩陣。"""
        if df is None or len(df) < 2:
            return pd.DataFrame()
        if "equity_net" in df.columns:
            twr = pd.to_numeric(df["equity_net"], errors="coerce").ffill().fillna(1.0)
        elif "portfolio_value" in df.columns:
            pv = pd.to_numeric(df["portfolio_value"], errors="coerce")
            base = float(pv.iloc[0]) if pv.iloc[0] > 0 else 1.0
            twr = pv / base
        else:
            return pd.DataFrame()

        if "total_invested" in df.columns:
            active_mask = pd.to_numeric(df["total_invested"], errors="coerce").fillna(0.0) > 0
            if active_mask.any():
                twr = twr.loc[active_mask]

        if len(twr) < 2:
            return pd.DataFrame()

        dt_idx = pd.DatetimeIndex(twr.index)
        twr_s = pd.Series(twr.values, index=dt_idx)
        monthly_last = twr_s.resample("ME").last().dropna()
        if monthly_last.empty:
            return pd.DataFrame()

        # 為首月補上該段起始基準淨值 1.0，使第 1 個月亦能算出真實報酬
        monthly_rets = monthly_last.pct_change()
        monthly_rets.iloc[0] = (monthly_last.iloc[0] / float(twr_s.iloc[0])) - 1.0 if twr_s.iloc[0] > 0 else 0.0

        years = sorted(monthly_rets.index.year.unique())
        rows = []
        for yr in years:
            row: Dict[str, Any] = {"年度": str(yr)}
            yr_vals = []
            for m in range(1, 13):
                sub = monthly_rets[(monthly_rets.index.year == yr) & (monthly_rets.index.month == m)]
                if not sub.empty:
                    val = float(sub.iloc[0])
                    row[cls.MONTH_NAMES[m - 1]] = val
                    yr_vals.append(val)
                else:
                    row[cls.MONTH_NAMES[m - 1]] = np.nan
            if yr_vals:
                row["全年"] = float(np.prod([1.0 + v for v in yr_vals]) - 1.0)
            else:
                row["全年"] = np.nan
            rows.append(row)

        return pd.DataFrame(rows).set_index("年度")

    @staticmethod
    def _calc_max_drawdown_duration(twr_series: pd.Series) -> int:
        """計算最長未創新高（水下套牢）之日曆天數。"""
        if twr_series is None or len(twr_series) < 2:
            return 0
        dates = pd.DatetimeIndex(twr_series.index)
        vals = twr_series.to_numpy(dtype=float)

        peak_val = vals[0]
        peak_date = dates[0]
        max_days = 0

        for dt, val in zip(dates[1:], vals[1:]):
            if val >= peak_val:
                peak_val = val
                peak_date = dt
            else:
                dur = int((dt - peak_date).days)
                if dur > max_days:
                    max_days = dur
        return max_days

    @classmethod
    def evaluate(
        cls,
        df: pd.DataFrame,
        curr_symbol: str = "NT$",
        initial_capital: float = 0.0,
        risk_free_rate: float = 0.015,
    ) -> Dict[str, Any]:
        if df is None or len(df) < 2 or "portfolio_value" not in df.columns:
            return {}

        final_value = float(df["portfolio_value"].iloc[-1])
        total_invested = float(df["total_invested"].iloc[-1]) if "total_invested" in df.columns else float(initial_capital)
        net_profit = final_value - total_invested
        roi = (net_profit / total_invested) if total_invested > 0 else 0.0

        twr_series = (
            pd.to_numeric(df["equity_net"], errors="coerce").ffill().fillna(1.0)
            if "equity_net" in df.columns
            else (df["portfolio_value"] / max(float(df["portfolio_value"].iloc[0]), 1.0))
        )
        total_twr = float(twr_series.iloc[-1] - 1.0)

        # 從第一筆實際資金進場日計算有效投資年數，避免期初本金為 0 時之空窗期稀釋 CAGR / 波動度
        if "total_invested" in df.columns:
            active_mask = pd.to_numeric(df["total_invested"], errors="coerce").fillna(0.0) > 0
            active_twr = twr_series.loc[active_mask] if active_mask.any() else twr_series
        else:
            active_twr = twr_series

        if len(active_twr) < 2:
            active_twr = twr_series

        elapsed_days = max(1, (pd.Timestamp(active_twr.index[-1]) - pd.Timestamp(active_twr.index[0])).days)
        years = elapsed_days / 365.25
        start_idx_val = float(active_twr.iloc[0]) if float(active_twr.iloc[0]) > 0 else 1.0
        ending_rel = max(float(active_twr.iloc[-1]) / start_idx_val, 0.0)
        cagr = (ending_rel ** (1.0 / years) - 1.0) if ending_rel > 0 and years > 0.05 else total_twr

        running_peak = active_twr.cummax()
        drawdown_series = ((active_twr - running_peak) / running_peak.replace(0, np.nan)).fillna(0.0)
        mdd = float(drawdown_series.min()) if not drawdown_series.empty else 0.0
        max_dd_days = cls._calc_max_drawdown_duration(active_twr)

        # 年化波動度、Sharpe、Sortino、Calmar
        daily_rets = active_twr.pct_change().dropna()
        ann_factor = max(1.0, len(active_twr) / max(years, 1e-4))
        vol = float(daily_rets.std(ddof=1) * np.sqrt(ann_factor)) if len(daily_rets) >= 2 else 0.0
        sharpe = ((cagr - risk_free_rate) / vol) if vol > 1e-8 else 0.0

        downside = daily_rets[daily_rets < 0]
        downside_dev = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(ann_factor)) if len(downside) > 0 else 0.0
        sortino = ((cagr - risk_free_rate) / downside_dev) if downside_dev > 1e-8 else (sharpe if cagr > risk_free_rate else 0.0)
        calmar = (cagr / abs(mdd)) if abs(mdd) > 1e-6 else 0.0

        # XIRR 現金流建構：精準處理第 0 天期初投入或首日扣款
        cashflows: List[float] = []
        dates: List[pd.Timestamp] = []
        first_dt = pd.Timestamp(df.index[0])
        first_flow = float(df["external_flow"].iloc[0]) if "external_flow" in df.columns else 0.0
        init_outflow = float(initial_capital) + first_flow
        if init_outflow > 0:
            cashflows.append(-init_outflow)
            dates.append(first_dt)

        if "external_flow" in df.columns:
            for d, flow in df["external_flow"].iloc[1:].items():
                if float(flow) > 0:
                    cashflows.append(-float(flow))
                    dates.append(pd.Timestamp(d))

        cashflows.append(final_value)
        dates.append(pd.Timestamp(df.index[-1]))
        xirr = cls.calculate_xirr(cashflows, dates)

        trades = (
            int(df["trades"].iloc[-1])
            if "trades" in df.columns and pd.notna(df["trades"].iloc[-1])
            else 0
        )

        return {
            "累積投入本金": f"{curr_symbol} {total_invested:,.0f}",
            "最終總資產": f"{curr_symbol} {final_value:,.0f}",
            "實賺獲利金額": f"{curr_symbol} {net_profit:+,.0f}",
            "淨獲利報酬率": f"{roi:+.2%}",
            "TWR 總報酬率": f"{total_twr:+.2%}",
            "CAGR 年化報酬": f"{cagr:+.2%}",
            "XIRR 資金加權": f"{xirr:+.2%}" if np.isfinite(xirr) else "-",
            "年化波動度": f"{vol:.2%}",
            "夏普比率 (Sharpe)": f"{sharpe:.2f}",
            "索提諾比率 (Sortino)": f"{sortino:.2f}",
            "卡瑪比率 (Calmar)": f"{calmar:.2f}",
            "最大歷史回撤 (MDD)": f"{mdd:.2%}",
            "最長套牢天數": f"{max_dd_days} 天",
            "累積交易次數": f"{trades} 次",
            "_raw_final": final_value,
            "_raw_invested": total_invested,
            "_raw_net_profit": net_profit,
            "_raw_roi": roi,
            "_raw_twr": total_twr,
            "_raw_cagr": cagr,
            "_raw_vol": vol,
            "_raw_sharpe": sharpe,
            "_raw_sortino": sortino,
            "_raw_calmar": calmar,
            "_raw_mdd": mdd,
            "_raw_max_dd_days": max_dd_days,
            "_raw_xirr": xirr,
            "_raw_trades": trades,
        }
