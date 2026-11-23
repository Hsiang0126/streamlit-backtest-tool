"""
現代投資組合理論最佳化模組 (Portfolio Optimizer Module - MPT)
升級項目：
1. 先將各資產價格依動態匯率換算為目標計價幣別 (target_currency)，真實納入金流與匯率波動。
2. 優先採用週報酬 (W-FRI, 年化乘數 52) 消除跨市場休市 ffill() 製造之大量假 0% 日報酬造成的波動低估與相關性失真。
3. 混合 Dirichlet(1.0)、Dirichlet(0.3) 與單資產頂點，高覆蓋探索效率前緣邊界。
4. 回傳相關係數矩陣 (corr_matrix) 與各單一資產年化報酬/波動率 (asset_stats)。
"""
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from core.cost_model import CostModel
from core.portfolio_engine import BacktestEngine


class PortfolioOptimizer:
    """MPT 現代投資組合最佳化器。"""

    @staticmethod
    def _detect_curr(aligned_panel: pd.DataFrame, ticker: str) -> str:
        if (ticker, "quote_currency") in aligned_panel.columns:
            c = str(aligned_panel[(ticker, "quote_currency")].iloc[0]).strip().upper()
            if c and c not in {"NAN", "NONE"}:
                return c
        if CostModel.is_taiwan_ticker(str(ticker)):
            return "TWD"
        return "USD"

    @classmethod
    def optimize_portfolio(
        cls,
        aligned_panel: pd.DataFrame,
        aligned_fx: Optional[pd.Series] = None,
        target_currency: str = "TWD",
        risk_free_rate: float = 0.015,
        num_simulations: int = 6000,
    ) -> Tuple[
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
        pd.DataFrame,
        Dict[str, Any],
    ]:
        if aligned_panel is None or aligned_panel.empty or not isinstance(aligned_panel.columns, pd.MultiIndex):
            return {}, {}, {}, {}, pd.DataFrame(), {}

        tickers = list(dict.fromkeys(aligned_panel.columns.get_level_values(0)))
        if len(tickers) < 2:
            return {}, {}, {}, {}, pd.DataFrame(), {}

        target_curr_code = "TWD" if "TWD" in target_currency.upper() else "USD"
        data_idx = aligned_panel.index
        if aligned_fx is not None and not aligned_fx.empty:
            fx_s = aligned_fx.reindex(data_idx).ffill().bfill().fillna(32.0)
        else:
            fx_s = pd.Series(32.0, index=data_idx)

        target_prices: Dict[str, pd.Series] = {}
        for t in tickers:
            col = "adj_close" if (t, "adj_close") in aligned_panel.columns else "close"
            price_s = pd.to_numeric(aligned_panel[(t, col)], errors="coerce").ffill().bfill()
            q_curr = cls._detect_curr(aligned_panel, t)
            if q_curr != target_curr_code:
                mult = pd.Series(
                    [BacktestEngine._resolve_fx_multiplier(q_curr, target_curr_code, float(v)) for v in fx_s.values],
                    index=data_idx,
                )
                target_prices[t] = price_s * mult
            else:
                target_prices[t] = price_s

        price_df = pd.DataFrame(target_prices).dropna()

        # 優先以週五收盤計算週報酬，徹底消除跨市場休市 ffill 產生的假 0% 日報酬偏差
        weekly_prices = price_df.resample("W-FRI").last().dropna()
        weekly_rets = weekly_prices.pct_change().dropna()
        if len(weekly_rets) >= 12:
            returns_df = weekly_rets
            ann_factor = 52.0
        else:
            returns_df = price_df.pct_change().dropna()
            ann_factor = 252.0

        if len(returns_df) < 8:
            eq_w = {t: round(100.0 / len(tickers), 1) for t in tickers}
            return eq_w, eq_w, eq_w, eq_w, pd.DataFrame(), {}

        n_assets = len(tickers)
        annual_mean_returns = returns_df.mean().values * ann_factor
        cov_matrix = returns_df.cov().values * ann_factor
        corr_matrix = returns_df.corr().fillna(0.0)
        asset_vols = returns_df.std().values * np.sqrt(ann_factor)

        rng = np.random.default_rng(42)
        n_uniform = int(num_simulations * 0.40)
        n_conc = max(1, num_simulations - n_uniform - n_assets)
        w_uniform = rng.dirichlet(np.ones(n_assets), size=n_uniform)
        w_conc = rng.dirichlet(np.full(n_assets, 0.3), size=n_conc)
        w_corners = np.eye(n_assets)
        weights_matrix = np.vstack([w_uniform, w_conc, w_corners])

        port_returns = np.dot(weights_matrix, annual_mean_returns)
        port_vols = np.sqrt(np.maximum(0.0, np.sum(np.dot(weights_matrix, cov_matrix) * weights_matrix, axis=1)))
        sharpe_ratios = (port_returns - risk_free_rate) / np.where(port_vols <= 1e-10, np.nan, port_vols)

        if np.all(np.isnan(sharpe_ratios)) or np.all(port_vols <= 1e-10):
            eq_w = {tickers[i]: round(100.0 / n_assets, 1) for i in range(n_assets)}
            return eq_w, eq_w, eq_w, eq_w, pd.DataFrame(), {}

        idx_max_sharpe = int(np.nanargmax(sharpe_ratios))
        idx_min_vol = int(np.nanargmin(port_vols))

        def normalize_weights_dict(raw_w: np.ndarray) -> Dict[str, float]:
            pcts = np.round(raw_w * 100.0, 1)
            diff = round(100.0 - float(np.sum(pcts)), 1)
            max_idx = int(np.argmax(pcts))
            pcts[max_idx] = round(pcts[max_idx] + diff, 1)
            return {tickers[i]: float(pcts[i]) for i in range(n_assets)}

        max_sharpe_weights = normalize_weights_dict(weights_matrix[idx_max_sharpe])
        min_vol_weights = normalize_weights_dict(weights_matrix[idx_min_vol])

        inv_vols = 1.0 / np.where(asset_vols <= 1e-6, 1e-6, asset_vols)
        rp_weights_raw = inv_vols / np.sum(inv_vols)
        risk_parity_weights = normalize_weights_dict(rp_weights_raw)

        eq_weights = normalize_weights_dict(np.ones(n_assets) / n_assets)

        total_pts = len(weights_matrix)
        sample_size = min(2200, total_pts)
        sample_indices = rng.choice(total_pts, size=sample_size, replace=False)

        frontier_df = pd.DataFrame(
            {
                "volatility": port_vols[sample_indices] * 100.0,
                "return": port_returns[sample_indices] * 100.0,
                "sharpe": sharpe_ratios[sample_indices],
            }
        )

        asset_stats = {
            tickers[i]: {
                "return": float(annual_mean_returns[i]),
                "volatility": float(asset_vols[i]),
                "sharpe": float((annual_mean_returns[i] - risk_free_rate) / asset_vols[i])
                if asset_vols[i] > 1e-8
                else 0.0,
            }
            for i in range(n_assets)
        }

        summary_stats = {
            "max_sharpe": {
                "return": float(port_returns[idx_max_sharpe]),
                "volatility": float(port_vols[idx_max_sharpe]),
                "sharpe": float(sharpe_ratios[idx_max_sharpe]),
            },
            "min_vol": {
                "return": float(port_returns[idx_min_vol]),
                "volatility": float(port_vols[idx_min_vol]),
                "sharpe": float(sharpe_ratios[idx_min_vol]),
            },
            "corr_matrix": corr_matrix,
            "asset_stats": asset_stats,
            "cov_matrix": cov_matrix,
            "annual_mean_returns": annual_mean_returns,
            "tickers": tickers,
        }

        return max_sharpe_weights, min_vol_weights, risk_parity_weights, eq_weights, frontier_df, summary_stats

    @staticmethod
    def evaluate_custom_weights(
        summary_stats: Dict[str, Any],
        weights_dict: Dict[str, float],
        risk_free_rate: float = 0.015,
    ) -> Optional[Dict[str, float]]:
        """計算使用者目前配置在效率前緣座標軸上之年化報酬、波動率與夏普點。"""
        if not summary_stats or "cov_matrix" not in summary_stats:
            return None
        tickers = summary_stats["tickers"]
        w = np.array([max(0.0, float(weights_dict.get(t, 0.0))) for t in tickers], dtype=float)
        if np.sum(w) <= 0:
            return None
        # 若有現金儲備部位 (不在 risky tickers 內)，總風險權重可能 < 1.0，波動會隨現金比例等比縮小
        mean_rets = summary_stats["annual_mean_returns"]
        cov = summary_stats["cov_matrix"]
        cash_w = max(0.0, 1.0 - float(np.sum(w)))
        port_ret = float(np.dot(w, mean_rets) + cash_w * risk_free_rate)
        port_vol = float(np.sqrt(max(0.0, np.dot(w, np.dot(cov, w)))))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 1e-8 else 0.0
        return {
            "return": port_ret,
            "volatility": port_vol,
            "sharpe": sharpe,
        }
