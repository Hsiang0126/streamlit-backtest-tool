"""
現代投資組合理論最佳化模組 (Portfolio Optimizer Module - MPT)
採用純 NumPy 向量化蒙地卡羅模擬 (5,000次抽樣)，免安裝額外套件，
極速計算：最大夏普比率 (Max Sharpe)、最小波動度 (Min Volatility) 與風險平價 (Risk Parity)。
"""
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np


class PortfolioOptimizer:
    """MPT 現代投資組合最佳化器。"""

    @classmethod
    def optimize_portfolio(
        cls,
        aligned_panel: pd.DataFrame,
        risk_free_rate: float = 0.015,
        num_simulations: int = 5000
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], pd.DataFrame, Dict[str, Any]]:
        """
        計算多資產之最佳配置比例。
        回傳：
        - max_sharpe_weights: 最高夏普權重字典
        - min_vol_weights: 最小波動權重字典
        - risk_parity_weights: 風險平價權重字典
        - equal_weights: 等權重字典
        - frontier_df: 效率前緣雲圖資料 (供繪圖使用)
        - summary_stats: 統計摘要
        """
        if isinstance(aligned_panel.columns, pd.MultiIndex):
            tickers = list(dict.fromkeys(aligned_panel.columns.get_level_values(0)))
        else:
            tickers = []

        if len(tickers) < 2:
            return {}, {}, {}, {}, pd.DataFrame(), {}

        # 提取各資產的日報酬率 (優先使用 adj_close)
        returns_dict = {}
        for t in tickers:
            col = "adj_close" if (t, "adj_close") in aligned_panel.columns else "close"
            price_s = pd.to_numeric(aligned_panel[(t, col)], errors="coerce").ffill().bfill()
            returns_dict[t] = price_s.pct_change().dropna()

        returns_df = pd.DataFrame(returns_dict).dropna()
        if len(returns_df) < 10:
            eq_w = {t: round(100.0 / len(tickers), 1) for t in tickers}
            return eq_w, eq_w, eq_w, eq_w, pd.DataFrame(), {}

        n_assets = len(tickers)
        mean_daily_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values * 252.0  # 年化共變異數

        annual_mean_returns = mean_daily_returns * 252.0

        # 向量化蒙地卡羅模擬 (Dirichlet 分布保證隨機權重和為 1 且皆非負)
        np.random.seed(42)
        weights_matrix = np.random.dirichlet(np.ones(n_assets), size=num_simulations)

        # 計算每個模擬組合的預期年化報酬與年化波動度
        port_returns = np.dot(weights_matrix, annual_mean_returns)
        port_vols = np.sqrt(np.sum(np.dot(weights_matrix, cov_matrix) * weights_matrix, axis=1))
        sharpe_ratios = (port_returns - risk_free_rate) / np.where(port_vols == 0, np.nan, port_vols)

        # 找出最高夏普與最小波動點
        idx_max_sharpe = int(np.nanargmax(sharpe_ratios))
        idx_min_vol = int(np.nanargmin(port_vols))

        def normalize_weights_dict(raw_w: np.ndarray) -> Dict[str, float]:
            pcts = np.round(raw_w * 100.0, 1)
            # 確保總和剛好為 100.0%
            diff = round(100.0 - float(np.sum(pcts)), 1)
            max_idx = int(np.argmax(pcts))
            pcts[max_idx] += diff
            return {tickers[i]: float(pcts[i]) for i in range(n_assets)}

        max_sharpe_weights = normalize_weights_dict(weights_matrix[idx_max_sharpe])
        min_vol_weights = normalize_weights_dict(weights_matrix[idx_min_vol])

        # 風險平價 (反波動率加權近似解)
        asset_vols = returns_df.std().values * np.sqrt(252.0)
        inv_vols = 1.0 / np.where(asset_vols == 0, 1e-6, asset_vols)
        rp_weights_raw = inv_vols / np.sum(inv_vols)
        risk_parity_weights = normalize_weights_dict(rp_weights_raw)

        # 等權重基準
        eq_weights = normalize_weights_dict(np.ones(n_assets) / n_assets)

        # 抽樣 2,000 個點供效率前緣繪圖，保持流暢
        sample_size = min(2000, num_simulations)
        sample_indices = np.random.choice(num_simulations, size=sample_size, replace=False)

        frontier_df = pd.DataFrame({
            "volatility": port_vols[sample_indices] * 100.0,
            "return": port_returns[sample_indices] * 100.0,
            "sharpe": sharpe_ratios[sample_indices]
        })

        summary_stats = {
            "max_sharpe": {
                "return": port_returns[idx_max_sharpe],
                "volatility": port_vols[idx_max_sharpe],
                "sharpe": sharpe_ratios[idx_max_sharpe]
            },
            "min_vol": {
                "return": port_returns[idx_min_vol],
                "volatility": port_vols[idx_min_vol],
                "sharpe": sharpe_ratios[idx_min_vol]
            }
        }

        return max_sharpe_weights, min_vol_weights, risk_parity_weights, eq_weights, frontier_df, summary_stats
