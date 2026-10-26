"""
逐筆成交帳本與績效分析模組 (Trade Ledger Module - 雙重分析升級版)
1. extract_buy_records: 專門為 DCA、Dip DCA 逐筆列出每一次購買日期、買入價格、投入金額、累計股數與本金。
2. extract_trades: 針對擇時策略配對進出場 (Round-Trip)，計算勝率、盈虧比、持倉天數。
"""
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np


class TradeLedger:
    """逐筆交易與購買歷史分析器。"""

    @staticmethod
    def extract_buy_records(
        res_df: pd.DataFrame,
        curr_symbol: str = "NT$"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        從回測結果中提取每一次「買入扣款」的完整歷史紀錄 (專門支援 DCA、Dip DCA 與定投買入)。
        """
        if res_df is None or "trades" not in res_df.columns:
            return pd.DataFrame(), {}

        trades_series = res_df["trades"]
        trade_diff = trades_series.diff().fillna(trades_series.iloc[0])
        buy_mask = trade_diff > 0

        sub_df = res_df[buy_mask].copy()
        if sub_df.empty:
            return pd.DataFrame(), {}

        records = []
        prev_shares = 0.0

        for idx, (dt, row) in enumerate(sub_df.iterrows()):
            close_p = float(row["close"]) if "close" in row else 1.0
            stock_val = float(row["stock_value"])
            curr_shares = stock_val / close_p if close_p > 0 else 0.0
            bought_shares = max(0.0, curr_shares - prev_shares)
            open_p = float(row["open"]) if "open" in row else close_p

            invested_now = float(row["total_invested"])
            cash_rem = float(row["cash"])
            port_val = float(row["portfolio_value"])

            records.append({
                "購買序號": f"第 {idx + 1} 次",
                "購買日期": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "買入價格": f"{curr_symbol} {open_p:,.2f}",
                "當期持股市值": f"{curr_symbol} {stock_val:,.0f}",
                "當期剩餘現金": f"{curr_symbol} {cash_rem:,.0f}",
                "累計投入本金": f"{curr_symbol} {invested_now:,.0f}",
                "帳戶總資產": f"{curr_symbol} {port_val:,.0f}"
            })
            prev_shares = curr_shares

        df_records = pd.DataFrame(records)

        # 統計資訊
        total_buys = len(records)
        final_stock_val = float(res_df["stock_value"].iloc[-1])
        final_invested = float(res_df["total_invested"].iloc[-1])
        final_close = float(res_df["close"].iloc[-1])
        final_shares = final_stock_val / final_close if final_close > 0 else 0.0
        avg_cost = (final_invested - float(res_df["cash"].iloc[-1])) / final_shares if final_shares > 0 else 0.0

        summary = {
            "累積購買次數": f"{total_buys} 次",
            "累積持有股數": f"{final_shares:,.2f} 股",
            "平均持股成本": f"{curr_symbol} {avg_cost:,.2f}",
            "期末持股市值": f"{curr_symbol} {final_stock_val:,.0f}"
        }

        display_cols = ["購買序號", "購買日期", "買入價格", "當期持股市值", "當期剩餘現金", "累計投入本金", "帳戶總資產"]
        return df_records[display_cols], summary

    @staticmethod
    def extract_trades(
        data: pd.DataFrame,
        pos_series: pd.Series,
        cost_fee: float = 0.00071,
        cost_tax: float = 0.003,
        cost_slip: float = 0.0003,
        curr_symbol: str = "NT$"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        trades = []
        in_trade = False
        entry_date = None
        entry_price = 0.0

        open_prices = data["open"].values
        close_prices = data["close"].values
        dates = data.index

        for i in range(len(data)):
            pos = int(pos_series.iloc[i])
            prev_pos = int(pos_series.iloc[i - 1]) if i > 0 else 0
            open_p = open_prices[i]

            if pos == 1 and prev_pos == 0:
                in_trade = True
                entry_date = dates[i]
                entry_price = open_p * (1.0 + cost_slip) * (1.0 + cost_fee)

            elif pos == 0 and prev_pos == 1 and in_trade:
                exit_date = dates[i]
                exit_price = open_p * (1.0 - cost_slip) * (1.0 - cost_fee) * (1.0 - cost_tax)
                pnl_pct = (exit_price / entry_price) - 1.0 if entry_price > 0 else 0.0
                holding_days = max(1, (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days)

                trades.append({
                    "進場日期": pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
                    "出場日期": pd.Timestamp(exit_date).strftime("%Y-%m-%d"),
                    "持倉天數": holding_days,
                    "進場價格": round(entry_price, 2),
                    "出場價格": round(exit_price, 2),
                    "單筆報酬率": f"{pnl_pct:+.2%}",
                    "_raw_pnl_pct": pnl_pct,
                    "狀態": "獲利 🟢" if pnl_pct > 0 else ("虧損 🔴" if pnl_pct < 0 else "平手 ⚪")
                })
                in_trade = False

        if in_trade and entry_date is not None:
            last_date = dates[-1]
            last_close = close_prices[-1]
            pnl_pct = (last_close / entry_price) - 1.0 if entry_price > 0 else 0.0
            holding_days = max(1, (pd.Timestamp(last_date) - pd.Timestamp(entry_date)).days)
            trades.append({
                "進場日期": pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
                "出場日期": pd.Timestamp(last_date).strftime("%Y-%m-%d (持有中)"),
                "持倉天數": holding_days,
                "進場價格": round(entry_price, 2),
                "出場價格": round(last_close, 2),
                "單筆報酬率": f"{pnl_pct:+.2%}",
                "_raw_pnl_pct": pnl_pct,
                "狀態": "持有中 🟡"
            })

        df_trades = pd.DataFrame(trades)
        if df_trades.empty:
            summary = {
                "總交易次數": 0,
                "已平倉交易": 0,
                "勝率": "-",
                "盈虧比 (Profit Factor)": "-",
                "平均持倉天數": "-",
                "獲利筆數": 0,
                "虧損筆數": 0,
                "_raw_win_rate": 0.0,
                "_raw_pf": 0.0
            }
            return pd.DataFrame(), summary

        closed_trades = [t for t in trades if "持有中" not in t["狀態"]]
        if closed_trades:
            pnl_list = [t["_raw_pnl_pct"] for t in closed_trades]
            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p < 0]
            win_count = len(wins)
            loss_count = len(losses)
            total_closed = len(closed_trades)

            win_rate = (win_count / total_closed) if total_closed > 0 else 0.0
            gross_win = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
            avg_holding = np.mean([t["持倉天數"] for t in closed_trades])
        else:
            win_rate = 0.0
            profit_factor = 0.0
            win_count = 0
            loss_count = 0
            avg_holding = 0

        summary = {
            "總交易次數": len(trades),
            "已平倉交易": len(closed_trades),
            "勝率": f"{win_rate:.1%}",
            "盈虧比 (Profit Factor)": f"{profit_factor:.2f}" if profit_factor > 0 else "-",
            "平均持倉天數": f"{avg_holding:.1f} 天",
            "獲利筆數": win_count,
            "虧損筆數": loss_count,
            "_raw_win_rate": win_rate,
            "_raw_pf": profit_factor
        }

        display_cols = ["進場日期", "出場日期", "持倉天數", "進場價格", "出場價格", "單筆報酬率", "狀態"]
        return df_trades[display_cols], summary
