"""
逐筆成交帳本與績效分析模組 (Trade Ledger Module)
1. extract_buy_records: 精準讀取真實持股股數 (shares) 與匯率 (to_target_fx)，為 DCA / Dip DCA 列出逐筆購買明細與正確平均成本。
2. extract_trades: 直接基於回測結果區間 (res_df) 配對進出場 (Round-Trip)，計入持倉期間股利與目標幣別換算價格，計算勝率、盈虧比、期望值與持倉天數。
"""
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from core.cost_model import CostModel


class TradeLedger:
    """逐筆交易與購買歷史分析器。"""

    @staticmethod
    def extract_buy_records(
        res_df: pd.DataFrame,
        curr_symbol: str = "NT$",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if res_df is None or res_df.empty or "trades" not in res_df.columns:
            return pd.DataFrame(), {}

        trades_series = res_df["trades"]
        trade_diff = trades_series.diff().fillna(trades_series.iloc[0])
        buy_mask = trade_diff > 0

        sub_df = res_df[buy_mask].copy()
        if sub_df.empty:
            return pd.DataFrame(), {}

        records = []
        prev_shares = 0.0
        has_shares_col = "shares" in res_df.columns

        for _, (dt, row) in enumerate(sub_df.iterrows()):
            close_p = float(row["close"]) if "close" in row else 1.0
            open_p = float(row["open"]) if "open" in row else close_p
            stock_val = float(row["stock_value"])
            cash_rem = float(row["cash"])
            invested_now = float(row["total_invested"])
            port_val = float(row["portfolio_value"])
            fx = float(row["to_target_fx"]) if "to_target_fx" in row else 1.0

            if has_shares_col:
                curr_shares = float(row["shares"])
            else:
                curr_shares = stock_val / (close_p * fx) if (close_p * fx) > 0 else 0.0
            bought_shares = max(0.0, curr_shares - prev_shares)

            if stock_val <= 0 and invested_now <= 0:
                continue

            open_p_target = open_p * fx
            est_buy_amt = bought_shares * open_p_target
            p_display = (
                f"{curr_symbol} {open_p_target:,.2f}"
                if abs(fx - 1.0) < 1e-4
                else f"{curr_symbol} {open_p_target:,.2f} (${open_p:,.2f})"
            )

            records.append(
                {
                    "購買序號": f"第 {len(records) + 1} 次",
                    "購買日期": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "買入價格": p_display,
                    "買入股數": f"{bought_shares:,.2f}",
                    "單筆買入金額": f"{curr_symbol} {est_buy_amt:,.0f}",
                    "累計股數": f"{curr_shares:,.2f}",
                    "當期持股市值": f"{curr_symbol} {stock_val:,.0f}",
                    "累計投入本金": f"{curr_symbol} {invested_now:,.0f}",
                    "帳戶總資產": f"{curr_symbol} {port_val:,.0f}",
                }
            )
            prev_shares = curr_shares

        df_records = pd.DataFrame(records)
        if df_records.empty:
            return pd.DataFrame(), {}

        total_buys = len(records)
        final_stock_val = float(res_df["stock_value"].iloc[-1])
        final_invested = float(res_df["total_invested"].iloc[-1])
        final_cash = float(res_df["cash"].iloc[-1])

        if has_shares_col:
            final_shares = float(res_df["shares"].iloc[-1])
        else:
            final_close = float(res_df["close"].iloc[-1])
            final_fx = float(res_df["to_target_fx"].iloc[-1]) if "to_target_fx" in res_df.columns else 1.0
            final_shares = final_stock_val / (final_close * final_fx) if (final_close * final_fx) > 0 else 0.0

        # 淨投入股票成本 (總投入本金減去尚未花掉的現金；未扣除已收到之股利以反映實付成本基準)
        avg_cost = max(0.0, final_invested - final_cash) / final_shares if final_shares > 0 else 0.0

        summary = {
            "累積購買次數": f"{total_buys} 次",
            "累積持有股數": f"{final_shares:,.2f} 股",
            "平均持股成本": f"{curr_symbol} {avg_cost:,.2f}",
            "期末持股市值": f"{curr_symbol} {final_stock_val:,.0f}",
        }

        display_cols = [
            "購買序號",
            "購買日期",
            "買入價格",
            "買入股數",
            "單筆買入金額",
            "累計股數",
            "當期持股市值",
            "累計投入本金",
            "帳戶總資產",
        ]
        return df_records[display_cols], summary

    @staticmethod
    def extract_trades(
        data: pd.DataFrame,
        pos_series: Optional[pd.Series] = None,
        cost_fee: float = 0.0007125,
        cost_tax: float = 0.003,
        cost_slip: float = 0.0003,
        curr_symbol: str = "NT$",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if data is None or data.empty:
            return pd.DataFrame(), {
                "總交易次數": 0,
                "已平倉交易": 0,
                "勝率": "-",
                "盈虧比 (Profit Factor)": "-",
                "單筆期望值": "-",
                "平均持倉天數": "-",
                "獲利筆數": 0,
                "虧損筆數": 0,
                "_raw_win_rate": 0.0,
                "_raw_pf": 0.0,
            }

        # 優先採用回測結果中之實際持倉 (已包含停損/移動停利狀態)，並確保只統計使用者回測區間
        if "position" in data.columns:
            eval_df = data
            pos_s = eval_df["position"].astype(int)
        elif pos_series is not None:
            if "user_start_idx" in data.columns:
                eval_df = data[data["user_start_idx"] == True]
                if eval_df.empty:
                    eval_df = data
            else:
                eval_df = data
            pos_s = pos_series.reindex(eval_df.index).fillna(0).astype(int)
        else:
            return pd.DataFrame(), {
                "總交易次數": 0,
                "已平倉交易": 0,
                "勝率": "-",
                "盈虧比 (Profit Factor)": "-",
                "單筆期望值": "-",
                "平均持倉天數": "-",
                "獲利筆數": 0,
                "虧損筆數": 0,
                "_raw_win_rate": 0.0,
                "_raw_pf": 0.0,
            }

        trades = []
        in_trade = False
        entry_date = None
        entry_price_target = 0.0
        cum_div_per_share_target = 0.0

        open_prices = eval_df["open"].to_numpy(dtype=float)
        close_prices = eval_df["close"].to_numpy(dtype=float)
        div_arr = eval_df["dividend"].to_numpy(dtype=float) if "dividend" in eval_df.columns else np.zeros(len(eval_df))
        fx_arr = eval_df["to_target_fx"].to_numpy(dtype=float) if "to_target_fx" in eval_df.columns else np.ones(len(eval_df))
        pos_arr = pos_s.to_numpy(dtype=int)
        dates = eval_df.index

        for i in range(len(eval_df)):
            pos = int(pos_arr[i])
            prev_pos = int(pos_arr[i - 1]) if i > 0 else 0
            open_p = float(open_prices[i])
            fx = float(fx_arr[i])

            if in_trade and float(div_arr[i]) > 0:
                cum_div_per_share_target += float(div_arr[i]) * fx

            if pos == 1 and prev_pos == 0:
                in_trade = True
                entry_date = dates[i]
                entry_price_target = CostModel.calculate_buy_fill(open_p, cost_fee, cost_slip) * fx
                cum_div_per_share_target = 0.0

            elif pos == 0 and prev_pos == 1 and in_trade:
                exit_date = dates[i]
                exit_price_target = CostModel.calculate_sell_net(open_p, cost_fee, cost_tax, cost_slip) * fx
                total_exit_val = exit_price_target + cum_div_per_share_target
                pnl_pct = (total_exit_val / entry_price_target) - 1.0 if entry_price_target > 0 else 0.0
                holding_days = max(1, (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days)

                trades.append(
                    {
                        "進場日期": pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
                        "出場日期": pd.Timestamp(exit_date).strftime("%Y-%m-%d"),
                        "持倉天數": holding_days,
                        "進場價格": f"{curr_symbol} {entry_price_target:,.2f}",
                        "出場價格": f"{curr_symbol} {exit_price_target:,.2f}",
                        "單筆報酬率": f"{pnl_pct:+.2%}",
                        "_raw_pnl_pct": pnl_pct,
                        "狀態": "獲利 🟢" if pnl_pct > 0 else ("虧損 🔴" if pnl_pct < 0 else "平手 ⚪"),
                    }
                )
                in_trade = False

        if in_trade and entry_date is not None:
            last_date = dates[-1]
            last_close_target = float(close_prices[-1]) * float(fx_arr[-1])
            total_exit_val = last_close_target + cum_div_per_share_target
            pnl_pct = (total_exit_val / entry_price_target) - 1.0 if entry_price_target > 0 else 0.0
            holding_days = max(1, (pd.Timestamp(last_date) - pd.Timestamp(entry_date)).days)
            trades.append(
                {
                    "進場日期": pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
                    "出場日期": pd.Timestamp(last_date).strftime("%Y-%m-%d (持有中)"),
                    "持倉天數": holding_days,
                    "進場價格": f"{curr_symbol} {entry_price_target:,.2f}",
                    "出場價格": f"{curr_symbol} {last_close_target:,.2f}",
                    "單筆報酬率": f"{pnl_pct:+.2%}",
                    "_raw_pnl_pct": pnl_pct,
                    "狀態": "持有中 🟡",
                }
            )

        df_trades = pd.DataFrame(trades)
        if df_trades.empty:
            summary = {
                "總交易次數": 0,
                "已平倉交易": 0,
                "勝率": "-",
                "盈虧比 (Profit Factor)": "-",
                "單筆期望值": "-",
                "平均持倉天數": "-",
                "獲利筆數": 0,
                "虧損筆數": 0,
                "_raw_win_rate": 0.0,
                "_raw_pf": 0.0,
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
            avg_holding = float(np.mean([t["持倉天數"] for t in closed_trades]))
            expectancy = float(np.mean(pnl_list))
        else:
            win_rate = 0.0
            profit_factor = 0.0
            win_count = 0
            loss_count = 0
            avg_holding = 0.0
            expectancy = 0.0

        summary = {
            "總交易次數": len(trades),
            "已平倉交易": len(closed_trades),
            "勝率": f"{win_rate:.1%}" if closed_trades else "-",
            "盈虧比 (Profit Factor)": f"{profit_factor:.2f}" if profit_factor > 0 else "-",
            "單筆期望值": f"{expectancy:+.2%}" if closed_trades else "-",
            "平均持倉天數": f"{avg_holding:.1f} 天" if closed_trades else "-",
            "獲利筆數": win_count,
            "虧損筆數": loss_count,
            "_raw_win_rate": win_rate,
            "_raw_pf": profit_factor,
        }

        display_cols = ["進場日期", "出場日期", "持倉天數", "進場價格", "出場價格", "單筆報酬率", "狀態"]
        return df_trades[display_cols], summary
