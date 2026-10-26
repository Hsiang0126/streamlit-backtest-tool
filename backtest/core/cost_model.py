"""
成本模型模組 (Cost Model)
支援台股上市/上櫃 (個股與 ETF 差異化稅率)、美股、加密貨幣之手續費、證券交易稅與市場滑價計算。
"""
from dataclasses import dataclass


@dataclass
class AssetCostStructure:
    fee_rate: float       # 單邊手續費率 (例如 0.00071 代表 0.071%)
    tax_rate: float       # 交易稅率 (僅在賣出時徵收，例如台股個股 0.3%，ETF 0.1%)
    slippage_rate: float  # 單邊預估滑價 (例如 0.0002 代表 0.02%)


class CostModel:
    """交易成本與市場衝擊滑點計算器。"""

    @staticmethod
    def get_default_costs(ticker_symbol: str) -> AssetCostStructure:
        clean = ticker_symbol.strip().upper()

        # 判斷是否為台股 (包含純數字、.TW、.TWO)
        is_tw = clean.isdigit() or clean.endswith((".TW", ".TWO"))
        if is_tw:
            symbol_core = clean.split(".")[0]
            # 台股 ETF (代號以 00 或 01 開頭) 證交稅為 0.1%，一般個股為 0.3%
            if symbol_core.startswith("00") or symbol_core.startswith("01"):
                return AssetCostStructure(fee_rate=0.00071, tax_rate=0.0010, slippage_rate=0.0002)
            else:
                return AssetCostStructure(fee_rate=0.00071, tax_rate=0.0030, slippage_rate=0.0003)

        # 加密貨幣 (常帶有 -USD, USDT, BTC, ETH 等後綴)
        if "-USD" in clean or clean.endswith("USDT"):
            return AssetCostStructure(fee_rate=0.0010, tax_rate=0.0, slippage_rate=0.0010)

        # 預設為美股主流市場 (零佣金、免交易稅、微量滑點)
        return AssetCostStructure(fee_rate=0.0000, tax_rate=0.0, slippage_rate=0.0003)

    @staticmethod
    def calculate_buy_fill(price: float, fee_rate: float, slippage_rate: float) -> float:
        """買入每股實際成本價格 (含向上滑價與手續費)。"""
        if price <= 0:
            raise ValueError(f"買入參考價必須為正數，目前值：{price}")
        return price * (1.0 + slippage_rate) * (1.0 + fee_rate)

    @staticmethod
    def calculate_sell_net(price: float, fee_rate: float, tax_rate: float, slippage_rate: float) -> float:
        """賣出每股實收淨價格 (扣除向下滑價、手續費與證券交易稅)。"""
        if price <= 0:
            raise ValueError(f"賣出參考價必須為正數，目前值：{price}")
        net = price * (1.0 - slippage_rate) * (1.0 - fee_rate) * (1.0 - tax_rate)
        return max(0.0, net)
