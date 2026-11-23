"""
成本模型模組 (Cost Model)
支援台股 (個股、股票型 ETF、免徵證交稅之債券 ETF 及自訂券商手續費折數)、
美股與加密貨幣之手續費、證券交易稅與市場滑價計算。
"""
from dataclasses import dataclass
import re


@dataclass
class AssetCostStructure:
    fee_rate: float      # 手續費率 (如 0.0007125 代表台股 5 折後 0.07125%)
    tax_rate: float      # 交易稅率 (僅賣出時徵收，如台股個股 0.3%，ETF 0.1%，債券 ETF 0%)
    slippage_rate: float # 單邊預估滑價 (如 0.0002 代表 0.02%)


class CostModel:
    """交易成本與滑點計算器。"""

    TW_BASE_FEE_RATE = 0.001425

    @staticmethod
    def is_taiwan_ticker(ticker: str) -> bool:
        """判定是否為台股上市櫃代碼 (支援 4~6 碼及含英文字尾，如 2330, 0050, 00631L, 00679B, 00981A, 2881A)。"""
        clean = str(ticker).strip().upper()
        return clean.endswith((".TW", ".TWO")) or bool(re.match(r"^\d{4,6}[A-Z]?$", clean))

    @classmethod
    def _is_taiwan_symbol(cls, clean: str) -> bool:
        return cls.is_taiwan_ticker(clean)

    @classmethod
    def _is_tw_bond_etf(cls, symbol_core: str) -> bool:
        return symbol_core.startswith("00") and symbol_core.endswith("B")

    @classmethod
    def get_default_costs(cls, ticker_symbol: str, tw_fee_discount: float = 0.5) -> AssetCostStructure:
        clean = ticker_symbol.strip().upper()
        discount = max(0.0, float(tw_fee_discount))

        if cls._is_taiwan_symbol(clean):
            symbol_core = clean.split(".")[0]
            fee = cls.TW_BASE_FEE_RATE * discount
            # 台股債券 ETF (如 00679B, 00687B, 00937B) 依法免徵證券交易稅
            if cls._is_tw_bond_etf(symbol_core):
                return AssetCostStructure(fee_rate=fee, tax_rate=0.0, slippage_rate=0.0002)
            # 台股股票型 ETF (00 / 01 開頭) 證交稅為千分之一 (0.1%)
            if symbol_core.startswith(("00", "01")):
                return AssetCostStructure(fee_rate=fee, tax_rate=0.0010, slippage_rate=0.0002)
            # 台股一般個股 / 特別股：證交稅千分之三 (0.3%)
            return AssetCostStructure(fee_rate=fee, tax_rate=0.0030, slippage_rate=0.0003)

        if "-USD" in clean or clean.endswith("USDT"):
            return AssetCostStructure(fee_rate=0.0010, tax_rate=0.0, slippage_rate=0.0010)

        # 美股與其他國際市場 (免手續費券商基準 + 微幅滑價)
        return AssetCostStructure(fee_rate=0.0, tax_rate=0.0, slippage_rate=0.0003)

    @classmethod
    def describe_cost(cls, ticker_symbol: str, cost: AssetCostStructure) -> str:
        clean = ticker_symbol.strip().upper()
        if cls._is_taiwan_symbol(clean):
            core = clean.split(".")[0]
            if cls._is_tw_bond_etf(core):
                kind = "台股債券 ETF (免證交稅)"
            elif core.startswith(("00", "01")):
                kind = "台股 ETF"
            else:
                kind = "台股個股"
        elif "-USD" in clean or clean.endswith("USDT"):
            kind = "加密貨幣"
        else:
            kind = "美股/海外"

        return (
            f"{kind}：手續費 {cost.fee_rate:.4%}｜賣出稅 {cost.tax_rate:.2%}｜滑價 {cost.slippage_rate:.3%}"
        )

    @staticmethod
    def calculate_buy_fill(price: float, fee_rate: float, slippage_rate: float) -> float:
        """買入每股實際成本價格 (含向上滑價與手續費)"""
        if price <= 0:
            raise ValueError(f"買入參考價必須為正數，目前值：{price}")
        return price * (1.0 + slippage_rate) * (1.0 + fee_rate)

    @staticmethod
    def calculate_sell_net(price: float, fee_rate: float, tax_rate: float, slippage_rate: float) -> float:
        """賣出每股實收淨價格 (扣除向下滑價、手續費與證券交易稅)"""
        if price <= 0:
            raise ValueError(f"賣出參考價必須為正數，目前值：{price}")
        net = price * (1.0 - slippage_rate) * (1.0 - fee_rate) * (1.0 - tax_rate)
        return max(0.0, net)
