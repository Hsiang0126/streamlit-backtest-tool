"""定期定額與資金管理策略。"""
from typing import List, Dict, Any
from strategies.base import BaseStrategy, ParamSpec
from strategies.registry import register_strategy


@register_strategy
class DCAStrategy(BaseStrategy):
    strategy_id = "dca_buy_and_hold"
    display_name = "定期定額 / 買入持有 (DCA / Buy & Hold)"
    description = "經典定投策略：期初全額買入，每期固定注入資金自動扣款再投資。"
    is_timing = False

    @classmethod
    def get_param_specs(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                key="freq",
                label="扣款頻率",
                param_type="select",
                default="每月 (Monthly)",
                options=["每月 (Monthly)", "每週 (Weekly)", "每 10 交易日"]
            )
        ]


@register_strategy
class DipBuyerStrategy(BaseStrategy):
    strategy_id = "dip_buyer"
    display_name = "跌破均線且創新低買入 (Dip DCA)"
    description = "逢低分批加碼策略：價格跌破指定均線且出現區間新低時加碼。"
    is_timing = False

    @classmethod
    def get_param_specs(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                key="sma_period",
                label="基準均線週期 (日)",
                param_type="int",
                default=60,
                min_value=5,
                max_value=300,
                step=5
            ),
            ParamSpec(
                key="buy_amount",
                label="每次抄底買入金額",
                param_type="float",
                default=50000.0,
                min_value=1000.0,
                step=5000.0
            ),
            ParamSpec(
                key="min_interval",
                label="加碼最小間隔天數",
                param_type="int",
                default=3,
                min_value=1,
                max_value=60,
                step=1
            )
        ]
