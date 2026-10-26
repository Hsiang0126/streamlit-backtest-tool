"""
策略抽象基類與參數規格定義 (Strategy Base Interface)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd


@dataclass
class ParamSpec:
    key: str
    label: str
    param_type: str  # "int", "float", "select", "bool"
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    help_text: Optional[str] = None


class BaseStrategy(ABC):
    """所有回測策略的抽象基類。"""

    strategy_id: str = "base_strategy"
    display_name: str = "基礎策略"
    description: str = ""
    is_timing: bool = True  # True 表示為 0/1 倉位擇時策略；False 為定投/階梯資金型策略

    @classmethod
    @abstractmethod
    def get_param_specs(cls) -> List[ParamSpec]:
        """返回策略參數規格列表，供 UI 動態生成控制項。"""
        pass

    @classmethod
    def generate_signals(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """產生擇時二元持倉訊號 (0: 空倉, 1: 全倉)。"""
        return pd.Series(1, index=df.index)