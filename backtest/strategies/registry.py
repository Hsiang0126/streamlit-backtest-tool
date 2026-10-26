"""
策略註冊中心 (Strategy Registry)
採用外掛式註冊工廠模式，新增策略免改核心與 UI。
"""
from typing import Dict, Type, List, Optional
from strategies.base import BaseStrategy


class StrategyRegistry:
    """策略註冊與工廠中心。"""
    _strategies: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, strategy_cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
        """類別裝飾器：註冊策略。"""
        if not issubclass(strategy_cls, BaseStrategy):
            raise TypeError(f"{strategy_cls.__name__} 必須繼承自 BaseStrategy")
        cls._strategies[strategy_cls.display_name] = strategy_cls
        return strategy_cls

    @classmethod
    def get(cls, display_name: str) -> Optional[Type[BaseStrategy]]:
        return cls._strategies.get(display_name)

    @classmethod
    def list_names(cls) -> List[str]:
        return list(cls._strategies.keys())

    @classmethod
    def get_all(cls) -> Dict[str, Type[BaseStrategy]]:
        return dict(cls._strategies)


def register_strategy(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
    """裝飾器語法糖：@register_strategy"""
    return StrategyRegistry.register(cls)