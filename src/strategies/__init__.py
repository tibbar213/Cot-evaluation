"""
CoT策略模块
"""

from .zero_shot import ZeroShotCoT
from .few_shot import FewShotCoT
from .auto_cot import AutoCoT
from .auto_reason import AutoReason
from .combined import CombinedStrategy
from .baseline import Baseline

# 导出所有策略类
__all__ = [
    'ZeroShotCoT',
    'FewShotCoT',
    'AutoCoT',
    'AutoReason',
    'CombinedStrategy',
    'Baseline'
]
