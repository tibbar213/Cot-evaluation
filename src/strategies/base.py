"""
CoT策略基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional

from config import LLM_MODEL

class BaseStrategy(ABC):
    """CoT策略基类"""
    
    def __init__(self, name: str, description: str, model: str = LLM_MODEL):
        """
        初始化策略
        
        Args:
            name (str): 策略名称
            description (str): 策略描述
            model (str): 使用的模型名称
        """
        self.name = name
        self.description = description
        self.model = model
    
    @abstractmethod
    def generate_prompt(self, question: str) -> str:
        """
        生成提示
        
        Args:
            question (str): 问题
            
        Returns:
            str: 生成的提示
        """
        pass
    
    @abstractmethod
    def process_response(self, response: str) -> Dict[str, Any]:
        """
        处理模型响应
        
        Args:
            response (str): 模型响应
            
        Returns:
            Dict[str, Any]: 处理后的响应，包含答案和其他信息
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将策略转换为字典
        
        Returns:
            Dict[str, Any]: 策略字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model
        }
