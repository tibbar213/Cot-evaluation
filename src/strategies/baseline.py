"""
Baseline策略（无CoT）实现
"""

import re
import logging
from typing import Dict, Any
from .base import BaseStrategy

# 配置日志
logger = logging.getLogger(__name__)

class Baseline(BaseStrategy):
    """Baseline策略（无CoT）"""
    
    def __init__(self):
        """初始化Baseline策略"""
        super().__init__(
            name="Baseline (无CoT)",
            description="直接向模型提问，不添加任何CoT提示"
        )
    
    def generate_prompt(self, question: str) -> str:
        """
        生成提示
        
        Args:
            question (str): 问题
            
        Returns:
            str: 生成的提示
        """
        logger.info(f"为问题生成Baseline提示: {question}")
        logger.info("Baseline策略不添加任何CoT提示")
        
        # 直接返回问题，不添加任何CoT提示
        return question
    
    def process_response(self, response: str) -> Dict[str, Any]:
        """
        处理模型响应
        
        Args:
            response (str): 模型响应
            
        Returns:
            Dict[str, Any]: 处理后的响应，包含答案和其他信息
        """
        logger.info("处理模型响应")
        
        # 尝试从响应中提取数字或简短答案
        answer = self._extract_answer(response)
        logger.info(f"提取的答案: '{answer}'")
        
        return {
            "full_response": response,
            "answer": answer,
            "has_reasoning": False,
            "reasoning": None,
            # 添加元数据
            "metadata": {
                "strategy_details": {
                    "name": self.name,
                    "description": self.description
                }
            }
        }
    
    def _extract_answer(self, response: str) -> str:
        """
        从响应中提取答案
        
        Args:
            response (str): 模型响应
            
        Returns:
            str: 提取的答案
        """
        # 尝试找到最后一个数字或者最后一句话作为答案
        
        # 尝试匹配"答案是X"或"结果是X"等模式
        answer_patterns = [
            r"答案是[：:]\s*(.+?)[\s\.。]",
            r"结果是[：:]\s*(.+?)[\s\.。]",
            r"答案为[：:]\s*(.+?)[\s\.。]",
            r"结果为[：:]\s*(.+?)[\s\.。]",
            r"等于[：:]\s*(.+?)[\s\.。]",
            r"一共有[：:]\s*(.+?)[\s\.。]",
            r"总共有[：:]\s*(.+?)[\s\.。]",
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()
        
        # 尝试匹配最后一个数字
        numbers = re.findall(r'\d+', response)
        if numbers:
            return numbers[-1]
        
        # 如果以上都失败，返回最后一句话
        sentences = re.split(r'[.。!！?？]', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            return sentences[-1]
        
        # 如果以上都失败，返回整个响应
        return response.strip()
