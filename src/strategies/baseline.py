"""
Baseline策略（无CoT）实现
"""

import re
import logging
from typing import Dict, Any
from .base import BaseStrategy
from config import COT_STRATEGIES, LLM_MODEL

# 配置日志
logger = logging.getLogger(__name__)

class Baseline(BaseStrategy):
    """Baseline策略（无CoT）"""
    
    def __init__(self):
        """初始化Baseline策略"""
        config = COT_STRATEGIES.get('baseline', {})
        model = config.get('model', LLM_MODEL)
        super().__init__(
            name=config.get('name', "Baseline (无CoT)"),
            description=config.get('description', "直接向模型提问，不添加任何CoT提示"),
            model=model
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
        
        # 不再提取答案，直接使用完整响应
        logger.info("不提取答案，使用完整响应")
        
        return {
            "full_response": response,
            "answer": response.strip(),  # 使用整个响应作为答案
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
        # 首先尝试判断响应是否为JSON格式
        response_trimmed = response.strip()
        # 检查是否是JSON开头（数组或对象）
        if response_trimmed.startswith('[') or response_trimmed.startswith('{'):
            logger.info("检测到可能的JSON格式响应")
            # 由于响应可能被截断，导致完整解析失败，但我们仍然希望返回原始JSON
            # 即使被截断的JSON也优先返回JSON格式而不是提取数字
            try:
                # 尝试解析JSON
                import json
                json.loads(response_trimmed)
                # 如果成功解析，直接返回整个JSON字符串
                logger.info("成功解析JSON格式响应")
                return response_trimmed
            except json.JSONDecodeError:
                logger.info("JSON解析失败，但仍然返回JSON格式响应")
                # 即使解析失败，仍然返回原始响应，因为它仍然是JSON格式的开头
                # 这比提取单个数字作为答案要好
                return response_trimmed
        
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
