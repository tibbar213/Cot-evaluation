"""
Zero-shot CoT策略实现
"""

import re
import logging
from typing import Dict, Any
from .base import BaseStrategy
from config import COT_STRATEGIES

# 配置日志
logger = logging.getLogger(__name__)

class ZeroShotCoT(BaseStrategy):
    """Zero-shot CoT策略"""
    
    def __init__(self):
        """初始化Zero-shot CoT策略"""
        config = COT_STRATEGIES.get('zero_shot', {})
        super().__init__(
            name=config.get('name', "Zero-shot CoT"),
            description=config.get('description', "在提示的最后添加'Let's think step by step.'")
        )
        self.prompt_suffix = config.get('prompt_suffix', "Let's think step by step.")
    
    def generate_prompt(self, question: str) -> str:
        """
        生成提示
        
        Args:
            question (str): 问题
            
        Returns:
            str: 生成的提示
        """
        logger.info(f"为问题生成Zero-shot CoT提示: {question}")
        logger.info(f"使用提示后缀: {self.prompt_suffix}")
        
        # 在问题后添加提示后缀
        prompt = f"{question}\n{self.prompt_suffix}"
        
        logger.info("Zero-shot CoT提示生成完成")
        return prompt
    
    def process_response(self, response: str) -> Dict[str, Any]:
        """
        处理模型响应
        
        Args:
            response (str): 模型响应
            
        Returns:
            Dict[str, Any]: 处理后的响应，包含答案和其他信息
        """
        logger.info("处理模型响应")
        
        # 尝试从响应中提取答案和推理过程
        reasoning, answer = self._extract_reasoning_and_answer(response)
        
        logger.info(f"提取的答案: '{answer}'")
        logger.info(f"提取的推理过程长度: {len(reasoning) if reasoning else 0}")
        
        return {
            "full_response": response,
            "answer": answer,
            "has_reasoning": bool(reasoning),
            "reasoning": reasoning,
            # 添加元数据
            "metadata": {
                "strategy_details": {
                    "name": self.name,
                    "description": self.description,
                    "prompt_suffix": self.prompt_suffix
                }
            }
        }
    
    def _extract_reasoning_and_answer(self, response: str) -> tuple:
        """
        从响应中提取推理过程和答案
        
        Args:
            response (str): 模型响应
            
        Returns:
            tuple: (推理过程, 答案)
        """
        # 将响应分成多行
        lines = response.strip().split('\n')
        
        # 尝试找到答案行（通常在最后）
        answer_line = ""
        reasoning_lines = []
        
        # 检查是否有明确的"答案是"或"所以"等标记
        answer_patterns = [
            r"答案是[：:]\s*(.+)",
            r"结果是[：:]\s*(.+)",
            r"答案为[：:]\s*(.+)",
            r"结果为[：:]\s*(.+)",
            r"所以[，,]?\s*(.+)",
            r"因此[，,]?\s*(.+)",
            r"综上所述[，,]?\s*(.+)",
        ]
        
        # 先查找明确的答案标记
        for i, line in enumerate(lines):
            for pattern in answer_patterns:
                match = re.search(pattern, line)
                if match:
                    answer_line = match.group(1).strip()
                    reasoning_lines = lines[:i]
                    break
            if answer_line:
                break
        
        # 如果没有找到明确的答案标记，假设最后一行是答案，前面的都是推理
        if not answer_line and lines:
            answer_line = lines[-1]
            reasoning_lines = lines[:-1]
        
        # 如果推理部分为空但有多行，至少保留一些内容作为推理
        if not reasoning_lines and len(lines) > 1:
            reasoning_lines = lines[:-1]
            answer_line = lines[-1]
        
        # 组合推理行
        reasoning = '\n'.join(reasoning_lines).strip()
        
        # 如果答案行中没有明确的数字或短语，尝试提取
        answer = answer_line
        
        # 尝试从答案行中提取数字
        numbers = re.findall(r'\d+', answer_line)
        if numbers:
            answer = numbers[-1]
        
        # 如果还是没有答案，尝试从整个响应中提取最后一个数字
        if not answer and response:
            numbers = re.findall(r'\d+', response)
            if numbers:
                answer = numbers[-1]
        
        return reasoning, answer
