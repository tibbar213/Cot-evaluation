"""
Zero-shot CoT策略实现
"""

import re
import logging
from typing import Dict, Any
from .base import BaseStrategy
from config import COT_STRATEGIES, LLM_MODEL

# 配置日志
logger = logging.getLogger(__name__)

class ZeroShot(BaseStrategy):
    """Zero-shot CoT策略"""
    
    def __init__(self):
        """初始化Zero-shot策略"""
        config = COT_STRATEGIES.get('zero_shot', {})
        model = config.get('model', LLM_MODEL)
        super().__init__(
            name=config.get('name', "Zero-shot CoT"),
            description=config.get('description', "在提示的最后添加'Let's think step by step.'"),
            model=model
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
        
        # 不再提取答案和推理，直接使用完整响应
        logger.info("不提取答案和推理，使用完整响应")
        
        return {
            "full_response": response,
            "answer": response.strip(),  # 使用整个响应作为答案
            "has_reasoning": True,  # 假设Zero-shot CoT产生的响应包含推理
            "reasoning": response.strip(),  # 使用整个响应作为推理
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
        # 检查是否是JSON格式响应
        response_trimmed = response.strip()
        if response_trimmed.startswith('[') or response_trimmed.startswith('{'):
            logger.info("检测到可能的JSON格式响应")
            # 由于响应可能被截断，导致完整解析失败，但我们仍然希望返回原始JSON
            try:
                # 尝试解析JSON
                import json
                json.loads(response_trimmed)
                # 如果成功解析，直接返回空推理和整个JSON字符串
                logger.info("成功解析JSON格式响应")
                return "", response_trimmed
            except json.JSONDecodeError:
                logger.info("JSON解析失败，但仍然返回JSON格式响应")
                # 即使解析失败，仍然返回原始响应，因为它仍然是JSON格式的开头
                return "", response_trimmed
            
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
