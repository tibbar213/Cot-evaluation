"""
AutoReason策略实现
"""

import re
import logging
from typing import Dict, Any
from .base import BaseStrategy
from config import COT_STRATEGIES, REASONING_MODEL
from models import generate_reasoning_chain

# 配置日志
logger = logging.getLogger(__name__)

class AutoReason(BaseStrategy):
    """AutoReason策略"""
    
    def __init__(self):
        """初始化AutoReason策略"""
        config = COT_STRATEGIES.get('auto_reason', {})
        super().__init__(
            name=config.get('name', "AutoReason"),
            description=config.get('description', "使用强模型生成详细的推理链")
        )
        self.reasoning_prompt = config.get('reasoning_prompt', 
            "您将获得一个问题，并使用该问题将其分解为一系列逻辑推理轨迹。仅写下推理过程，不要给出答案")
        self.reasoning_model = config.get('reasoning_model', REASONING_MODEL)
    
    def generate_prompt(self, question: str) -> str:
        """
        生成提示
        
        Args:
            question (str): 问题
            
        Returns:
            str: 生成的提示
        """
        # 使用强模型生成推理链
        reasoning_chain = self._generate_reasoning_chain(question)
        
        # 保存生成的推理链，供process_response使用
        self._last_reasoning_chain = reasoning_chain
        
        # 构建AutoReason提示
        prompt = f"""
        {question}
        
        (推理链：
        {reasoning_chain}
        )
        """
        
        return prompt
    
    def _generate_reasoning_chain(self, question: str) -> str:
        """
        为问题生成推理链
        
        Args:
            question (str): 问题
            
        Returns:
            str: 生成的推理链
        """
        # 构建生成推理链的提示模板
        prompt_template = f"{self.reasoning_prompt}\n\n问题: {{question}}"
        
        # 生成推理链
        try:
            reasoning_chain = generate_reasoning_chain(
                question=question,
                model=REASONING_MODEL,
                prompt_template=prompt_template
            )
            return reasoning_chain
        except Exception as e:
            # 如果生成失败，返回一个简单的推理链
            return f"推理链：\n1. 分析问题\n2. 确定关键信息\n3. 计算答案"
    
    def process_response(self, response: str) -> Dict[str, Any]:
        """
        处理模型响应
        
        Args:
            response (str): 模型响应
            
        Returns:
            Dict[str, Any]: 处理后的响应，包含答案和其他信息
        """
        # 尝试从响应中提取答案
        answer = self._extract_answer(response)
        logger.info(f"提取的答案: {answer}")
        
        # 尝试提取推理过程（如果有）
        reasoning = self._extract_reasoning(response)
        logger.info(f"提取的推理过程长度: {len(reasoning) if reasoning else 0}")
        
        return {
            "full_response": response,
            "answer": answer,
            "has_reasoning": bool(reasoning),
            "reasoning": reasoning,
            # 添加元数据信息
            "metadata": {
                "strategy_details": {
                    "name": self.name,
                    "description": self.description,
                    "reasoning_model": self.reasoning_model
                },
                "generated_reasoning_chain": getattr(self, "_last_reasoning_chain", "")
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
        # 尝试匹配数字答案
        answer_match = re.search(r'答案[是为：:]\s*(\d+)', response)
        if answer_match:
            return answer_match.group(1)
        
        # 尝试匹配最后一个数字
        numbers = re.findall(r'\d+', response)
        if numbers:
            return numbers[-1]
        
        # 如果没有找到数字，返回空字符串
        return ""
    
    def _extract_reasoning(self, response: str) -> str:
        """
        从响应中提取推理过程
        
        Args:
            response (str): 模型响应
            
        Returns:
            str: 提取的推理过程
        """
        # 尝试匹配推理链部分
        reasoning_match = re.search(r'\(推理链：(.*?)\)', response, re.DOTALL)
        if reasoning_match:
            return reasoning_match.group(1).strip()
        
        # 如果没有明确的推理链标记，尝试提取所有非答案部分
        answer = self._extract_answer(response)
        if answer and answer in response:
            # 将响应分成多行
            lines = response.strip().split('\n')
            
            # 找到答案所在行
            answer_line_idx = -1
            for i, line in enumerate(lines):
                if answer in line:
                    answer_line_idx = i
                    break
            
            # 如果找到答案行，提取之前的所有内容作为推理
            if answer_line_idx > 0:
                return '\n'.join(lines[:answer_line_idx]).strip()
        
        # 如果以上都失败，返回空字符串
        return ""
