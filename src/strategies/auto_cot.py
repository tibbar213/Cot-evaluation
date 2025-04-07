"""
Auto-CoT策略实现
"""

import re
import logging
from typing import Dict, Any, List, Tuple
from .base import BaseStrategy
from config import COT_STRATEGIES, LLM_MODEL
from vector_db import VectorDatabase
from models import generate_completion

# 配置日志
logger = logging.getLogger(__name__)

class AutoCoT(BaseStrategy):
    """Auto-CoT策略"""
    
    def __init__(self, vector_db: VectorDatabase = None):
        """
        初始化Auto-CoT策略
        
        Args:
            vector_db (VectorDatabase, optional): 向量数据库实例
        """
        config = COT_STRATEGIES.get('auto_cot', {})
        model = config.get('model', LLM_MODEL)
        super().__init__(
            name=config.get('name', "Auto-CoT"),
            description=config.get('description', "使用向量数据库检索相似问题，并为其生成CoT推理过程"),
            model=model
        )
        self.num_examples = config.get('num_examples', 2)
        self.cot_prefix = config.get('cot_prefix', "Let's think step by step。")
        self.vector_db = vector_db or VectorDatabase()
    
    def generate_prompt(self, question: str) -> str:
        """
        生成提示
        
        Args:
            question (str): 问题
            
        Returns:
            str: 生成的提示
        """
        logger.info(f"为问题生成Auto-CoT提示: {question}")
        
        # 从向量数据库中检索相似问题及其答案
        examples = self.vector_db.get_similar_questions(question, k=self.num_examples, exclude_exact_match=True)
        logger.info(f"从向量数据库检索到 {len(examples)} 个相似问题")
        
        # 为元数据存储相似问题
        self._last_similar_questions = []
        for i, (q, a) in enumerate(examples):
            similarity = 1.0 - (0.1 * i)  # 模拟相似度分数
            self._last_similar_questions.append((str(i), q, a, similarity))
            logger.info(f"相似问题 #{i+1}: '{q}', 答案: '{a}', 相似度: {similarity:.4f}")
        
        # 为每个示例生成CoT推理过程
        examples_with_cot = []
        self._last_example_cots = []
        
        for i, (example_q, example_a) in enumerate(examples):
            # 生成CoT推理过程
            logger.info(f"为示例 #{i+1} 生成CoT推理过程")
            cot = self._generate_cot_for_example(example_q, example_a)
            examples_with_cot.append((example_q, cot))
            
            # 存储生成的CoT用于元数据
            self._last_example_cots.append({
                "question": example_q,
                "answer": example_a,
                "cot": cot
            })
            logger.info(f"示例 #{i+1} 的CoT长度: {len(cot)}")
        
        # 构建Auto-CoT提示
        prompt = ""
        
        # 添加示例
        for i, (example_q, example_cot) in enumerate(examples_with_cot):
            prompt += f"Q: {example_q}\nA: {example_cot}\n\n"
        
        # 添加目标问题
        prompt += f"Q: {question}\nA:"
        
        logger.info("Auto-CoT提示生成完成")
        return prompt
    
    def _generate_cot_for_example(self, question: str, answer: str) -> str:
        """
        为示例问题生成CoT推理过程
        
        Args:
            question (str): 示例问题
            answer (str): 示例答案
            
        Returns:
            str: 生成的CoT推理过程
        """
        # 构建生成CoT的提示
        prompt = f"""
        请为以下问题生成一个详细的思维链（Chain of Thought）推理过程，最后得出给定的答案。
        
        问题: {question}
        答案: {answer}
        
        请以"{self.cot_prefix}"开始，然后详细解释解题思路，包括每一步的推理过程，最后得出答案。
        """
        
        # 生成CoT
        try:
            cot = generate_completion(prompt, temperature=0.3)
            
            # 如果生成的CoT没有以指定前缀开始，添加前缀
            if not cot.strip().startswith(self.cot_prefix):
                cot = f"{self.cot_prefix} {cot}"
            
            return cot
        except Exception as e:
            # 如果生成失败，返回一个简单的CoT
            return f"{self.cot_prefix}首先，我们分析问题。{question}根据问题，我们可以直接计算得出答案。答案是{answer}。"
    
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
                    "num_examples": self.num_examples,
                    "cot_prefix": self.cot_prefix
                },
                "similar_questions": getattr(self, "_last_similar_questions", []),
                "example_cots": getattr(self, "_last_example_cots", [])
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
