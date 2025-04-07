"""
Few-shot CoT策略实现
"""

import re
import logging
from typing import Dict, Any, List, Tuple
from .base import BaseStrategy
from config import COT_STRATEGIES
from vector_db import VectorDatabase

# 配置日志
logger = logging.getLogger(__name__)

class FewShotCoT(BaseStrategy):
    """Few-shot CoT策略"""
    
    def __init__(self, vector_db: VectorDatabase = None):
        """
        初始化Few-shot CoT策略
        
        Args:
            vector_db (VectorDatabase, optional): 向量数据库实例
        """
        config = COT_STRATEGIES.get('few_shot', {})
        super().__init__(
            name=config.get('name', "Few-shot CoT"),
            description=config.get('description', "使用向量数据库检索相似问题及其答案作为示例")
        )
        self.num_examples = config.get('num_examples', 2)
        self.vector_db = vector_db or VectorDatabase()
    
    def generate_prompt(self, question: str) -> str:
        """
        生成提示
        
        Args:
            question (str): 问题
            
        Returns:
            str: 生成的提示
        """
        logger.info(f"为问题生成Few-shot提示: {question}")
        
        # 从向量数据库中检索相似问题及其答案，排除与当前问题完全相同的问题
        examples = self.vector_db.get_similar_questions(question, k=self.num_examples, exclude_exact_match=True)
        logger.info(f"从向量数据库检索到 {len(examples)} 个相似问题")
        
        # 为元数据存储相似问题
        self._last_similar_questions = []
        for i, (q, a) in enumerate(examples):
            similarity = 1.0 - (0.1 * i)  # 模拟相似度分数
            self._last_similar_questions.append((str(i), q, a, similarity))
            logger.info(f"相似问题 #{i+1}: '{q}', 答案: '{a}', 相似度: {similarity:.4f}")
        
        # 构建Few-shot提示
        prompt = ""
        
        # 添加示例
        for i, (example_q, example_a) in enumerate(examples):
            prompt += f"Q: {example_q}\nA: {example_a}\n\n"
            logger.info(f"添加示例 #{i+1} 到提示")
        
        # 添加目标问题
        prompt += f"Q: {question}\nA:"
        
        logger.info("Few-shot提示生成完成")
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
        
        # 尝试从响应中提取答案
        answer = self._extract_answer(response)
        logger.info(f"提取的答案: '{answer}'")
        
        return {
            "full_response": response,
            "answer": answer,
            "has_reasoning": False,  # Few-shot CoT不显式要求推理过程
            "reasoning": None,
            # 添加元数据
            "metadata": {
                "strategy_details": {
                    "name": self.name,
                    "description": self.description,
                    "num_examples": self.num_examples
                },
                "similar_questions": getattr(self, "_last_similar_questions", [])
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
                return response_trimmed
        
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
