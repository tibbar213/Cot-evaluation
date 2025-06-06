"""
组合策略（Auto-CoT + AutoReason）实现
"""

import re
import logging
import json
from typing import Dict, Any, List, Tuple
from .base import BaseStrategy
from config import COT_STRATEGIES, REASONING_MODEL, LLM_MODEL
from vector_db import VectorDatabase
from models import generate_completion, generate_reasoning_chain

# 获取日志器
logger = logging.getLogger(__name__)

class CombinedStrategy(BaseStrategy):
    """组合策略（Auto-CoT + AutoReason）"""
    
    def __init__(self, vector_db: VectorDatabase = None):
        """
        初始化组合策略
        
        Args:
            vector_db (VectorDatabase, optional): 向量数据库实例
        """
        config = COT_STRATEGIES.get('combined', {})
        model = config.get('model', LLM_MODEL)
        super().__init__(
            name=config.get('name', "Auto-CoT + AutoReason"),
            description=config.get('description', "结合Auto-CoT和AutoReason的优势"),
            model=model
        )
        self.num_examples = config.get('num_examples', 2)
        self.reasoning_model = config.get('reasoning_model', REASONING_MODEL)
        self.vector_db = vector_db or VectorDatabase()
        
        # 存储最近一次查询的相似问题和为它们生成的推理链
        self._last_similar_questions = []
        self._last_example_reasoning_chains = []
        
        logger.info(f"初始化组合策略 - 示例数量: {self.num_examples}, 推理模型: {self.reasoning_model}")
    
    def generate_prompt(self, question: str) -> str:
        """
        生成提示
        
        Args:
            question (str): 问题
            
        Returns:
            str: 生成的提示
        """
        logger.info(f"为问题生成提示: {question}")
        
        # 检索相似问题
        similar_questions = self._get_similar_questions(question, self.num_examples)
        logger.info(f"找到 {len(similar_questions)} 个相似问题")
        
        # 保存相似问题，供process_response使用
        self._last_similar_questions = similar_questions
        self._last_example_reasoning_chains = []
        
        # 为相似问题生成推理链
        examples = []
        for i, (q_id, q_text, q_answer, similarity) in enumerate(similar_questions):
            logger.info(f"为示例 #{i+1} 生成推理链 - 问题ID: {q_id}, 相似度: {similarity}")
            logger.info(f"示例问题: {q_text}")
            logger.info(f"示例答案: {q_answer}")
            
            # 为示例生成推理链
            reasoning_chain = self._generate_reasoning_chain(q_text)
            logger.info(f"生成的推理链: {reasoning_chain}")
            
            # 保存推理链，供process_response使用
            self._last_example_reasoning_chains.append({
                "question_id": q_id,
                "question": q_text,
                "answer": q_answer,
                "similarity": similarity,
                "reasoning_chain": reasoning_chain
            })
            
            # 构建示例
            example = f"""
            问题: {q_text}
            推理链:
            {reasoning_chain}
            答案: {q_answer}
            """
            examples.append(example)
            logger.info(f"示例 #{i+1} 构建完成")
        
        # 组合示例
        examples_text = "\n\n".join(examples)
        
        # 构建提示
        prompt = f"""
        以下是一些示例问题、推理链和答案:
        
        {examples_text}
        
        现在，请回答这个问题，遵循上面示例中的推理步骤:
        问题: {question}
        """
        
        logger.info("提示生成完成")
        return prompt
    
    def _get_similar_questions(self, question: str, num_examples: int) -> List[Tuple[str, str, str, float]]:
        """
        获取与问题相似的问题
        
        Args:
            question (str): 问题
            num_examples (int): 示例数量
            
        Returns:
            List[Tuple[str, str, str, float]]: 相似问题列表，每项包含(问题ID, 问题文本, 答案, 相似度)
        """
        logger.info(f"在向量数据库中搜索与问题相似的 {num_examples} 个示例: {question}")
        
        # 使用向量数据库检索相似问题，排除完全相同的问题
        similar_questions = self.vector_db.get_similar_questions(question, k=num_examples, exclude_exact_match=True)
        
        # 转换为所需的格式
        formatted_results = []
        for i, (q_text, q_answer) in enumerate(similar_questions):
            # 我们可能没有问题ID和相似度，所以使用索引作为ID，计算一个模拟的相似度分数
            similarity_score = 1.0 - (0.1 * i)  # 模拟相似度分数，第一个最相似
            formatted_results.append((str(i), q_text, q_answer, similarity_score))
            logger.info(f"找到相似问题 #{i+1}: '{q_text}', 答案: '{q_answer}', 相似度: {similarity_score:.4f}")
            
        return formatted_results
    
    def _generate_reasoning_chain(self, question: str) -> str:
        """
        为问题生成推理链
        
        Args:
            question (str): 问题
            
        Returns:
            str: 生成的推理链
        """
        logger.info(f"为问题生成推理链: {question}")
        
        # 构建生成推理链的提示模板
        prompt_template = """
        您将获得一个问题，并使用该问题将其分解为一系列逻辑推理轨迹。
        仅写下推理过程，不要自己回答问题。
        
        问题: {question}
        """
        
        # 生成推理链
        try:
            logger.info(f"调用推理模型 {self.reasoning_model} 生成推理链")
            reasoning_chain = generate_reasoning_chain(
                question=question,
                model=self.reasoning_model,
                prompt_template=prompt_template
            )
            logger.info(f"推理链生成成功: {reasoning_chain[:100]}...")
            return reasoning_chain
        except Exception as e:
            logger.error(f"生成推理链失败: {e}")
            # 如果生成失败，返回一个简单的推理链
            fallback_chain = f"1. 分析问题\n2. 确定关键信息\n3. 计算答案"
            logger.warning(f"使用备用推理链: {fallback_chain}")
            return fallback_chain
    
    def process_response(self, response: str) -> Dict[str, Any]:
        """
        处理模型响应
        
        Args:
            response (str): 模型响应
            
        Returns:
            Dict[str, Any]: 处理后的响应，包含答案和其他信息
        """
        logger.info("处理模型响应")
        logger.info(f"原始响应: {response}")
        
        # 不再提取答案和推理，直接使用完整响应
        logger.info("不提取答案和推理，使用完整响应")
        
        result = {
            "full_response": response,
            "answer": response.strip(),  # 使用整个响应作为答案
            "has_reasoning": True,  # 假设包含推理
            "reasoning": response.strip(),  # 使用整个响应作为推理
            # 添加额外信息，用于在conversation_logs中记录
            "metadata": {
                "strategy_details": {
                    "name": self.name,
                    "description": self.description,
                    "reasoning_model": self.reasoning_model,
                    "num_examples": self.num_examples
                },
                "similar_questions": getattr(self, "_last_similar_questions", []),
                "example_reasoning_chains": getattr(self, "_last_example_reasoning_chains", [])
            }
        }
        
        return result
