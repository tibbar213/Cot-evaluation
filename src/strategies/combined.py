"""
组合策略（Auto-CoT + AutoReason）实现
"""

import re
import logging
import json
from typing import Dict, Any, List, Tuple
from .base import BaseStrategy
from config import COT_STRATEGIES
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
        super().__init__(
            name=config.get('name', "Auto-CoT + AutoReason"),
            description=config.get('description', "结合Auto-CoT和AutoReason的优势")
        )
        self.num_examples = config.get('num_examples', 2)
        self.reasoning_model = config.get('reasoning_model')
        self.vector_db = vector_db or VectorDatabase()
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
        self._last_example_chains = []
        
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
            self._last_example_chains.append({
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
        
        # 尝试从响应中提取答案和推理过程
        reasoning, answer = self._extract_reasoning_and_answer(response)
        
        result = {
            "full_response": response,
            "answer": answer,
            "has_reasoning": bool(reasoning),
            "reasoning": reasoning,
            # 添加额外信息，用于在conversation_logs中记录
            "metadata": {
                "strategy_details": {
                    "name": self.name,
                    "description": self.description,
                    "reasoning_model": self.reasoning_model,
                    "num_examples": self.num_examples
                },
                "similar_questions": getattr(self, "_last_similar_questions", []),
                "example_reasoning_chains": getattr(self, "_last_example_chains", [])
            }
        }
        
        logger.info(f"提取的答案: '{answer}'")
        logger.info(f"是否包含推理: {bool(reasoning)}")
        if reasoning:
            logger.info(f"提取的推理: {reasoning[:100]}...")
        
        return result
    
    def _extract_reasoning_and_answer(self, response: str) -> tuple:
        """
        从响应中提取推理过程和答案
        
        Args:
            response (str): 模型响应
            
        Returns:
            tuple: (推理过程, 答案)
        """
        logger.info("从响应中提取推理过程和答案")
        
        # 尝试匹配"答案是X"或"结果是X"等模式
        answer_patterns = [
            r"答案[是为:：]\s*(.+?)[\s\.。]",
            r"结果[是为:：]\s*(.+?)[\s\.。]",
            r"等于[：:]\s*(.+?)[\s\.。]",
            r"得到[：:]\s*(.+?)[\s\.。]",
            r"有[：:]\s*(.+?)[\s\.。]个",
            r"一共有[：:]\s*(.+?)[\s\.。]",
            r"总共有[：:]\s*(.+?)[\s\.。]",
            r"最终答案[：:]\s*(.+?)[\s\.。]",
        ]
        
        answer = ""
        matched_pattern = None
        
        for pattern in answer_patterns:
            match = re.search(pattern, response)
            if match:
                answer = match.group(1).strip()
                matched_pattern = pattern
                logger.info(f"使用模式 '{pattern}' 匹配到答案: '{answer}'")
                break
        
        # 如果没有找到匹配，尝试找到最后一个数字
        if not answer:
            numbers = re.findall(r'\d+', response)
            if numbers:
                answer = numbers[-1]
                logger.info(f"使用最后一个数字作为答案: '{answer}'")
            else:
                logger.warning("无法从响应中提取答案")
        
        # 尝试提取推理过程
        if answer and answer in response:
            # 找到答案在响应中的位置
            answer_pos = response.find(answer)
            if answer_pos > 0:
                reasoning = response[:answer_pos].strip()
                logger.info(f"从答案位置提取推理过程，长度: {len(reasoning)}")
            else:
                reasoning = ""
                logger.warning("答案位于响应开头，无法提取推理过程")
        else:
            # 如果没有找到答案，假设整个响应都是推理过程
            reasoning = response.strip()
            logger.info("使用整个响应作为推理过程")
            
            # 如果推理过程以"答案："结尾，移除它
            if re.search(r'答案[：:]\s*$', reasoning):
                reasoning = re.sub(r'答案[：:]\s*$', '', reasoning).strip()
                logger.info("从推理过程中移除'答案:'后缀")
        
        return reasoning, answer
