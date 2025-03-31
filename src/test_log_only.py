"""
测试脚本，仅用于测试对话日志功能
"""

import json
import logging
import time
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

from models import generate_completion
from strategies import Baseline
from conversation_logger import ConversationLogger

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_questions(file_path: str) -> List[Dict[str, Any]]:
    """
    加载问题集
    
    Args:
        file_path (str): 问题集文件路径
        
    Returns:
        List[Dict[str, Any]]: 问题集
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        logger.info(f"已加载 {len(questions)} 个问题")
        return questions
    except Exception as e:
        logger.error(f"加载问题集时出错: {e}")
        return []

def run_test(
    questions: List[Dict[str, Any]], 
    max_questions: Optional[int] = None
) -> None:
    """
    运行测试
    
    Args:
        questions (List[Dict[str, Any]]): 问题集
        max_questions (Optional[int]): 最大问题数
    """
    # 创建对话日志记录器
    conversation_logger = ConversationLogger()
    logger.info(f"创建会话: {conversation_logger.session_id}")
    
    # 创建策略 (只使用Baseline策略，不需要向量数据库)
    strategy = Baseline()
    strategy_name = "baseline"
    
    # 限制问题数量
    if max_questions and max_questions < len(questions):
        filtered_questions = questions[:max_questions]
    else:
        filtered_questions = questions
    
    logger.info(f"将处理 {len(filtered_questions)} 个问题")
    
    # 开始处理
    start_time = time.time()
    
    for i, question in enumerate(filtered_questions):
        question_id = question["id"]
        question_text = question["question"]
        reference_answer = question["answer"]
        category = question.get("category", "")
        difficulty = question.get("difficulty", "")
        
        logger.info(f"处理问题 {i+1}/{len(filtered_questions)}: {question_id}")
        
        try:
            # 生成提示
            prompt = strategy.generate_prompt(question_text)
            
            # 获取模型回答
            response = generate_completion(prompt)
            
            # 处理回答
            processed_response = strategy.process_response(response)
            
            # 保存对话日志
            conversation_logger.log_conversation(
                question=question_text,
                model_response=processed_response,
                strategy_name=strategy_name,
                question_id=question_id,
                reference_answer=reference_answer,
                question_category=category,
                question_difficulty=difficulty
            )
            logger.info(f"已记录对话日志")
                
        except Exception as e:
            logger.error(f"处理时出错: {e}")
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    logger.info(f"处理完成，总耗时: {elapsed_time:.2f}秒")
    logger.info(f"对话日志已保存在: {conversation_logger.log_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="对话日志测试工具")
    parser.add_argument("--questions", type=str, default="data/questions.json", help="问题集文件路径")
    parser.add_argument("--max-questions", type=int, default=3, help="最大问题数")
    parser.add_argument("--session-id", type=str, help="指定会话ID，如果不指定则使用当前时间戳")
    
    args = parser.parse_args()
    
    # 加载问题集
    questions = load_questions(args.questions)
    if not questions:
        logger.error("未能加载问题集，程序退出")
        return
    
    # 运行测试
    run_test(
        questions=questions,
        max_questions=args.max_questions
    )

if __name__ == "__main__":
    main() 