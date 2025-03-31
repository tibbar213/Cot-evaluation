"""
主程序，用于运行LLM评估
"""

import json
import logging
import time
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

from config import COT_STRATEGIES
from models import generate_completion
from vector_db import VectorDatabase
from evaluation import Evaluator
from conversation_logger import ConversationLogger  # 导入新添加的对话日志记录器
from strategies import (
    Baseline,
    ZeroShotCoT,
    FewShotCoT,
    AutoCoT,
    AutoReason,
    CombinedStrategy
)

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

def init_vector_db(questions: List[Dict[str, Any]], force_rebuild: bool = False) -> VectorDatabase:
    """
    初始化向量数据库
    
    Args:
        questions (List[Dict[str, Any]]): 问题集
        force_rebuild (bool): 是否强制重建
        
    Returns:
        VectorDatabase: 向量数据库实例
    """
    vector_db = VectorDatabase()
    
    # 如果强制重建或者数据库为空，则加载问题
    if force_rebuild or len(vector_db.metadata) == 0:
        logger.info("正在初始化向量数据库...")
        vector_db.clear()
        
        # 逐个添加问题
        for q in questions:
            metadata = {k: v for k, v in q.items() if k != 'question'}
            vector_db.add_question(q['question'], metadata)
        
        logger.info(f"向量数据库初始化完成，包含 {len(vector_db.metadata)} 个问题")
    else:
        logger.info(f"使用现有向量数据库，包含 {len(vector_db.metadata)} 个问题")
    
    return vector_db

def init_strategies(vector_db: VectorDatabase) -> Dict[str, Any]:
    """
    初始化策略
    
    Args:
        vector_db (VectorDatabase): 向量数据库实例
        
    Returns:
        Dict[str, Any]: 策略字典
    """
    strategies = {
        "baseline": Baseline(),
        "zero_shot": ZeroShotCoT(),
        "few_shot": FewShotCoT(vector_db=vector_db),
        "auto_cot": AutoCoT(vector_db=vector_db),
        "auto_reason": AutoReason(),
        "combined": CombinedStrategy(vector_db=vector_db)
    }
    
    logger.info(f"已初始化 {len(strategies)} 个策略")
    return strategies

def run_evaluation(
    questions: List[Dict[str, Any]], 
    strategies: Dict[str, Any],
    evaluator: Optional[Evaluator] = None,
    conversation_logger: Optional[ConversationLogger] = None,
    strategy_filter: Optional[List[str]] = None,
    question_filter: Optional[List[str]] = None,
    max_questions: Optional[int] = None,
    log_only: bool = False
) -> None:
    """
    运行评估
    
    Args:
        questions (List[Dict[str, Any]]): 问题集
        strategies (Dict[str, Any]): 策略字典
        evaluator (Optional[Evaluator]): 评估器实例
        conversation_logger (Optional[ConversationLogger]): 对话日志记录器实例
        strategy_filter (Optional[List[str]]): 策略过滤器
        question_filter (Optional[List[str]]): 问题过滤器
        max_questions (Optional[int]): 最大问题数
        log_only (bool): 是否只记录对话日志而不进行评估
    """
    # 过滤策略
    if strategy_filter:
        filtered_strategies = {k: v for k, v in strategies.items() if k in strategy_filter}
    else:
        filtered_strategies = strategies
    
    logger.info(f"将使用 {len(filtered_strategies)} 个策略进行评估")
    
    # 过滤问题
    if question_filter:
        filtered_questions = [q for q in questions if q["id"] in question_filter]
    else:
        filtered_questions = questions
    
    # 限制问题数量
    if max_questions and max_questions < len(filtered_questions):
        filtered_questions = filtered_questions[:max_questions]
    
    logger.info(f"将评估 {len(filtered_questions)} 个问题")
    
    # 开始评估
    total_questions = len(filtered_questions)
    total_strategies = len(filtered_strategies)
    total_evaluations = total_questions * total_strategies
    
    if log_only:
        logger.info(f"开始生成并记录对话日志，总共 {total_evaluations} 次对话")
    else:
        logger.info(f"开始评估，总共 {total_evaluations} 次评估")
    
    start_time = time.time()
    
    for i, question in enumerate(filtered_questions):
        question_id = question["id"]
        question_text = question["question"]
        reference_answer = question["answer"]
        category = question.get("category", "")
        difficulty = question.get("difficulty", "")
        
        logger.info(f"处理问题 {i+1}/{total_questions}: {question_id}")
        
        for strategy_name, strategy in filtered_strategies.items():
            logger.info(f"  使用策略 {strategy_name}")
            
            try:
                # 生成提示
                prompt = strategy.generate_prompt(question_text)
                
                # 获取模型回答
                response = generate_completion(prompt)
                
                # 处理回答
                processed_response = strategy.process_response(response)
                
                # 如果有对话日志记录器，保存对话日志
                if conversation_logger:
                    conversation_logger.log_conversation(
                        question=question_text,
                        model_response=processed_response,
                        strategy_name=strategy_name,
                        question_id=question_id,
                        reference_answer=reference_answer,
                        question_category=category,
                        question_difficulty=difficulty
                    )
                    logger.info(f"    已记录对话日志")
                
                # 如果不是只记录日志且有评估器，评估回答
                if not log_only and evaluator:
                    eval_result = evaluator.evaluate_answer(
                        question=question_text,
                        reference_answer=reference_answer,
                        model_response=processed_response,
                        strategy_name=strategy_name,
                        question_id=question_id,
                        question_category=category,
                        question_difficulty=difficulty
                    )
                    
                    logger.info(f"    准确率: {eval_result['metrics']['accuracy']['score']}")
                
            except Exception as e:
                logger.error(f"处理时出错: {e}")
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    logger.info(f"处理完成，总耗时: {elapsed_time:.2f}秒")
    
    # 如果不是只记录日志且有评估器，保存结果并打印摘要
    if not log_only and evaluator:
        # 保存结果
        evaluator.save_results()
        
        # 打印摘要
        evaluator.print_summary()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM评估工具")
    parser.add_argument("--questions", type=str, default="data/questions.json", help="问题集文件路径")
    parser.add_argument("--rebuild-db", action="store_true", help="重建向量数据库")
    parser.add_argument("--strategies", type=str, nargs="+", help="要评估的策略列表")
    parser.add_argument("--question-ids", type=str, nargs="+", help="要评估的问题ID列表")
    parser.add_argument("--max-questions", type=int, help="最大问题数")
    parser.add_argument("--summary-only", action="store_true", help="仅显示摘要，不进行评估")
    parser.add_argument("--log-only", action="store_true", help="仅记录对话日志，不进行评估")
    parser.add_argument("--session-id", type=str, help="指定会话ID，如果不指定则使用当前时间戳")
    
    args = parser.parse_args()
    
    # 加载问题集
    questions = load_questions(args.questions)
    if not questions:
        logger.error("未能加载问题集，程序退出")
        return
    
    # 初始化向量数据库
    vector_db = init_vector_db(questions, force_rebuild=args.rebuild_db)
    
    # 初始化策略
    strategies = init_strategies(vector_db)
    
    # 如果仅显示摘要，则加载现有结果并打印摘要
    if args.summary_only:
        evaluator = Evaluator()
        evaluator.load_results()
        evaluator.print_summary()
        return
    
    # 初始化评估器和对话日志记录器
    evaluator = None if args.log_only else Evaluator()
    conversation_logger = ConversationLogger()
    
    # 如果指定了会话ID，设置会话ID
    if args.session_id:
        conversation_logger.session_id = args.session_id
    
    # 运行评估
    run_evaluation(
        questions=questions,
        strategies=strategies,
        evaluator=evaluator,
        conversation_logger=conversation_logger,
        strategy_filter=args.strategies,
        question_filter=args.question_ids,
        max_questions=args.max_questions,
        log_only=args.log_only
    )

if __name__ == "__main__":
    main()
