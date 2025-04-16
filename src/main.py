"""
主程序，用于运行LLM评估
"""

import json
import logging
import time
import argparse
import concurrent.futures
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from threading import Lock

from config import COT_STRATEGIES, LLM_MODEL
from models import generate_completion
from vector_db import VectorDatabase
from evaluation import Evaluator
from conversation_logger import ConversationLogger
from dataset_loader import load_livebench_dataset, combine_datasets
from sqlite_backup import SQLiteBackup
from strategies import (
    Baseline,
    ZeroShot,
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
        "zero_shot": ZeroShot(),
        "few_shot": FewShotCoT(vector_db=vector_db),
        "auto_cot": AutoCoT(vector_db=vector_db),
        "auto_reason": AutoReason(),
        "combined": CombinedStrategy(vector_db=vector_db)
    }
    
    logger.info(f"已初始化 {len(strategies)} 个策略")
    return strategies

def process_question_strategy(
    question: Dict[str, Any],
    strategy_name: str,
    strategy: Any,
    evaluator: Optional[Evaluator] = None,
    conversation_logger: Optional[ConversationLogger] = None,
    log_only: bool = False
) -> Dict[str, Any]:
    """
    处理单个问题和策略组合
    
    Args:
        question (Dict[str, Any]): 问题
        strategy_name (str): 策略名称
        strategy (Any): 策略实例
        evaluator (Optional[Evaluator]): 评估器实例
        conversation_logger (Optional[ConversationLogger]): 对话日志记录器实例
        log_only (bool): 是否只记录对话日志而不进行评估
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    question_id = question["id"]
    question_text = question["question"]
    reference_answer = question.get("answer", question.get("reference_answer", ""))
    category = question.get("category", "")
    difficulty = question.get("difficulty", "")
    
    result = {
        "question_id": question_id,
        "strategy_name": strategy_name,
        "success": False,
        "error": None
    }
    
    try:
        # 生成提示
        prompt = strategy.generate_prompt(question_text)
        
        # 获取模型回答
        model_to_use = getattr(strategy, 'model', LLM_MODEL)
        logger.info(f"    使用模型: {model_to_use}")
        
        # 模拟模式，用于测试
        mock_mode = os.environ.get("MOCK_MODE", "").lower() in ("true", "1", "yes")
        if mock_mode:
            logger.info("    使用模拟模式")
            response = f"模拟回答：问题 {question_id}, 策略 {strategy_name}。答案是：42"
        else:
            # 实际调用API
            try:
                response = generate_completion(prompt, model=model_to_use)
            except Exception as api_error:
                logger.error(f"    API调用失败: {api_error}")
                # 不使用模拟模式，直接抛出异常
                raise api_error  # 这样会中断当前处理，不会记录到日志
        
        # 处理回答
        try:
            processed_response = strategy.process_response(response)
            
            # 确保processed_response是字典
            if not isinstance(processed_response, dict):
                logger.warning(f"    策略 {strategy_name} 的process_response未返回字典，将包装为字典")
                processed_response = {
                    "full_response": response,
                    "answer": str(processed_response),
                    "has_reasoning": False
                }
            
            # 确保包含必要的字段
            if "answer" not in processed_response:
                logger.warning(f"    策略 {strategy_name} 的响应中缺少answer字段，将设为完整响应")
                processed_response["answer"] = response
                
            if "full_response" not in processed_response:
                processed_response["full_response"] = response
                
            if "has_reasoning" not in processed_response:
                processed_response["has_reasoning"] = False
                
            if "reasoning" not in processed_response:
                processed_response["reasoning"] = None
                
        except Exception as process_error:
            logger.error(f"    处理响应失败: {process_error}")
            # 提供默认处理结果
            processed_response = {
                "full_response": response,
                "answer": response,
                "has_reasoning": False,
                "reasoning": None
            }
        
        # 如果有对话日志记录器，保存对话日志
        if conversation_logger:
            conversation_logger.log_conversation(
                question=question_text,
                model_response=processed_response,
                strategy_name=strategy_name,
                question_id=question_id,
                reference_answer=reference_answer,
                question_category=category,
                question_difficulty=difficulty,
                metadata=processed_response.get("metadata") if isinstance(processed_response, dict) else None,
                model_name=model_to_use
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
            
            # 如果有对话日志记录器，将评估结果添加到日志
            if conversation_logger:
                log_file = f"{conversation_logger.log_dir}/{strategy_name}/{question_id}-{int(time.time())}.json"
                # 尝试找到最近记录的日志文件
                log_files = list(Path(f"{conversation_logger.log_dir}/{strategy_name}").glob(f"{question_id}-*.json"))
                if log_files:
                    # 按修改时间排序，取最新的
                    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    log_file = str(log_files[0])
                    
                    # 添加评估结果到日志
                    accuracy_score = eval_result['metrics']['accuracy']['score']
                    accuracy_explanation = eval_result['metrics']['accuracy']['explanation']
                    
                    # 构建其他评估指标
                    other_metrics = {}
                    for metric_name, metric_value in eval_result['metrics'].items():
                        if metric_name != 'accuracy':
                            other_metrics[metric_name] = metric_value
                    
                    # 将评估结果添加到日志
                    conversation_logger.add_evaluation_metrics(
                        log_file=log_file,
                        accuracy_score=accuracy_score,
                        accuracy_explanation=accuracy_explanation,
                        metrics=other_metrics
                    )
            
            result["success"] = True
            result["eval_result"] = eval_result
            
        else:
            # 如果只记录日志，也标记为成功
            result["success"] = True
            
    except Exception as e:
        logger.error(f"处理时出错: {e}")
        logger.exception("详细错误：")
        result["error"] = str(e)
    
    return result

def run_evaluation(
    questions: List[Dict[str, Any]], 
    strategies: Dict[str, Any],
    evaluator: Optional[Evaluator] = None,
    conversation_logger: Optional[ConversationLogger] = None,
    strategy_filter: Optional[List[str]] = None,
    question_filter: Optional[List[str]] = None,
    max_questions: Optional[int] = None,
    log_only: bool = False,
    num_threads: int = 1,
    sqlite_backup: Optional[SQLiteBackup] = None,
    dataset: str = None,
    model: str = None
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
        num_threads (int): 线程数，用于并行处理评估任务
        sqlite_backup (Optional[SQLiteBackup]): SQLite备份实例
        dataset (str): 数据集名称
        model (str): 模型名称
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
    
    # 决定是否使用多线程
    if num_threads <= 1:
        # 单线程处理
        logger.info("使用单线程处理评估任务")
        
        for i, question in enumerate(filtered_questions):
            question_id = question["id"]
            logger.info(f"处理问题 {i+1}/{total_questions}: {question_id}")
            
            for strategy_name, strategy in filtered_strategies.items():
                logger.info(f"  使用策略 {strategy_name}")
                
                try:
                    result = process_question_strategy(
                        question=question,
                        strategy_name=strategy_name,
                        strategy=strategy,
                        evaluator=evaluator,
                        conversation_logger=conversation_logger,
                        log_only=log_only
                    )
                    
                    if result["success"]:
                        logger.info(f"完成问题 {question_id} 使用策略 {strategy_name} 的评估")
                    else:
                        logger.error(f"问题 {question_id} 使用策略 {strategy_name} 的评估失败: {result['error']}")
                except Exception as e:
                    # 检查是否是API调用相关错误
                    if "API调用失败" in str(e) or "account balance is insufficient" in str(e):
                        logger.warning(f"问题 {question_id} 使用策略 {strategy_name} 的API调用失败，跳过此评估: {e}")
                    else:
                        logger.error(f"处理问题 {question_id} 使用策略 {strategy_name} 时出错: {e}")
    else:
        # 多线程处理
        logger.info(f"使用多线程处理评估任务 (线程数: {num_threads})")
        
        # 创建任务列表
        tasks = []
        for question in filtered_questions:
            for strategy_name, strategy in filtered_strategies.items():
                tasks.append((question, strategy_name, strategy))
        
        logger.info(f"共创建了 {len(tasks)} 个评估任务")
        
        # 使用线程池处理任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(
                    process_question_strategy, 
                    question, strategy_name, strategy, 
                    evaluator, conversation_logger, log_only
                ): (question["id"], strategy_name) 
                for question, strategy_name, strategy in tasks
            }
            
            # 处理完成的任务
            completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                question_id, strategy_name = future_to_task[future]
                try:
                    result = future.result()
                    if result["success"]:
                        logger.info(f"完成问题 {question_id} 使用策略 {strategy_name} 的评估")
                    else:
                        logger.error(f"问题 {question_id} 使用策略 {strategy_name} 的评估失败: {result['error']}")
                except Exception as e:
                    # 检查是否是API调用相关错误
                    if "API调用失败" in str(e) or "account balance is insufficient" in str(e):
                        logger.warning(f"问题 {question_id} 使用策略 {strategy_name} 的API调用失败，跳过此评估: {e}")
                    else:
                        logger.error(f"获取任务结果时出错: {e}")
                
                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    logger.info(f"已完成 {completed}/{len(tasks)} 个评估任务 ({completed/len(tasks)*100:.1f}%)")
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    logger.info(f"处理完成，总耗时: {elapsed_time:.2f}秒")
    
    # 如果不是只记录日志且有评估器，打印摘要
    if not log_only and evaluator:
        # 如果有SQLite备份实例，备份结果
        if sqlite_backup and conversation_logger:
            session_id = conversation_logger.session_id
            try:
                logger.info(f"备份评估结果到SQLite数据库，会话ID: {session_id}")
                sqlite_backup.backup_all_results(evaluator.results, session_id, dataset, model)
                logger.info("评估结果备份完成")
            except Exception as e:
                logger.error(f"备份评估结果时出错: {e}")
        
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
    parser.add_argument("--threads", type=int, default=1, help="线程数，用于并行处理评估任务")
    
    # SQLite备份相关参数
    parser.add_argument("--sqlite-backup", action="store_true", help="是否启用SQLite备份")
    parser.add_argument("--sqlite-db", type=str, default="data/backup.db", help="SQLite数据库路径")
    
    # Hugging Face数据集相关参数
    parser.add_argument("--use-hf-dataset", action="store_true", help="使用Hugging Face数据集")
    parser.add_argument("--hf-dataset", type=str, nargs="+", 
                        choices=["livebench/math", "livebench/reasoning", "livebench/data_analysis"],
                        help="要使用的Hugging Face数据集名称")
    parser.add_argument("--hf-split", type=str, default="test", help="数据集分割")
    parser.add_argument("--max-samples-per-dataset", type=int, help="每个数据集的最大样本数")
    
    # 本地JSON文件和缓存目录
    parser.add_argument("--local-json-dir", type=str, help="本地JSON文件目录，如果提供则从本地加载而非HF")
    parser.add_argument("--cache-dir", type=str, default="data/hf_datasets", help="HF数据集缓存目录")
    parser.add_argument("--save-datasets", action="store_true", help="是否保存HF数据集到本地JSON文件")
    parser.add_argument("--save-dir", type=str, default="data/processed_datasets", 
                       help="保存处理后的数据集到JSON文件的目录")
    
    # 向量数据库相关参数
    parser.add_argument("--vector-db-dir", type=str, default="data/vector_store", 
                       help="向量数据库目录")
    parser.add_argument("--separate-db", action="store_true", 
                       help="为每个数据集使用单独的向量数据库（仅在处理多个数据集时有效）")
    
    # 输出与保存相关
    parser.add_argument("--result-prefix", type=str, help="结果文件前缀，用于区分不同评估任务")
    parser.add_argument("--model", type=str, help="指定使用的模型名称（仅用于记录）")
    
    args = parser.parse_args()
    
    # 初始化SQLite备份（如果启用）
    sqlite_backup = None
    if args.sqlite_backup:
        try:
            sqlite_backup = SQLiteBackup(db_path=args.sqlite_db)
            logger.info(f"已启用SQLite备份，数据库路径: {args.sqlite_db}")
        except Exception as e:
            logger.error(f"初始化SQLite备份失败: {e}")
            sqlite_backup = None
    
    # 初始化questions变量
    questions = None
    
    # 如果要使用HF数据集或者本地JSON数据集
    if args.use_hf_dataset and args.hf_dataset:
        # 如果提供了多个数据集并且要求使用单独的向量数据库
        if len(args.hf_dataset) > 1 and args.separate_db:
            logger.info(f"将分别对 {len(args.hf_dataset)} 个数据集进行评估，每个数据集使用独立的向量数据库")
            
            # 为每个数据集单独运行评估
            for dataset_name in args.hf_dataset:
                # 设置单个数据集的本地目录和结果前缀
                dataset_simple_name = dataset_name.split("/")[-1]
                if args.result_prefix:
                    result_prefix = f"{args.result_prefix}_{dataset_simple_name}"
                else:
                    result_prefix = dataset_simple_name
                
                logger.info(f"开始评估数据集: {dataset_name}, 结果前缀: {result_prefix}")
                
                # 加载单个数据集
                questions = combine_datasets(
                    [dataset_name], 
                    max_samples_per_dataset=args.max_samples_per_dataset,
                    cache_dir=args.cache_dir,
                    local_json_dir=args.local_json_dir,
                    save_dir=args.save_dir if args.save_datasets else None
                )
                
                if not questions:
                    logger.error(f"未能加载数据集 {dataset_name}，跳过评估")
                    continue
                
                # 初始化该数据集的向量数据库
                if args.vector_db_dir:
                    db_path = f"{args.vector_db_dir}_{dataset_simple_name}"
                else:
                    db_path = f"data/vector_store_{dataset_simple_name}"
                
                vector_db = VectorDatabase(db_path)
                
                # 如果强制重建或者数据库为空，则加载问题
                if args.rebuild_db or len(vector_db.metadata) == 0:
                    logger.info(f"正在初始化向量数据库 {db_path}...")
                    vector_db.clear()
                    
                    # 逐个添加问题
                    for q in questions:
                        metadata = {k: v for k, v in q.items() if k != 'question'}
                        vector_db.add_question(q['question'], metadata)
                    
                    logger.info(f"向量数据库初始化完成，包含 {len(vector_db.metadata)} 个问题")
                else:
                    logger.info(f"使用现有向量数据库 {db_path}，包含 {len(vector_db.metadata)} 个问题")
                
                # 初始化策略
                strategies = init_strategies(vector_db)
                
                # 初始化评估器和对话日志记录器
                evaluator = None if args.log_only else Evaluator(result_prefix=result_prefix)
                conversation_logger = ConversationLogger(result_prefix=result_prefix, sqlite_backup=sqlite_backup)
                
                # 如果指定了会话ID，设置会话ID
                if args.session_id:
                    conversation_logger.session_id = f"{args.session_id}_{dataset_simple_name}"
                
                # 运行评估
                run_evaluation(
                    questions=questions,
                    strategies=strategies,
                    evaluator=evaluator,
                    conversation_logger=conversation_logger,
                    strategy_filter=args.strategies,
                    question_filter=args.question_ids,
                    max_questions=args.max_questions,
                    log_only=args.log_only,
                    num_threads=args.threads,
                    sqlite_backup=sqlite_backup,
                    dataset=dataset_name,
                    model=args.model
                )
        else:
            # 加载所有指定的数据集
            questions = combine_datasets(
                args.hf_dataset, 
                max_samples_per_dataset=args.max_samples_per_dataset,
                cache_dir=args.cache_dir,
                local_json_dir=args.local_json_dir,
                save_dir=args.save_dir if args.save_datasets else None
            )
    else:
        # 不使用HF数据集，加载本地问题集
        questions = load_questions(args.questions)
        logger.info(f"从本地加载问题集: {args.questions}")
    
    if not questions or len(questions) == 0:
        logger.error("未能加载问题集，程序退出")
        return
    
    logger.info(f"成功加载 {len(questions)} 个问题")
    
    # 初始化向量数据库
    vector_db = VectorDatabase(args.vector_db_dir)
    
    # 如果强制重建或者数据库为空，则加载问题
    if args.rebuild_db or len(vector_db.metadata) == 0:
        logger.info("正在初始化向量数据库...")
        vector_db.clear()
        
        # 逐个添加问题
        for q in questions:
            metadata = {k: v for k, v in q.items() if k != 'question'}
            vector_db.add_question(q['question'], metadata)
        
        logger.info(f"向量数据库初始化完成，包含 {len(vector_db.metadata)} 个问题")
    else:
        logger.info(f"使用现有向量数据库，包含 {len(vector_db.metadata)} 个问题")
    
    # 初始化策略
    strategies = init_strategies(vector_db)
    
    # 如果仅显示摘要，则加载现有结果并打印摘要
    if args.summary_only:
        evaluator = Evaluator(result_prefix=args.result_prefix)
        evaluator.load_results()
        evaluator.print_summary()
        return
    
    # 初始化评估器和对话日志记录器
    evaluator = None if args.log_only else Evaluator(result_prefix=args.result_prefix)
    conversation_logger = ConversationLogger(result_prefix=args.result_prefix, sqlite_backup=sqlite_backup)
    
    # 如果指定了会话ID，设置会话ID
    if args.session_id:
        conversation_logger.session_id = args.session_id
    
    # 确定数据集名称
    dataset_name = args.hf_dataset[0] if args.hf_dataset else None
    
    # 运行评估
    run_evaluation(
        questions=questions,
        strategies=strategies,
        evaluator=evaluator,
        conversation_logger=conversation_logger,
        strategy_filter=args.strategies,
        question_filter=args.question_ids,
        max_questions=args.max_questions,
        log_only=args.log_only,
        num_threads=args.threads,
        sqlite_backup=sqlite_backup,
        dataset=dataset_name,
        model=args.model
    )
    
    # 关闭SQLite连接
    if sqlite_backup:
        sqlite_backup.close()

if __name__ == "__main__":
    main()
