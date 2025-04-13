"""
批量评估模块，用于评估存储的对话日志
"""

import json
import logging
import time
import argparse
import concurrent.futures
from typing import Dict, List, Any, Optional
from pathlib import Path
from threading import Lock

from config import RESULT_PATH, EVAL_RESULT_FILE
from models import evaluate_response
from conversation_logger import ConversationLogger
from evaluation import Evaluator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchEvaluator:
    """批量评估器，用于评估存储的对话日志"""
    
    def __init__(self, conversation_logger: ConversationLogger = None):
        """
        初始化批量评估器
        
        Args:
            conversation_logger (ConversationLogger): 对话日志记录器实例
        """
        self.conversation_logger = conversation_logger or ConversationLogger()
        self.evaluator = Evaluator()
        self.eval_lock = Lock()  # 用于保护评估结果
    
    def process_log(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个日志
        
        Args:
            log (Dict[str, Any]): 日志数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        result = {
            "log_id": log.get("log_id", "unknown"),
            "question_id": log.get("question_id", "unknown"),
            "strategy": log.get("strategy", "unknown"),
            "success": False,
            "error": None
        }
        
        try:
            # 评估回答
            eval_result = self.evaluator.evaluate_answer(
                question=log["question"],
                reference_answer=log["reference_answer"],
                model_response={
                    "answer": log["model_answer"],
                    "full_response": log["full_response"],
                    "has_reasoning": log["has_reasoning"],
                    "reasoning": log["reasoning"]
                },
                strategy_name=log["strategy"],
                question_id=log["question_id"],
                question_category=log.get("category", ""),
                question_difficulty=log.get("difficulty", "")
            )
            
            # 标记为已评估
            if "log_file" in log:
                self.conversation_logger.mark_log_as_evaluated(log["log_file"], eval_result)
            
            result["success"] = True
            result["eval_result"] = eval_result
            
            # 使用锁保护结果添加
            with self.eval_lock:
                # 这里不需要额外的操作，因为evaluator.evaluate_answer已经添加了结果
                pass
                
        except Exception as e:
            logger.error(f"评估日志时出错: {e}")
            logger.exception("详细错误：")
            result["error"] = str(e)
        
        return result
    
    def evaluate_logs(
        self, 
        strategy_name: Optional[str] = None, 
        session_id: Optional[str] = None,
        batch_size: int = 10,
        num_threads: int = 1
    ) -> Dict[str, Any]:
        """
        评估对话日志
        
        Args:
            strategy_name (Optional[str]): 策略名称，如果为None则评估所有策略的日志
            session_id (Optional[str]): 会话ID，如果为None则评估所有会话的日志
            batch_size (int): 批处理大小
            num_threads (int): 线程数
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        # 获取未评估的日志
        logs = []
        
        if session_id:
            # 获取指定会话的日志
            all_session_logs = self.conversation_logger.get_logs_by_session(session_id)
            # 过滤未评估的日志
            logs = [log for log in all_session_logs if not log.get("evaluated", False)]
            
            # 如果还指定了策略，进一步过滤
            if strategy_name:
                logs = [log for log in logs if log.get("strategy") == strategy_name]
        else:
            # 获取所有未评估的日志
            logs = self.conversation_logger.get_unevaluated_logs(strategy_name)
        
        logger.info(f"开始评估 {len(logs)} 条对话日志，使用 {num_threads} 个线程")
        
        # 总日志数
        total_logs = len(logs)
        results = []
        
        if num_threads <= 1:
            # 单线程处理
            for i in range(0, total_logs, batch_size):
                batch_logs = logs[i:i+batch_size]
                logger.info(f"正在评估批次 {i//batch_size + 1}/{(total_logs + batch_size - 1)//batch_size}，包含 {len(batch_logs)} 条日志")
                
                for log in batch_logs:
                    result = self.process_log(log)
                    if result["success"]:
                        results.append(result["eval_result"])
                        logger.info(f"已评估日志 {result['question_id']}-{result['strategy']}")
                    else:
                        logger.error(f"评估日志失败: {result['error']}")
        else:
            # 多线程处理
            logger.info(f"使用 {num_threads} 个线程处理 {total_logs} 条日志")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                # 提交所有任务
                future_to_log = {executor.submit(self.process_log, log): i for i, log in enumerate(logs)}
                
                # 处理完成的任务
                completed = 0
                for future in concurrent.futures.as_completed(future_to_log):
                    log_index = future_to_log[future]
                    try:
                        result = future.result()
                        if result["success"]:
                            results.append(result["eval_result"])
                            logger.info(f"已评估日志 {result['question_id']}-{result['strategy']}")
                        else:
                            logger.error(f"评估日志失败: {result['error']}")
                    except Exception as e:
                        logger.error(f"获取任务结果时出错: {e}")
                    
                    completed += 1
                    if completed % 10 == 0 or completed == total_logs:
                        logger.info(f"已完成 {completed}/{total_logs} 条日志评估 ({completed/total_logs*100:.1f}%)")
        
        # 不再保存评估结果文件
        
        return {
            "total_evaluated": len(results),
            "results": results
        }
    
    def generate_session_report(self, session_id: str) -> Dict[str, Any]:
        """
        生成会话报告
        
        Args:
            session_id (str): 会话ID
            
        Returns:
            Dict[str, Any]: 会话报告
        """
        # 获取会话的所有日志
        logs = self.conversation_logger.get_logs_by_session(session_id)
        
        if not logs:
            logger.warning(f"未找到会话 {session_id} 的日志")
            return {"session_id": session_id, "error": "未找到会话日志"}
        
        # 按策略分组
        strategies = {}
        for log in logs:
            strategy_name = log.get("strategy", "unknown")
            if strategy_name not in strategies:
                strategies[strategy_name] = []
            strategies[strategy_name].append(log)
        
        # 计算每个策略的评估指标
        report = {
            "session_id": session_id,
            "timestamp": time.time(),
            "total_questions": len({log.get("question_id") for log in logs}),
            "total_logs": len(logs),
            "strategies": {}
        }
        
        for strategy_name, strategy_logs in strategies.items():
            # 计算已评估的比例
            evaluated_logs = [log for log in strategy_logs if log.get("evaluated", False)]
            
            strategy_report = {
                "total_logs": len(strategy_logs),
                "evaluated_logs": len(evaluated_logs),
                "evaluation_rate": len(evaluated_logs) / len(strategy_logs) if strategy_logs else 0
            }
            
            # 如果有评估结果，计算平均分数
            if evaluated_logs:
                accuracy_scores = []
                reasoning_scores = []
                
                for log in evaluated_logs:
                    eval_result = log.get("evaluation_result", {})
                    metrics = eval_result.get("metrics", {})
                    
                    if "accuracy" in metrics:
                        accuracy_scores.append(float(metrics["accuracy"]["score"]))
                    
                    if "reasoning_quality" in metrics:
                        reasoning_scores.append(float(metrics["reasoning_quality"]["score"]))
                
                if accuracy_scores:
                    strategy_report["accuracy"] = {
                        "average": sum(accuracy_scores) / len(accuracy_scores),
                        "count": len(accuracy_scores)
                    }
                
                if reasoning_scores:
                    strategy_report["reasoning_quality"] = {
                        "average": sum(reasoning_scores) / len(reasoning_scores),
                        "count": len(reasoning_scores)
                    }
            
            report["strategies"][strategy_name] = strategy_report
        
        return report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量评估工具")
    parser.add_argument("--strategy", type=str, help="要评估的策略名称")
    parser.add_argument("--session", type=str, help="要评估的会话ID")
    parser.add_argument("--batch-size", type=int, default=10, help="批处理大小")
    parser.add_argument("--list-sessions", action="store_true", help="列出所有会话")
    parser.add_argument("--report", type=str, help="生成指定会话的报告")
    parser.add_argument("--threads", type=int, default=1, help="线程数")
    
    args = parser.parse_args()
    
    conversation_logger = ConversationLogger()
    batch_evaluator = BatchEvaluator(conversation_logger)
    
    # 列出所有会话
    if args.list_sessions:
        sessions = conversation_logger.get_all_sessions()
        print(f"找到 {len(sessions)} 个会话:")
        for session in sessions:
            print(f"- {session}")
        return
    
    # 生成会话报告
    if args.report:
        report = batch_evaluator.generate_session_report(args.report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    
    # 评估日志
    result = batch_evaluator.evaluate_logs(
        strategy_name=args.strategy,
        session_id=args.session,
        batch_size=args.batch_size,
        num_threads=args.threads
    )
    
    print(f"已评估 {result['total_evaluated']} 条日志")

if __name__ == "__main__":
    main() 