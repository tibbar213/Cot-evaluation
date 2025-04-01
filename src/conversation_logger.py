"""
对话日志记录模块，用于存储模型对话并后续评估
"""

import json
import time
import logging
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

from config import RESULT_PATH

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationLogger:
    """对话日志记录器，用于存储模型对话并后续评估"""
    
    def __init__(self, log_dir: str = os.path.join(RESULT_PATH, "conversation_logs"), result_prefix: Optional[str] = None):
        """
        初始化对话日志记录器
        
        Args:
            log_dir (str): 日志保存目录
            result_prefix (Optional[str]): 结果文件前缀，用于区分不同评估任务
        """
        if result_prefix:
            log_dir = os.path.join(log_dir, result_prefix)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前会话ID，用当前时间戳表示
        self.session_id = str(int(time.time()))
        logger.info(f"创建新的会话: {self.session_id}")
    
    def log_conversation(
        self, 
        question: str, 
        model_response: Dict[str, Any],
        strategy_name: str,
        question_id: str,
        reference_answer: str = "",
        question_category: str = "",
        question_difficulty: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        记录对话日志
        
        Args:
            question (str): 问题
            model_response (Dict[str, Any]): 模型回答，包含answer和reasoning等
            strategy_name (str): 策略名称
            question_id (str): 问题ID
            reference_answer (str): 参考答案
            question_category (str): 问题类别
            question_difficulty (str): 问题难度
            metadata (Optional[Dict[str, Any]]): 额外的元数据
            
        Returns:
            str: 日志文件路径
        """
        # 创建策略目录
        strategy_dir = self.log_dir / strategy_name
        strategy_dir.mkdir(exist_ok=True)
        
        # 创建日志对象
        log_entry = {
            "question_id": question_id,
            "question": question,
            "reference_answer": reference_answer,
            "model_answer": model_response.get("answer", ""),
            "full_response": model_response.get("full_response", ""),
            "has_reasoning": model_response.get("has_reasoning", False),
            "reasoning": model_response.get("reasoning", None),
            "strategy": strategy_name,
            "category": question_category,
            "difficulty": question_difficulty,
            "timestamp": time.time(),
            "session_id": self.session_id,
            "evaluated": False
        }
        
        # 添加metadata字段（如果存在）
        if metadata:
            log_entry["metadata"] = metadata
            logger.info(f"添加元数据到日志：包含 {len(metadata)} 个字段")
        
        # 日志文件名格式: question_id-timestamp.json
        filename = f"{question_id}-{int(time.time())}.json"
        log_file = strategy_dir / filename
        
        # 保存日志
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        
        logger.info(f"对话日志已保存: {log_file}")
        return str(log_file)
    
    def get_unevaluated_logs(self, strategy_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取未评估的对话日志
        
        Args:
            strategy_name (Optional[str]): 策略名称，如果为None则获取所有策略的日志
            
        Returns:
            List[Dict[str, Any]]: 未评估的对话日志列表
        """
        logs = []
        
        # 如果指定了策略，只搜索该策略目录
        if strategy_name:
            strategy_dirs = [self.log_dir / strategy_name]
        else:
            # 否则搜索所有策略目录
            strategy_dirs = [d for d in self.log_dir.iterdir() if d.is_dir()]
        
        # 遍历策略目录
        for strategy_dir in strategy_dirs:
            if not strategy_dir.exists():
                continue
                
            # 遍历日志文件
            for log_file in strategy_dir.glob("*.json"):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_entry = json.load(f)
                        
                    # 只返回未评估的日志
                    if not log_entry.get("evaluated", False):
                        # 添加文件路径信息
                        log_entry["log_file"] = str(log_file)
                        logs.append(log_entry)
                        
                except Exception as e:
                    logger.error(f"读取日志文件 {log_file} 时出错: {e}")
        
        logger.info(f"发现 {len(logs)} 条未评估的对话日志")
        return logs
    
    def mark_log_as_evaluated(self, log_file: str, evaluation_result: Dict[str, Any]) -> bool:
        """
        将日志标记为已评估
        
        Args:
            log_file (str): 日志文件路径
            evaluation_result (Dict[str, Any]): 评估结果
            
        Returns:
            bool: 是否成功
        """
        try:
            log_path = Path(log_file)
            
            # 读取日志文件
            with open(log_path, 'r', encoding='utf-8') as f:
                log_entry = json.load(f)
            
            # 更新评估状态和结果
            log_entry["evaluated"] = True
            log_entry["evaluation_result"] = evaluation_result
            log_entry["evaluation_timestamp"] = time.time()
            
            # 保存更新后的日志
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已将日志 {log_file} 标记为已评估")
            return True
            
        except Exception as e:
            logger.error(f"标记日志 {log_file} 为已评估时出错: {e}")
            return False
    
    def get_logs_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取指定会话的所有日志
        
        Args:
            session_id (str): 会话ID
            
        Returns:
            List[Dict[str, Any]]: 日志列表
        """
        logs = []
        
        # 遍历所有策略目录
        for strategy_dir in [d for d in self.log_dir.iterdir() if d.is_dir()]:
            # 遍历日志文件
            for log_file in strategy_dir.glob("*.json"):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_entry = json.load(f)
                    
                    # 只返回指定会话的日志
                    if log_entry.get("session_id") == session_id:
                        logs.append(log_entry)
                        
                except Exception as e:
                    logger.error(f"读取日志文件 {log_file} 时出错: {e}")
        
        return logs
    
    def get_all_sessions(self) -> List[str]:
        """
        获取所有会话ID
        
        Returns:
            List[str]: 会话ID列表
        """
        sessions = set()
        
        # 遍历所有策略目录
        for strategy_dir in [d for d in self.log_dir.iterdir() if d.is_dir()]:
            # 遍历日志文件
            for log_file in strategy_dir.glob("*.json"):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_entry = json.load(f)
                    
                    # 收集会话ID
                    if "session_id" in log_entry:
                        sessions.add(log_entry["session_id"])
                        
                except Exception as e:
                    logger.error(f"读取日志文件 {log_file} 时出错: {e}")
        
        return list(sessions)
    
    def add_evaluation_metrics(
        self, 
        log_file: str, 
        accuracy_score: float, 
        accuracy_explanation: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        将评估指标添加到日志文件中
        
        Args:
            log_file (str): 日志文件路径
            accuracy_score (float): 准确率评分
            accuracy_explanation (str): 准确率评分说明
            metrics (Optional[Dict[str, Any]]): 其他评估指标
            
        Returns:
            bool: 是否成功
        """
        try:
            log_path = Path(log_file)
            
            # 读取日志文件
            with open(log_path, 'r', encoding='utf-8') as f:
                log_entry = json.load(f)
            
            # 更新评估状态和结果
            log_entry["evaluated"] = True
            if "evaluation_result" not in log_entry:
                log_entry["evaluation_result"] = {}
            
            # 添加准确率评分
            log_entry["evaluation_result"]["accuracy"] = {
                "score": accuracy_score,
                "explanation": accuracy_explanation
            }
            
            # 添加其他评估指标
            if metrics:
                for key, value in metrics.items():
                    log_entry["evaluation_result"][key] = value
            
            # 添加评估时间戳
            log_entry["evaluation_timestamp"] = time.time()
            
            # 保存更新后的日志
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已添加评估指标到日志 {log_file}，准确率: {accuracy_score}")
            return True
            
        except Exception as e:
            logger.error(f"添加评估指标到日志 {log_file} 时出错: {e}")
            return False 