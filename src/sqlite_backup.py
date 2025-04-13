import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SQLiteBackup:
    """SQLite备份类，用于将评估结果和对话日志保存到SQLite数据库"""

    def __init__(self, db_path: str = "data/backup.db"):
        """
        初始化SQLite备份类
        
        参数:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._ensure_dir_exists()
        self.conn = None
        self.init_db()

    def _ensure_dir_exists(self):
        """确保数据库目录存在"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def init_db(self):
        """初始化数据库连接和表"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            
            # 创建评估结果表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT NOT NULL,
                strategy TEXT NOT NULL,
                dataset TEXT,
                model TEXT,
                question TEXT NOT NULL,
                reference_answer TEXT,
                model_answer TEXT,
                reasoning TEXT,
                category TEXT,
                difficulty TEXT,
                accuracy_score REAL,
                accuracy_explanation TEXT,
                reasoning_score REAL,
                reasoning_explanation TEXT,
                timestamp REAL NOT NULL,
                session_id TEXT,
                UNIQUE(question_id, strategy, session_id)
            )
            ''')
            
            # 创建会话元数据表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                result_prefix TEXT,
                dataset TEXT,
                model TEXT,
                start_time REAL,
                end_time REAL,
                total_questions INTEGER,
                metadata TEXT
            )
            ''')
            
            # 创建策略元数据表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                strategy TEXT NOT NULL,
                name TEXT,
                description TEXT,
                parameters TEXT,
                UNIQUE(session_id, strategy)
            )
            ''')
            
            # 创建总体评估指标表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS overall_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                strategy TEXT NOT NULL,
                total_questions INTEGER,
                avg_accuracy REAL,
                avg_reasoning_quality REAL,
                metrics_json TEXT,
                timestamp REAL,
                UNIQUE(session_id, strategy)
            )
            ''')
            
            self.conn.commit()
            logger.info(f"已初始化SQLite数据库: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"初始化数据库失败: {e}")
            if self.conn:
                self.conn.close()
                self.conn = None
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def backup_evaluation_result(self, result: Dict[str, Any], strategy: str, 
                                 session_id: str, dataset: str = None, model: str = None):
        """
        备份单个评估结果
        
        参数:
            result: 评估结果字典
            strategy: 策略名称
            session_id: 会话ID
            dataset: 数据集名称
            model: 模型名称
        """
        if not self.conn:
            self.init_db()
        
        try:
            cursor = self.conn.cursor()
            
            # 准备插入数据
            metrics = result.get('metrics', {})
            accuracy = metrics.get('accuracy', {})
            reasoning = metrics.get('reasoning_quality', {})
            
            cursor.execute('''
            INSERT OR REPLACE INTO evaluation_results 
            (question_id, strategy, dataset, model, question, reference_answer, 
            model_answer, reasoning, category, difficulty, 
            accuracy_score, accuracy_explanation, reasoning_score, reasoning_explanation,
            timestamp, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.get('id', ''),
                strategy,
                dataset,
                model,
                result.get('question', ''),
                result.get('reference_answer', ''),
                result.get('model_answer', ''),
                result.get('reasoning', ''),
                result.get('category', ''),
                result.get('difficulty', ''),
                accuracy.get('score', 0),
                accuracy.get('explanation', ''),
                reasoning.get('score', 0),
                reasoning.get('explanation', ''),
                result.get('timestamp', datetime.now().timestamp()),
                session_id
            ))
            
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"备份评估结果失败: {e}")
            self.conn.rollback()
            raise
    
    def backup_conversation_log(self, log: Dict[str, Any]):
        """
        备份对话日志
        
        参数:
            log: 对话日志字典
        """
        if not self.conn:
            self.init_db()
        
        try:
            cursor = self.conn.cursor()
            
            # 准备插入数据
            session_id = log.get('session_id', '')
            strategy = log.get('strategy', '')
            metrics = log.get('evaluation_result', {})
            accuracy = metrics.get('accuracy', {})
            reasoning = metrics.get('reasoning_quality', {})
            
            cursor.execute('''
            INSERT OR REPLACE INTO evaluation_results 
            (question_id, strategy, question, reference_answer, 
            model_answer, reasoning, category, difficulty, 
            accuracy_score, accuracy_explanation, reasoning_score, reasoning_explanation,
            timestamp, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log.get('question_id', ''),
                strategy,
                log.get('question', ''),
                log.get('reference_answer', ''),
                log.get('model_answer', ''),
                log.get('reasoning', ''),
                log.get('category', ''),
                log.get('difficulty', ''),
                accuracy.get('score', 0),
                accuracy.get('explanation', ''),
                reasoning.get('score', 0),
                reasoning.get('explanation', ''),
                log.get('timestamp', datetime.now().timestamp()),
                session_id
            ))
            
            # 保存策略元数据
            metadata = log.get('metadata', {})
            strategy_details = metadata.get('strategy_details', {})
            if strategy_details:
                cursor.execute('''
                INSERT OR REPLACE INTO strategy_metadata
                (session_id, strategy, name, description, parameters)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    strategy,
                    strategy_details.get('name', ''),
                    strategy_details.get('description', ''),
                    json.dumps(strategy_details)
                ))
            
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"备份对话日志失败: {e}")
            self.conn.rollback()
            raise
    
    def backup_overall_metrics(self, metrics: Dict[str, Any], strategy: str, session_id: str):
        """
        备份策略的总体评估指标
        
        参数:
            metrics: 总体评估指标
            strategy: 策略名称
            session_id: 会话ID
        """
        if not self.conn:
            self.init_db()
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO overall_metrics
            (session_id, strategy, total_questions, avg_accuracy, avg_reasoning_quality, 
            metrics_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                strategy,
                metrics.get('total_questions', 0),
                metrics.get('metrics', {}).get('accuracy', {}).get('average_score', 0),
                metrics.get('metrics', {}).get('reasoning_quality', {}).get('average_score', 0),
                json.dumps(metrics),
                datetime.now().timestamp()
            ))
            
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"备份总体评估指标失败: {e}")
            self.conn.rollback()
            raise
    
    def backup_session(self, session_id: str, result_prefix: str = None,
                     dataset: str = None, model: str = None, 
                     start_time: float = None, end_time: float = None,
                     total_questions: int = 0, metadata: Dict[str, Any] = None):
        """
        备份会话元数据
        
        参数:
            session_id: 会话ID
            result_prefix: 结果前缀
            dataset: 数据集名称
            model: 模型名称
            start_time: 开始时间戳
            end_time: 结束时间戳
            total_questions: 总问题数
            metadata: 其他元数据
        """
        if not self.conn:
            self.init_db()
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO sessions
            (session_id, result_prefix, dataset, model, 
            start_time, end_time, total_questions, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                result_prefix,
                dataset,
                model,
                start_time or datetime.now().timestamp(),
                end_time,
                total_questions,
                json.dumps(metadata or {})
            ))
            
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"备份会话元数据失败: {e}")
            self.conn.rollback()
            raise
    
    def backup_all_results(self, results: Dict[str, List[Dict[str, Any]]], session_id: str,
                          dataset: str = None, model: str = None):
        """
        备份所有评估结果
        
        参数:
            results: 评估结果字典，键为策略名称，值为评估结果列表
            session_id: 会话ID
            dataset: 数据集名称
            model: 模型名称
        """
        start_time = datetime.now().timestamp()
        total_questions = 0
        
        for strategy, result_list in results.items():
            if strategy in ['timestamp', 'overall_metrics']:
                continue
                
            for result in result_list:
                self.backup_evaluation_result(result, strategy, session_id, dataset, model)
                total_questions += 1
        
        # 备份总体指标
        if 'overall_metrics' in results:
            for strategy, metrics in results['overall_metrics'].items():
                self.backup_overall_metrics(metrics, strategy, session_id)
        
        # 备份会话信息
        self.backup_session(
            session_id=session_id,
            dataset=dataset,
            model=model,
            start_time=start_time,
            end_time=datetime.now().timestamp(),
            total_questions=total_questions
        )
    
    def get_sessions(self) -> List[Dict[str, Any]]:
        """获取所有会话"""
        if not self.conn:
            self.init_db()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            SELECT session_id, result_prefix, dataset, model, 
                   start_time, end_time, total_questions, metadata
            FROM sessions
            ORDER BY start_time DESC
            ''')
            
            sessions = []
            for row in cursor.fetchall():
                session_id, result_prefix, dataset, model, start_time, end_time, total_questions, metadata = row
                sessions.append({
                    'session_id': session_id,
                    'result_prefix': result_prefix,
                    'dataset': dataset,
                    'model': model,
                    'start_time': start_time,
                    'end_time': end_time,
                    'total_questions': total_questions,
                    'metadata': json.loads(metadata) if metadata else {}
                })
            
            return sessions
        except sqlite3.Error as e:
            logger.error(f"获取会话列表失败: {e}")
            return []
    
    def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """
        获取指定会话的评估结果
        
        参数:
            session_id: 会话ID
            
        返回:
            评估结果字典，格式与API返回格式相同
        """
        if not self.conn:
            self.init_db()
        
        try:
            cursor = self.conn.cursor()
            
            # 获取会话信息
            cursor.execute('''
            SELECT result_prefix, dataset, model, start_time, end_time, total_questions
            FROM sessions
            WHERE session_id = ?
            ''', (session_id,))
            
            session_row = cursor.fetchone()
            if not session_row:
                return {}
                
            result_prefix, dataset, model, start_time, end_time, total_questions = session_row
            
            # 获取策略列表
            cursor.execute('''
            SELECT DISTINCT strategy
            FROM evaluation_results
            WHERE session_id = ?
            ''', (session_id,))
            
            strategies = [row[0] for row in cursor.fetchall()]
            
            # 构建结果字典
            results = {}
            results['timestamp'] = end_time or start_time
            
            # 获取每个策略的评估结果
            for strategy in strategies:
                cursor.execute('''
                SELECT question_id, question, reference_answer, model_answer, reasoning,
                       category, difficulty, accuracy_score, accuracy_explanation,
                       reasoning_score, reasoning_explanation, timestamp
                FROM evaluation_results
                WHERE session_id = ? AND strategy = ?
                ''', (session_id, strategy))
                
                strategy_results = []
                for row in cursor.fetchall():
                    (question_id, question, reference_answer, model_answer, reasoning,
                     category, difficulty, accuracy_score, accuracy_explanation,
                     reasoning_score, reasoning_explanation, timestamp) = row
                    
                    strategy_results.append({
                        'id': question_id,
                        'question': question,
                        'reference_answer': reference_answer,
                        'model_answer': model_answer,
                        'reasoning': reasoning,
                        'category': category,
                        'difficulty': difficulty,
                        'metrics': {
                            'accuracy': {
                                'score': accuracy_score,
                                'explanation': accuracy_explanation
                            },
                            'reasoning_quality': {
                                'score': reasoning_score,
                                'explanation': reasoning_explanation
                            }
                        },
                        'timestamp': timestamp
                    })
                
                results[strategy] = strategy_results
            
            # 获取总体指标
            cursor.execute('''
            SELECT strategy, metrics_json
            FROM overall_metrics
            WHERE session_id = ?
            ''', (session_id,))
            
            overall_metrics = {}
            for row in cursor.fetchall():
                strategy, metrics_json = row
                overall_metrics[strategy] = json.loads(metrics_json)
            
            results['overall_metrics'] = overall_metrics
            
            return results
        except sqlite3.Error as e:
            logger.error(f"获取会话评估结果失败: {e}")
            return {}
    
    def export_to_json(self, session_id: str, output_path: str = None) -> str:
        """
        将会话评估结果导出为JSON文件
        
        参数:
            session_id: 会话ID
            output_path: 输出路径，默认为results/backup_{session_id}.json
            
        返回:
            JSON文件路径
        """
        results = self.get_session_results(session_id)
        if not results:
            logger.warning(f"未找到会话 {session_id} 的评估结果")
            return None
        
        if not output_path:
            output_path = f"results/backup_{session_id}.json"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已将会话 {session_id} 的评估结果导出到 {output_path}")
        return output_path 