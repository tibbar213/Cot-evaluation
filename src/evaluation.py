"""
评估框架，用于评估模型回答的质量
"""

import json
import logging
import time
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from config import EVALUATION_METRICS, RESULT_PATH, EVAL_RESULT_FILE
from models import evaluate_response

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Evaluator:
    """评估器类，用于评估模型回答"""
    
    def __init__(self, result_path: str = RESULT_PATH):
        """
        初始化评估器
        
        Args:
            result_path (str): 结果保存路径
        """
        self.result_path = Path(result_path)
        self.result_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics = EVALUATION_METRICS
        self.results = {}
        logger.info(f"初始化评估器 - 结果路径: {result_path}, 评估指标: {self.metrics}")
    
    def evaluate_answer(
        self, 
        question: str, 
        reference_answer: str, 
        model_response: Dict[str, Any],
        strategy_name: str,
        question_id: str,
        question_category: str = "",
        question_difficulty: str = ""
    ) -> Dict[str, Any]:
        """
        评估模型回答
        
        Args:
            question (str): 问题
            reference_answer (str): 参考答案
            model_response (Dict[str, Any]): 模型回答，包含answer和reasoning等
            strategy_name (str): 策略名称
            question_id (str): 问题ID
            question_category (str): 问题类别
            question_difficulty (str): 问题难度
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        logger.info(f"评估问题 {question_id} 的回答 - 策略: {strategy_name}")
        logger.info(f"问题: {question}")
        logger.info(f"参考答案: {reference_answer}")
        logger.info(f"模型回答: {model_response.get('answer', '')}")
        logger.info(f"问题类别: {question_category}, 难度: {question_difficulty}")
        
        eval_result = {
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
            "metrics": {},
            "timestamp": time.time()
        }
        
        # 评估准确率
        if "accuracy" in self.metrics:
            logger.info("评估准确率...")
            accuracy_result = evaluate_response(
                question=question,
                reference_answer=reference_answer,
                model_response=model_response.get("answer", ""),
                metric="accuracy"
            )
            eval_result["metrics"]["accuracy"] = accuracy_result
            logger.info(f"准确率评分: {accuracy_result['score']}")
            logger.info(f"准确率评估说明: {accuracy_result['explanation']}")
        
        # 评估推理质量（如果有推理过程）
        if "reasoning_quality" in self.metrics and model_response.get("has_reasoning", False):
            logger.info("评估推理质量...")
            if model_response.get("reasoning"):
                logger.info(f"推理过程: {model_response.get('reasoning')[:200]}...")
                
                reasoning_quality_result = evaluate_response(
                    question=question,
                    reference_answer=reference_answer,
                    model_response=model_response.get("reasoning", ""),
                    metric="reasoning_quality"
                )
                eval_result["metrics"]["reasoning_quality"] = reasoning_quality_result
                logger.info(f"推理质量评分: {reasoning_quality_result['score']}/10")
                logger.info(f"推理质量评估说明: {reasoning_quality_result['explanation']}")
            else:
                logger.warning("has_reasoning为True但推理内容为空，跳过推理质量评估")
        
        # 添加到结果集合
        if strategy_name not in self.results:
            self.results[strategy_name] = []
        
        self.results[strategy_name].append(eval_result)
        logger.info(f"完成问题 {question_id} 的评估")
        
        return eval_result
    
    def calculate_overall_metrics(self) -> Dict[str, Any]:
        """
        计算总体评估指标
        
        Returns:
            Dict[str, Any]: 总体评估指标
        """
        logger.info("计算总体评估指标...")
        overall_metrics = {}
        
        for strategy, evals in self.results.items():
            logger.info(f"计算策略 '{strategy}' 的指标 - 共有 {len(evals)} 个评估结果")
            
            strategy_metrics = {
                "total_questions": len(evals),
                "metrics": {}
            }
            
            # 计算准确率
            if "accuracy" in self.metrics:
                accuracy_scores = [
                    float(e["metrics"]["accuracy"]["score"]) 
                    for e in evals 
                    if "accuracy" in e["metrics"]
                ]
                
                if accuracy_scores:
                    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
                    strategy_metrics["metrics"]["accuracy"] = {
                        "average_score": avg_accuracy,
                        "count": len(accuracy_scores)
                    }
                    logger.info(f"策略 '{strategy}' 的平均准确率: {avg_accuracy:.4f} (基于 {len(accuracy_scores)} 个问题)")
            
            # 计算推理质量
            if "reasoning_quality" in self.metrics:
                reasoning_scores = [
                    float(e["metrics"]["reasoning_quality"]["score"]) 
                    for e in evals 
                    if "reasoning_quality" in e["metrics"]
                ]
                
                if reasoning_scores:
                    avg_reasoning = sum(reasoning_scores) / len(reasoning_scores)
                    strategy_metrics["metrics"]["reasoning_quality"] = {
                        "average_score": avg_reasoning,
                        "count": len(reasoning_scores)
                    }
                    logger.info(f"策略 '{strategy}' 的平均推理质量: {avg_reasoning:.4f}/10 (基于 {len(reasoning_scores)} 个问题)")
            
            # 按难度分类计算准确率
            difficulty_metrics = {}
            for difficulty in ["easy", "medium", "hard"]:
                difficulty_evals = [e for e in evals if e["difficulty"] == difficulty]
                if difficulty_evals:
                    difficulty_accuracy = [
                        float(e["metrics"]["accuracy"]["score"]) 
                        for e in difficulty_evals 
                        if "accuracy" in e["metrics"]
                    ]
                    
                    if difficulty_accuracy:
                        avg_diff_accuracy = sum(difficulty_accuracy) / len(difficulty_accuracy)
                        difficulty_metrics[difficulty] = {
                            "count": len(difficulty_evals),
                            "accuracy": avg_diff_accuracy
                        }
                        logger.info(f"策略 '{strategy}' 在 {difficulty} 难度上的准确率: {avg_diff_accuracy:.4f} (基于 {len(difficulty_evals)} 个问题)")
            
            if difficulty_metrics:
                strategy_metrics["difficulty_breakdown"] = difficulty_metrics
            
            # 按类别分类计算准确率
            category_metrics = {}
            categories = set([e["category"] for e in evals if e["category"]])
            for category in categories:
                category_evals = [e for e in evals if e["category"] == category]
                if category_evals:
                    category_accuracy = [
                        float(e["metrics"]["accuracy"]["score"]) 
                        for e in category_evals 
                        if "accuracy" in e["metrics"]
                    ]
                    
                    if category_accuracy:
                        avg_cat_accuracy = sum(category_accuracy) / len(category_accuracy)
                        category_metrics[category] = {
                            "count": len(category_evals),
                            "accuracy": avg_cat_accuracy
                        }
                        logger.info(f"策略 '{strategy}' 在 {category} 类别上的准确率: {avg_cat_accuracy:.4f} (基于 {len(category_evals)} 个问题)")
            
            if category_metrics:
                strategy_metrics["category_breakdown"] = category_metrics
            
            overall_metrics[strategy] = strategy_metrics
        
        logger.info("总体评估指标计算完成")
        return overall_metrics
    
    def save_results(self, filename: str = EVAL_RESULT_FILE) -> str:
        """
        保存评估结果
        
        Args:
            filename (str): 文件名
            
        Returns:
            str: 保存的文件路径
        """
        logger.info(f"保存评估结果到 {filename}")
        result_file = self.result_path / filename
        
        # 计算总体指标
        overall_metrics = self.calculate_overall_metrics()
        
        # 构建结果数据
        result_data = {
            "overall_metrics": overall_metrics,
            "detailed_results": self.results,
            "timestamp": time.time()
        }
        
        # 保存结果
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存到 {result_file}")
        return str(result_file)
    
    def load_results(self, filename: str = EVAL_RESULT_FILE) -> Dict[str, Any]:
        """
        加载评估结果
        
        Args:
            filename (str): 文件名
            
        Returns:
            Dict[str, Any]: 加载的结果
        """
        logger.info(f"加载评估结果从 {filename}")
        result_file = self.result_path / filename
        
        if not result_file.exists():
            logger.warning(f"结果文件 {result_file} 不存在")
            return {}
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # 更新本地结果
            if "detailed_results" in result_data:
                self.results = result_data["detailed_results"]
                logger.info(f"已加载评估结果，共 {len(self.results)} 个策略")
                for strategy, evals in self.results.items():
                    logger.info(f"  策略 '{strategy}': {len(evals)} 个评估结果")
            
            return result_data
        
        except Exception as e:
            logger.error(f"加载评估结果时出错: {e}")
            return {}
    
    def print_summary(self) -> None:
        """打印评估结果摘要"""
        logger.info("生成评估结果摘要")
        overall_metrics = self.calculate_overall_metrics()
        
        print("\n===== 评估结果摘要 =====\n")
        
        for strategy, metrics in overall_metrics.items():
            print(f"## 策略: {strategy}")
            print(f"总问题数: {metrics['total_questions']}")
            
            if "accuracy" in metrics["metrics"]:
                accuracy = metrics["metrics"]["accuracy"]
                print(f"准确率: {accuracy['average_score']:.4f} (基于 {accuracy['count']} 个问题)")
            
            if "reasoning_quality" in metrics["metrics"]:
                reasoning = metrics["metrics"]["reasoning_quality"]
                print(f"推理质量: {reasoning['average_score']:.4f}/10 (基于 {reasoning['count']} 个问题)")
            
            print()
            
            if "difficulty_breakdown" in metrics:
                print("难度分析:")
                for difficulty, diff_metrics in metrics["difficulty_breakdown"].items():
                    print(f"  - {difficulty}: 准确率 {diff_metrics['accuracy']:.4f} (基于 {diff_metrics['count']} 个问题)")
                
                print()
            
            if "category_breakdown" in metrics:
                print("类别分析:")
                for category, cat_metrics in metrics["category_breakdown"].items():
                    print(f"  - {category}: 准确率 {cat_metrics['accuracy']:.4f} (基于 {cat_metrics['count']} 个问题)")
                
                print()
        
        print("=========================\n")
        logger.info("评估结果摘要已打印")
