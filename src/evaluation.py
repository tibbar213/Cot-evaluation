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
    
    def __init__(self, result_path: str = RESULT_PATH, result_prefix: Optional[str] = None):
        """
        初始化评估器
        
        Args:
            result_path (str): 结果保存路径
            result_prefix (Optional[str]): 结果文件前缀，用于区分不同评估任务
        """
        self.result_path = Path(result_path)
        self.result_path.mkdir(parents=True, exist_ok=True)
        
        # 使用前缀构建结果文件路径
        if result_prefix:
            self.result_file = self.result_path / f"{result_prefix}_eval_results.json"
        else:
            self.result_file = self.result_path / EVAL_RESULT_FILE
        
        # 初始化评估结果
        self.results = {}
        
        # 初始化评估指标
        self.metrics = EVALUATION_METRICS
        
        logger.info(f"初始化评估器 - 结果路径: {self.result_path}, 评估指标: {self.metrics}")
    
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
            # 确保evals是列表
            if not isinstance(evals, list):
                logger.warning(f"策略 '{strategy}' 的评估结果不是列表，跳过")
                continue
                
            logger.info(f"计算策略 '{strategy}' 的指标 - 共有 {len(evals)} 个评估结果")
            
            strategy_metrics = {
                "total_questions": len(evals),
                "metrics": {}
            }
            
            # 计算准确率
            if "accuracy" in self.metrics:
                accuracy_scores = []
                for e in evals:
                    if isinstance(e, dict) and "metrics" in e and "accuracy" in e["metrics"]:
                        try:
                            score = float(e["metrics"]["accuracy"]["score"])
                            accuracy_scores.append(score)
                        except (ValueError, TypeError, KeyError) as err:
                            logger.warning(f"处理准确率分数时出错: {err}")
                
                if accuracy_scores:
                    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
                    strategy_metrics["metrics"]["accuracy"] = {
                        "average_score": avg_accuracy,
                        "count": len(accuracy_scores)
                    }
                    logger.info(f"策略 '{strategy}' 的平均准确率: {avg_accuracy:.4f} (基于 {len(accuracy_scores)} 个问题)")
            
            # 计算推理质量
            if "reasoning_quality" in self.metrics:
                reasoning_scores = []
                for e in evals:
                    if isinstance(e, dict) and "metrics" in e and "reasoning_quality" in e["metrics"]:
                        try:
                            score = float(e["metrics"]["reasoning_quality"]["score"])
                            reasoning_scores.append(score)
                        except (ValueError, TypeError, KeyError) as err:
                            logger.warning(f"处理推理质量分数时出错: {err}")
                
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
                difficulty_evals = [e for e in evals if isinstance(e, dict) and e.get("difficulty") == difficulty]
                if difficulty_evals:
                    difficulty_accuracy = []
                    for e in difficulty_evals:
                        if isinstance(e, dict) and "metrics" in e and "accuracy" in e["metrics"]:
                            try:
                                score = float(e["metrics"]["accuracy"]["score"])
                                difficulty_accuracy.append(score)
                            except (ValueError, TypeError, KeyError) as err:
                                logger.warning(f"处理难度 {difficulty} 的准确率分数时出错: {err}")
                    
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
            categories = set(e.get("category", "") for e in evals if isinstance(e, dict) and e.get("category"))
            for category in categories:
                category_evals = [e for e in evals if isinstance(e, dict) and e.get("category") == category]
                if category_evals:
                    category_accuracy = []
                    for e in category_evals:
                        if isinstance(e, dict) and "metrics" in e and "accuracy" in e["metrics"]:
                            try:
                                score = float(e["metrics"]["accuracy"]["score"])
                                category_accuracy.append(score)
                            except (ValueError, TypeError, KeyError) as err:
                                logger.warning(f"处理类别 {category} 的准确率分数时出错: {err}")
                    
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
    
    def save_results(self):
        """保存评估结果"""
        try:
            # 如果已有结果，先加载原有结果
            if self.result_file.exists():
                try:
                    with open(self.result_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                    
                    # 合并结果
                    for strategy, strategy_results in self.results.items():
                        if strategy in existing_results:
                            # 原有策略，添加新问题
                            existing_results[strategy].update(strategy_results)
                        else:
                            # 新策略，直接添加
                            existing_results[strategy] = strategy_results
                    
                    # 更新结果
                    self.results = existing_results
                except Exception as e:
                    logger.warning(f"加载原有结果时出错，将覆盖原有结果: {e}")
            
            # 保存结果
            with open(self.result_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"评估结果已保存到 {self.result_file}")
        except Exception as e:
            logger.error(f"保存评估结果时出错: {e}")
    
    def load_results(self):
        """加载评估结果"""
        try:
            if self.result_file.exists():
                with open(self.result_file, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                logger.info(f"已加载评估结果，包含 {len(self.results)} 个策略")
            else:
                logger.warning(f"评估结果文件 {self.result_file} 不存在")
        except Exception as e:
            logger.error(f"加载评估结果时出错: {e}")
    
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
