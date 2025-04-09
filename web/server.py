from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
import logging
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 配置CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:3001"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# 创建一个模拟评估结果的函数
def create_mock_data(dataset=None, model=None):
    logger.info(f"创建模拟数据，数据集: {dataset}, 模型: {model}")
    
    # 使用默认值
    if not dataset:
        dataset = 'livebench/math'
    if not model:
        model = 'gpt-3.5'
        
    # 定义策略
    strategies = ["baseline", "zero_shot", "few_shot", "auto_cot", "auto_reason", "combined"]
    
    # 评估结果
    result_data = {}
    timestamp = 1649123456.789
    
    # 调整数据生成，根据数据集和模型有不同的结果
    dataset_factor = 1.0
    model_factor = 1.0
    
    if dataset:
        if 'math' in dataset:
            dataset_factor = 1.2
        elif 'reasoning' in dataset:
            dataset_factor = 0.9
        elif 'data_analysis' in dataset:
            dataset_factor = 1.1
    
    if model:
        if 'gpt-4' in model:
            model_factor = 1.3
        elif 'gpt-3.5' in model:
            model_factor = 0.8
        elif 'deepseek' in model:
            model_factor = 1.1
    
    # 为每个策略创建评估结果
    for strategy in strategies:
        result_data[strategy] = []
        strategy_index = strategies.index(strategy)
        strategy_factor = 0.5 + 0.5 * (strategy_index / len(strategies))
        
        # 为每个策略创建10个示例问题的评估结果
        for i in range(1, 11):
            accuracy_score = round(min(0.95, 0.3 + strategy_factor * dataset_factor * model_factor + 0.05 * (i % 3 - 1)), 2)
            
            # 定义难度
            difficulty = "easy" if i <= 3 else "medium" if i <= 7 else "hard"
            
            # 定义类别
            categories = ["arithmetic", "algebra", "geometry", "logic", "probability"]
            category = categories[i % len(categories)]
            
            result_data[strategy].append({
                "id": f"{category}_{i}",
                "question": f"这是{dataset}数据集中的第{i}个{category}问题，使用{model}模型解答，难度为{difficulty}",
                "reference_answer": f"参考答案{i}",
                "model_answer": f"模型{model}使用{strategy}策略的回答{i}",
                "reasoning": f"这是{model}模型的推理过程...\n步骤1: 分析问题\n步骤2: 应用公式\n步骤3: 计算结果\n最终结果是{i}",
                "category": category,
                "difficulty": difficulty,
                "metrics": {
                    "accuracy": {
                        "score": accuracy_score,
                        "explanation": f"准确率评估解释，得分{accuracy_score}"
                    },
                    "reasoning_quality": {
                        "score": 0,  # 不再使用推理质量
                        "explanation": ""
                    }
                },
                "timestamp": timestamp + i
            })
    
    # 添加整体指标
    result_data["timestamp"] = timestamp
    result_data["overall_metrics"] = {}
    
    for strategy in strategies:
        total_questions = len(result_data[strategy])
        accuracy_scores = [item["metrics"]["accuracy"]["score"] for item in result_data[strategy]]
        
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        
        # 按难度统计
        easy_items = [item for item in result_data[strategy] if item["difficulty"] == "easy"]
        medium_items = [item for item in result_data[strategy] if item["difficulty"] == "medium"]
        hard_items = [item for item in result_data[strategy] if item["difficulty"] == "hard"]
        
        easy_accuracy = sum([item["metrics"]["accuracy"]["score"] for item in easy_items]) / len(easy_items) if easy_items else 0
        medium_accuracy = sum([item["metrics"]["accuracy"]["score"] for item in medium_items]) / len(medium_items) if medium_items else 0
        hard_accuracy = sum([item["metrics"]["accuracy"]["score"] for item in hard_items]) / len(hard_items) if hard_items else 0
        
        # 按类别统计
        category_breakdown = {}
        categories = ["arithmetic", "algebra", "geometry", "logic", "probability"]
        for category in categories:
            cat_items = [item for item in result_data[strategy] if item["category"] == category]
            if cat_items:
                cat_accuracy = sum([item["metrics"]["accuracy"]["score"] for item in cat_items]) / len(cat_items)
                category_breakdown[category] = {"count": len(cat_items), "accuracy": cat_accuracy}
        
        result_data["overall_metrics"][strategy] = {
            "total_questions": total_questions,
            "metrics": {
                "accuracy": {
                    "average_score": avg_accuracy,
                    "count": total_questions
                },
                "reasoning_quality": {
                    "average_score": 0,  # 不再使用推理质量
                    "count": total_questions
                }
            },
            "difficulty_breakdown": {
                "easy": {"count": len(easy_items), "accuracy": easy_accuracy},
                "medium": {"count": len(medium_items), "accuracy": medium_accuracy},
                "hard": {"count": len(hard_items), "accuracy": hard_accuracy}
            },
            "category_breakdown": category_breakdown
        }
    
    return result_data

@app.route('/')
def index():
    return jsonify({"status": "ok", "message": "API服务正常运行"})

@app.route('/api/evaluation-results')
def get_evaluation_results():
    try:
        logger.info("收到获取评估结果的请求")
        
        # 获取查询参数
        dataset = request.args.get('dataset', 'livebench/math')
        model = request.args.get('model', 'gpt-3.5')
        
        logger.info(f"查询参数 - 数据集: {dataset}, 模型: {model}")
        
        # 始终返回模拟数据，确保前端能正常工作
        mock_data = create_mock_data(dataset, model)
        logger.info(f"成功生成模拟数据，策略数量: {len(mock_data.keys()) - 2}")  # 减去timestamp和overall_metrics
        
        # 返回结果前先验证数据结构
        assert "timestamp" in mock_data, "Missing timestamp in result data"
        assert "overall_metrics" in mock_data, "Missing overall_metrics in result data"
        assert len(mock_data.keys()) > 2, "No strategy data in result"
        
        return jsonify(mock_data)
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "timestamp": 1649123456.789,
            "baseline": [],  # 至少提供一个策略，防止前端出错
            "overall_metrics": {
                "baseline": {
                    "total_questions": 0,
                    "metrics": {
                        "accuracy": {
                            "average_score": 0,
                            "count": 0
                        },
                        "reasoning_quality": {
                            "average_score": 0,
                            "count": 0
                        }
                    },
                    "difficulty_breakdown": {
                        "easy": {"count": 0, "accuracy": 0},
                        "medium": {"count": 0, "accuracy": 0},
                        "hard": {"count": 0, "accuracy": 0}
                    },
                    "category_breakdown": {}
                }
            }
        }), 500

if __name__ == '__main__':
    logger.info("启动Flask服务器...")
    app.run(port=5000, debug=True) 