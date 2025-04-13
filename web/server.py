from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
import logging
import traceback
import sys
import argparse
import time
import re

# 添加项目根目录到PATH，以便导入sqlite_backup模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.sqlite_backup import SQLiteBackup
except ImportError:
    # 如果导入失败，定义一个空的SQLiteBackup类
    class SQLiteBackup:
        def __init__(self, *args, **kwargs):
            pass
        def get_session_results(self, *args, **kwargs):
            return None
        def get_sessions(self, *args, **kwargs):
            return []

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

# SQLite备份实例
sqlite_backup = None

# 创建一个模拟评估结果的函数
def generate_mock_data(strategies=None, dataset=None, model=None):
    """
    生成模拟评估结果数据
    
    参数:
        strategies: 策略列表，如果为None则使用默认策略
        dataset: 数据集名称
        model: 模型名称
        
    返回:
        模拟评估结果字典
    """
    # 如果没有提供策略，使用默认策略
    if not strategies:
        strategies = ["baseline", "zero_shot", "few_shot", "auto_cot", "auto_reason", "combined"]
    
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
            "total_records": total_questions,
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

def get_sqlite_data(dataset=None, model=None, session_id=None):
    """
    从SQLite数据库获取数据
    
    参数:
        dataset: 数据集名称
        model: 模型名称
        session_id: 会话ID
    
    返回:
        评估结果字典
    """
    global sqlite_backup
    if not sqlite_backup:
        from src.sqlite_backup import SQLiteBackup
        sqlite_backup = SQLiteBackup()
    
    try:
        # 获取所有会话
        sessions = sqlite_backup.get_sessions()
        if not sessions:
            logger.warning("SQLite数据库中没有会话记录")
            return None
        
        # 如果指定了会话ID，直接获取
        if session_id:
            return sqlite_backup.get_session_results(session_id)
        
        # 根据数据集和模型过滤会话
        filtered_sessions = sessions
        if dataset:
            filtered_sessions = [s for s in filtered_sessions if s.get('dataset') == dataset]
        
        if model:
            filtered_sessions = [s for s in filtered_sessions if s.get('model') == model]
        
        # 如果没有找到匹配的会话，返回第一个会话的结果
        if not filtered_sessions and sessions:
            logger.info(f"没有找到匹配的会话，使用第一个会话: {sessions[0]['session_id']}")
            return sqlite_backup.get_session_results(sessions[0]['session_id'])
        
        # 按时间戳倒序排序，取最新的会话
        if filtered_sessions:
            filtered_sessions.sort(key=lambda x: x.get('start_time', 0), reverse=True)
            session_id = filtered_sessions[0]['session_id']
            logger.info(f"使用会话: {session_id}")
            return sqlite_backup.get_session_results(session_id)
        
        return None
    except Exception as e:
        logger.error(f"从SQLite获取数据失败: {e}")
        return None

def get_json_data(json_path="../results/eval_results.json"):
    """
    从JSON文件中加载评估结果
    
    参数:
        json_path: JSON文件路径
        
    返回:
        评估结果字典
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        # 如果没有timestamp字段，添加一个
        if "timestamp" not in result_data:
            result_data["timestamp"] = time.time()
        
        # 计算总体评估指标
        if "overall_metrics" not in result_data:
            result_data["overall_metrics"] = {}
            for strategy, evals in result_data.items():
                if strategy == "timestamp":
                    continue
                
                total_questions = len(evals)
                accuracy_scores = [item["metrics"]["accuracy"]["score"] for item in evals if "metrics" in item and "accuracy" in item["metrics"]]
                
                avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
                
                # 按难度统计
                easy_items = [item for item in evals if item.get("difficulty") == "easy"]
                medium_items = [item for item in evals if item.get("difficulty") == "medium"]
                hard_items = [item for item in evals if item.get("difficulty") == "hard"]
                
                easy_accuracy = sum([item["metrics"]["accuracy"]["score"] for item in easy_items if "metrics" in item and "accuracy" in item["metrics"]]) / len(easy_items) if easy_items else 0
                medium_accuracy = sum([item["metrics"]["accuracy"]["score"] for item in medium_items if "metrics" in item and "accuracy" in item["metrics"]]) / len(medium_items) if medium_items else 0
                hard_accuracy = sum([item["metrics"]["accuracy"]["score"] for item in hard_items if "metrics" in item and "accuracy" in item["metrics"]]) / len(hard_items) if hard_items else 0
                
                # 按类别统计
                category_breakdown = {}
                categories = set(item.get("category", "") for item in evals if item.get("category"))
                for category in categories:
                    cat_items = [item for item in evals if item.get("category") == category]
                    if cat_items:
                        cat_accuracy = sum([item["metrics"]["accuracy"]["score"] for item in cat_items if "metrics" in item and "accuracy" in item["metrics"]]) / len(cat_items)
                        category_breakdown[category] = {"count": len(cat_items), "accuracy": cat_accuracy}
                
                result_data["overall_metrics"][strategy] = {
                    "total_records": total_questions,
                    "metrics": {
                        "accuracy": {
                            "average_score": avg_accuracy,
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
        
        logger.info(f"从JSON文件加载了评估结果: {len(result_data)} 个策略的数据")
        return result_data
    except Exception as e:
        logger.error(f"从JSON文件加载评估结果失败: {e}")
        logger.exception("详细错误：")
        return None

def load_from_conversation_logs(logs_path, dataset_filter=None, model_filter=None):
    """从对话日志目录加载数据"""
    results = {}
    
    # 检查目录是否存在
    if not os.path.exists(logs_path):
        logger.error(f"对话日志目录 {logs_path} 不存在")
        return results
    
    # 获取所有策略目录
    strategy_dirs = [d for d in os.listdir(logs_path) if os.path.isdir(os.path.join(logs_path, d))]
    logger.info(f"找到 {len(strategy_dirs)} 个策略目录: {', '.join(strategy_dirs)}")
    
    for strategy in strategy_dirs:
        strategy_path = os.path.join(logs_path, strategy)
        log_files = [f for f in os.listdir(strategy_path) if f.endswith('.json')]
        
        logger.info(f"策略 {strategy} 中找到 {len(log_files)} 个日志文件")
        results[strategy] = []
        
        for log_file in log_files:
            # 从文件名提取数据集
            dataset_match = re.match(r'([^_]+)_', log_file)
            if dataset_match:
                dataset = dataset_match.group(1)
                
                # 如果有数据集过滤，并且当前数据集不匹配，则跳过
                if dataset_filter and dataset != dataset_filter:
                    continue
            
            try:
                with open(os.path.join(strategy_path, log_file), 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    
                    # 如果有模型过滤器，并且当前模型不匹配，则跳过
                    if model_filter and log_data.get('model_name') != model_filter:
                        continue
                    
                    # 创建一个新的评估记录，只包含需要的字段
                    eval_item = {
                        "id": log_data.get("question_id", "unknown"),
                        "question": log_data.get("question", ""),
                        "category": log_data.get("category", ""),
                        "difficulty": log_data.get("difficulty", ""),
                        "strategy": log_data.get("strategy", strategy),
                        "model_name": log_data.get("model_name", "Unknown"),
                        "reference_answer": log_data.get("reference_answer", ""),
                        "model_answer": log_data.get("model_answer", ""),
                        "full_response": log_data.get("full_response", ""),
                        "reasoning": log_data.get("reasoning", None),
                        "has_reasoning": log_data.get("has_reasoning", False),
                        "timestamp": log_data.get("timestamp", time.time())
                    }
                    
                    # 添加评估指标
                    if "evaluation_result" in log_data and log_data["evaluation_result"]:
                        eval_item["metrics"] = log_data["evaluation_result"]
                    else:
                        # 未评估或无结果时提供默认值
                        eval_item["metrics"] = {
                            "accuracy": {
                                "score": 0,
                                "explanation": "未评估或无评估结果"
                            }
                        }
                    
                    results[strategy].append(eval_item)
            except Exception as e:
                logger.error(f"读取日志文件 {log_file} 失败: {e}")
    
    # 计算总体指标
    overall_metrics = calculate_overall_metrics(results)
    results['overall_metrics'] = overall_metrics
    
    # 添加时间戳
    results['timestamp'] = time.time()
    
    logger.info(f"从对话日志加载了 {len(strategy_dirs)} 个策略的数据")
    return results

# 添加计算总体指标的函数
def calculate_overall_metrics(results):
    """计算总体评估指标"""
    overall_metrics = {}
    for strategy, evals in results.items():
        if strategy == "timestamp" or not evals:
            continue
            
        total_questions = len(evals)
        accuracy_scores = [
            item["metrics"]["accuracy"]["score"] 
            for item in evals 
            if "metrics" in item and "accuracy" in item["metrics"]
        ]
        
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        
        # 按难度统计
        difficulty_breakdown = {}
        for difficulty in ["easy", "medium", "hard"]:
            diff_items = [item for item in evals if item.get("difficulty") == difficulty]
            if diff_items:
                diff_accuracy = sum([
                    item["metrics"]["accuracy"]["score"] 
                    for item in diff_items 
                    if "metrics" in item and "accuracy" in item["metrics"]
                ]) / len(diff_items) if diff_items else 0
                
                difficulty_breakdown[difficulty] = {
                    "count": len(diff_items),
                    "accuracy": diff_accuracy
                }
        
        # 按类别统计
        category_breakdown = {}
        categories = set(item.get("category", "") for item in evals if item.get("category"))
        for category in categories:
            cat_items = [item for item in evals if item.get("category") == category]
            if cat_items:
                cat_accuracy = sum([
                    item["metrics"]["accuracy"]["score"] 
                    for item in cat_items 
                    if "metrics" in item and "accuracy" in item["metrics"]
                ]) / len(cat_items) if cat_items else 0
                
                category_breakdown[category] = {
                    "count": len(cat_items),
                    "accuracy": cat_accuracy
                }
        
        # 构建策略整体指标
        overall_metrics[strategy] = {
            "total_records": total_questions,
            "metrics": {
                "accuracy": {
                    "average_score": avg_accuracy,
                    "count": total_questions
                }
            },
            "difficulty_breakdown": difficulty_breakdown,
            "category_breakdown": category_breakdown
        }
        
    return overall_metrics

@app.route('/')
def index():
    return jsonify({"status": "ok", "message": "API服务正常运行"})

@app.route('/api/sessions')
def get_sessions():
    """获取所有会话列表"""
    try:
        if sqlite_backup:
            sessions = sqlite_backup.get_sessions()
            return jsonify(sessions)
        else:
            return jsonify([])
    except Exception as e:
        logger.error(f"获取会话列表失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluation-results')
def get_evaluation_results():
    try:
        logger.info("收到获取评估结果的请求")
        
        # 获取查询参数
        dataset = request.args.get('dataset', 'livebench/math')
        model = request.args.get('model', 'gpt-3.5')
        session_id = request.args.get('session_id')
        json_path = request.args.get('json_path', '../results/eval_results.json')
        logs_path = request.args.get('logs_path', '../results/conversation_logs')
        use_json = request.args.get('use_json', 'false').lower() in ('true', '1', 'yes')
        use_logs = request.args.get('use_logs', 'true').lower() in ('true', '1', 'yes')
        
        # 优先从对话日志加载数据
        if use_logs:
            logger.info(f"尝试从对话日志目录 {logs_path} 加载数据")
            result_data = load_from_conversation_logs(logs_path)
            if result_data:
                logger.info("从对话日志目录加载数据成功")
                return jsonify(result_data)
            else:
                logger.info("从对话日志目录加载数据失败，尝试其他数据源")
        
        # 尝试从JSON文件加载数据
        if use_json:
            logger.info(f"尝试从JSON文件 {json_path} 加载数据")
            result_data = get_json_data(json_path)
            if result_data:
                logger.info("从JSON文件加载数据成功")
                return jsonify(result_data)
            else:
                logger.info("从JSON文件加载数据失败，尝试从SQLite数据库获取数据")
        
        # 从SQLite数据库获取数据
        if sqlite_backup:
            logger.info("尝试从SQLite数据库获取数据")
            result_data = get_sqlite_data(dataset, model, session_id)
            if result_data:
                logger.info("从SQLite数据库获取数据成功")
                return jsonify(result_data)
            else:
                logger.info("从SQLite数据库获取数据失败，使用模拟数据")
        
        # 生成模拟数据
        mock_data = generate_mock_data(
            strategies=["baseline", "zero_shot", "few_shot", "auto_cot", "auto_reason", "combined"],
            dataset=dataset,
            model=model
        )
        
        # 返回模拟数据
        logger.info("使用模拟数据")
        return jsonify(mock_data)
    except Exception as e:
        logger.error(f"获取评估结果时出错: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/dataset-model-strategy-options')
def get_options():
    """获取可用的数据集、模型和策略选项"""
    try:
        logger.info("收到获取选项的请求")
        
        # 获取查询参数
        logs_path = request.args.get('logs_path', '../results/conversation_logs')
        
        available_options = {
            "strategies": [],
            "datasets": [],
            "models": []
        }
        
        # 检查目录是否存在
        if not os.path.exists(logs_path):
            logger.error(f"对话日志目录 {logs_path} 不存在")
            return jsonify(available_options)
        
        # 获取所有策略目录
        strategy_dirs = [d for d in os.listdir(logs_path) if os.path.isdir(os.path.join(logs_path, d))]
        available_options["strategies"] = strategy_dirs
        
        # 获取所有数据集和模型
        datasets = set()
        models = set()
        
        for strategy in strategy_dirs:
            strategy_path = os.path.join(logs_path, strategy)
            log_files = [f for f in os.listdir(strategy_path) if f.endswith('.json')]
            
            for log_file in log_files:
                # 从文件名提取数据集
                dataset_match = re.match(r'([^_]+)_', log_file)
                if dataset_match:
                    datasets.add(dataset_match.group(1))
                
                # 读取文件获取模型信息
                try:
                    with open(os.path.join(strategy_path, log_file), 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                        if 'model_name' in log_data:
                            models.add(log_data['model_name'])
                except Exception as e:
                    logger.error(f"读取日志文件 {log_file} 失败: {e}")
        
        available_options["datasets"] = list(datasets)
        available_options["models"] = list(models)
        
        logger.info(f"可用选项: {available_options}")
        return jsonify(available_options)
    except Exception as e:
        logger.error(f"获取选项时出错: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def parse_args():
    parser = argparse.ArgumentParser(description="CoT评估Web API服务器")
    parser.add_argument("--port", type=int, default=5000, help="服务器端口")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--use-sqlite", action="store_true", help="使用SQLite数据库")
    parser.add_argument("--db-path", type=str, default="data/backup.db", help="SQLite数据库路径")
    parser.add_argument("--use-json", action="store_true", help="使用JSON文件")
    parser.add_argument("--json-path", type=str, default="../results/eval_results.json", help="JSON文件路径")
    parser.add_argument("--use-logs", action="store_true", default=True, help="使用对话日志目录")
    parser.add_argument("--logs-path", type=str, default="../results/conversation_logs", help="对话日志目录路径")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 初始化SQLite备份
    if args.use_sqlite:
        try:
            from src.sqlite_backup import SQLiteBackup
            sqlite_backup = SQLiteBackup(db_path=args.db_path)
            logger.info(f"已初始化SQLite备份，数据库路径: {args.db_path}")
        except Exception as e:
            logger.error(f"初始化SQLite备份失败: {e}")
            logger.exception("详细错误：")
    
    # 测试JSON文件是否存在并可读
    if args.use_json:
        try:
            result_data = get_json_data(args.json_path)
            if result_data:
                logger.info(f"已成功读取JSON文件 {args.json_path}")
            else:
                logger.warning(f"读取JSON文件 {args.json_path} 失败")
        except Exception as e:
            logger.error(f"测试读取JSON文件失败: {e}")
            logger.exception("详细错误：")
    
    # 测试对话日志是否存在并可读
    if args.use_logs:
        try:
            result_data = load_from_conversation_logs(args.logs_path)
            if result_data:
                logger.info(f"已成功读取对话日志 {args.logs_path}")
            else:
                logger.warning(f"读取对话日志 {args.logs_path} 失败")
        except Exception as e:
            logger.error(f"测试读取对话日志失败: {e}")
            logger.exception("详细错误：")
    
    # 启动Flask服务器
    logger.info("启动Flask服务器...")
    app.run(host=args.host, port=args.port) 