"""
配置文件，用于管理项目的配置参数
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
load_dotenv()

# 基本路径
BASE_DIR = Path(__file__).resolve().parent.parent

# OpenAI API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# 模型配置
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
EVALUATION_MODEL = os.getenv("EVALUATION_MODEL", "gpt-4")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
REASONING_MODEL = os.getenv("REASONING_MODEL", "deepseek-ai/DeepSeek-V3")

# 向量数据库配置
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(BASE_DIR / "data" / "vector_store"))

# 数据配置
QUESTIONS_PATH = str(BASE_DIR / "data" / "questions.json")

# 结果配置
RESULT_PATH = os.getenv("RESULT_PATH", str(BASE_DIR / "results"))
EVAL_RESULT_FILE = "eval_results.json"

# CoT策略配置
COT_STRATEGIES = {
    "baseline": {
        "name": "Baseline (无CoT)",
        "description": "直接向模型提问，不添加任何CoT提示"
    },
    "zero_shot": {
        "name": "Zero-shot CoT",
        "description": "在提示的最后添加'Let's think step by step.'",
        "prompt_suffix": "Let's think step by step."
    },
    "few_shot": {
        "name": "Few-shot CoT",
        "description": "使用向量数据库检索相似问题及其答案作为示例",
        "num_examples": 2  # 检索的示例数量
    },
    "auto_cot": {
        "name": "Auto-CoT",
        "description": "使用向量数据库检索相似问题，并为其生成CoT推理过程",
        "num_examples": 2,  # 检索的示例数量
        "cot_prefix": "Let's think step by step。"
    },
    "auto_reason": {
        "name": "AutoReason",
        "description": "使用强模型生成详细的推理链",
        "reasoning_prompt": "您将获得一个问题，并使用该问题将其分解为一系列逻辑推理轨迹。仅写下推理过程，不要自己回答问题",
        "reasoning_model": REASONING_MODEL
    },
    "combined": {
        "name": "Auto-CoT + AutoReason",
        "description": "结合Auto-CoT和AutoReason的优势",
        "num_examples": 2,  # 检索的示例数量
        "reasoning_model": REASONING_MODEL  # 用于生成推理链的模型
    }
}

# 评估指标配置
EVALUATION_METRICS = {
    "accuracy": {
        "name": "准确率",
        "description": "模型回答的正确率",
        "weight": 0.4
    },
    "reasoning_quality": {
        "name": "推理质量",
        "description": "评估模型推理过程的合理性和逻辑性",
        "weight": 0.3,
        "prompt": "评估以下回答的推理质量。考虑推理的清晰度、逻辑性和步骤的合理性。评分从1到10，其中1表示推理质量很差，10表示推理质量极佳。"
    },
    "robustness": {
        "name": "鲁棒性",
        "description": "在不同类型问题上的表现一致性",
        "weight": 0.2
    },
    "efficiency": {
        "name": "效率",
        "description": "生成答案所需的时间和计算资源",
        "weight": 0.1
    }
}
