"""
数据集加载模块，用于从Hugging Face的datasets库加载LiveBench数据集
"""

import os
import logging
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from datasets import load_dataset, Dataset

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义默认的数据集缓存目录
DEFAULT_CACHE_DIR = "data/hf_datasets"

def load_livebench_dataset(
    dataset_name: str, 
    split: str = "test",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    local_json_path: Optional[str] = None,
    save_to_json: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    加载LiveBench数据集
    
    Args:
        dataset_name (str): 数据集名称，如 "livebench/math", "livebench/reasoning", "livebench/data_analysis"
        split (str): 数据集分割，默认为 "test"
        max_samples (Optional[int]): 最大样本数量，如果为None则加载全部
        cache_dir (Optional[str]): 数据集缓存目录，默认为"data/hf_datasets"
        local_json_path (Optional[str]): 本地JSON文件路径，如果提供则从本地加载而非HF
        save_to_json (Optional[str]): 保存转换后的数据集到JSON文件的路径
        
    Returns:
        List[Dict[str, Any]]: 格式化后的问题列表
    """
    questions = []
    
    try:
        # 如果提供了本地JSON文件路径，则从本地加载
        if local_json_path and os.path.exists(local_json_path):
            logger.info(f"从本地JSON文件 {local_json_path} 加载数据集")
            with open(local_json_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            logger.info(f"已从本地JSON文件加载 {len(questions)} 个问题")
            
            # 如果指定了最大样本数量，则进行截断
            if max_samples and max_samples < len(questions):
                questions = questions[:max_samples]
                logger.info(f"已截断至 {len(questions)} 个问题")
                
            return questions
        
        # 否则，从Hugging Face加载
        logger.info(f"正在从Hugging Face加载数据集 {dataset_name}，分割: {split}")
        
        # 创建缓存目录
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"使用缓存目录: {cache_dir}")
        
        # 加载数据集
        ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        
        # 如果指定了最大样本数量，则进行截断
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        
        # 将数据集转换为列表格式
        questions = convert_dataset_to_questions(ds, dataset_name)
        
        # 如果指定了保存路径，则保存到本地JSON文件
        if save_to_json:
            save_dir = os.path.dirname(save_to_json)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            with open(save_to_json, 'w', encoding='utf-8') as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已将 {len(questions)} 个问题保存到 {save_to_json}")
        
        logger.info(f"已加载 {len(questions)} 个问题")
        return questions
    
    except Exception as e:
        logger.error(f"加载数据集 {dataset_name} 时出错: {e}")
        return []

def combine_datasets(
    dataset_names: List[str], 
    max_samples_per_dataset: Optional[int] = None,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    local_json_dir: Optional[str] = None,
    save_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    组合多个数据集
    
    Args:
        dataset_names (List[str]): 数据集名称列表
        max_samples_per_dataset (Optional[int]): 每个数据集的最大样本数量
        cache_dir (Optional[str]): 数据集缓存目录
        local_json_dir (Optional[str]): 本地JSON文件目录，如果提供则从本地加载
        save_dir (Optional[str]): 保存转换后的数据集到JSON文件的目录
        
    Returns:
        List[Dict[str, Any]]: 组合后的问题列表
    """
    combined_questions = []
    
    for dataset_name in dataset_names:
        # 构建本地JSON文件路径
        local_json_path = None
        if local_json_dir:
            dataset_simple_name = dataset_name.split("/")[-1]
            local_json_path = os.path.join(local_json_dir, f"{dataset_simple_name}.json")
        
        # 构建保存JSON文件路径
        save_to_json = None
        if save_dir:
            dataset_simple_name = dataset_name.split("/")[-1]
            save_to_json = os.path.join(save_dir, f"{dataset_simple_name}.json")
        
        # 加载数据集
        questions = load_livebench_dataset(
            dataset_name, 
            max_samples=max_samples_per_dataset,
            cache_dir=cache_dir,
            local_json_path=local_json_path,
            save_to_json=save_to_json
        )
        
        combined_questions.extend(questions)
        logger.info(f"已添加 {len(questions)} 个问题 (来自 {dataset_name})")
    
    logger.info(f"总共组合了 {len(combined_questions)} 个问题")
    return combined_questions 

def convert_dataset_to_questions(ds: Dataset, dataset_name: str) -> List[Dict[str, Any]]:
    """
    将数据集转换为问题列表格式
    
    Args:
        ds (Dataset): Hugging Face数据集
        dataset_name (str): 数据集名称
        
    Returns:
        List[Dict[str, Any]]: 格式化后的问题列表
    """
    questions = []
    
    # 根据数据集类型选择不同的转换方式
    if "math" in dataset_name:
        category = "math"
        for i, item in enumerate(ds):
            question_id = f"math_{i+1}"
            
            # LiveBench/math 数据集格式处理
            if "question_id" in item:
                # 使用原始问题ID
                question_id = f"math_{item['question_id']}"
                
            if "turns" in item and len(item["turns"]) > 0:
                # 第一个turn通常是问题
                question_text = item["turns"][0]
                # 使用ground_truth作为答案
                answer = str(item.get("ground_truth", ""))
                sub_category = item.get("category", "math")
                
                # 确定难度
                if "hardness" in item and item["hardness"] is not None:
                    try:
                        hardness = float(item["hardness"])
                        if hardness < 0.3:
                            difficulty = "easy"
                        elif hardness < 0.7:
                            difficulty = "medium"
                        else:
                            difficulty = "hard"
                    except (TypeError, ValueError):
                        difficulty = "medium"
                        logger.warning(f"无法转换hardness值 '{item['hardness']}' 为浮点数，使用默认难度 'medium'")
                else:
                    difficulty = "medium"
                
                questions.append({
                    "id": question_id,
                    "question": question_text,
                    "answer": answer,
                    "category": sub_category,
                    "difficulty": difficulty
                })
    
    elif "reasoning" in dataset_name:
        category = "reasoning"
        for i, item in enumerate(ds):
            question_id = f"reasoning_{i+1}"
            
            # LiveBench/reasoning 数据集格式处理
            if "question_id" in item:
                # 使用原始问题ID
                question_id = f"reasoning_{item['question_id']}"
            
            if "turns" in item and len(item["turns"]) > 0:
                # 第一个turn通常是问题
                question_text = item["turns"][0]
                # 使用ground_truth作为答案
                answer = str(item.get("ground_truth", ""))
                sub_category = item.get("category", "reasoning")
                difficulty = item.get("difficulty", "medium")
                
                questions.append({
                    "id": question_id,
                    "question": question_text,
                    "answer": answer,
                    "category": sub_category,
                    "difficulty": difficulty
                })
    
    elif "data_analysis" in dataset_name:
        category = "data_analysis"
        for i, item in enumerate(ds):
            question_id = f"data_analysis_{i+1}"
            
            # LiveBench/data_analysis 数据集格式处理
            if "question_id" in item:
                # 使用原始问题ID
                question_id = f"data_analysis_{item['question_id']}"
            
            if "turns" in item and len(item["turns"]) > 0:
                # 第一个turn通常是问题
                question_text = item["turns"][0]
                # 使用ground_truth作为答案
                answer = str(item.get("ground_truth", ""))
                sub_category = item.get("category", "data_analysis")
                difficulty = item.get("difficulty", "hard")
                
                questions.append({
                    "id": question_id,
                    "question": question_text,
                    "answer": answer,
                    "category": sub_category,
                    "difficulty": difficulty
                })
    
    else:
        # 默认处理方式，尝试处理未知格式的数据集
        for i, item in enumerate(ds):
            question_id = f"question_{i+1}"
            
            # 尝试从不同字段提取问题和答案
            question_text = None
            answer = None
            
            # 如果有turns字段
            if "turns" in item and len(item["turns"]) > 0:
                question_text = item["turns"][0]
            # 如果有question字段
            elif "question" in item:
                question_text = item["question"]
            
            # 如果有ground_truth字段
            if "ground_truth" in item:
                answer = str(item["ground_truth"])
            # 如果有answer字段
            elif "answer" in item:
                answer = str(item["answer"])
            
            # 如果没有找到问题或答案，跳过
            if not question_text or not answer:
                logger.warning(f"跳过项 {i+1}：未找到问题或答案")
                continue
            
            # 处理类别和难度
            category = item.get("category", "general")
            difficulty = item.get("difficulty", "medium")
            
            questions.append({
                "id": question_id,
                "question": question_text,
                "answer": answer,
                "category": category,
                "difficulty": difficulty
            })
    
    logger.info(f"已将数据集 {dataset_name} 转换为 {len(questions)} 个问题")
    return questions 