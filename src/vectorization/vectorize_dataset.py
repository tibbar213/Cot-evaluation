"""
数据集向量化工具，用于将问题集转换为向量形式
"""

import json
import logging
import time
import argparse
from typing import Dict, List, Any
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import get_embedding
from vectorization.vector_store import VectorStore

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

def vectorize_questions(
    questions: List[Dict[str, Any]],
    vector_store: VectorStore,
    batch_size: int = 10
) -> None:
    """
    将问题集向量化并存储
    
    Args:
        questions (List[Dict[str, Any]]): 问题集
        vector_store (VectorStore): 向量存储对象
        batch_size (int): 批处理大小
    """
    total_questions = len(questions)
    logger.info(f"开始向量化 {total_questions} 个问题")
    
    # 使用tqdm创建进度条
    with tqdm(total=total_questions, desc="向量化进度") as pbar:
        for i in range(0, total_questions, batch_size):
            batch = questions[i:i + batch_size]
            
            # 处理每个问题
            for question in batch:
                try:
                    # 获取问题的向量表示
                    question_text = question["question"]
                    vector = get_embedding(question_text)
                    
                    # 验证向量维度
                    if len(vector) != 1024:
                        logger.error(f"向量维度不正确: {len(vector)}")
                        continue
                    
                    # 存储向量和元数据
                    metadata = {
                        "id": question["id"],
                        "question": question_text,
                        "answer": question["answer"],
                        "category": question.get("category", ""),
                        "difficulty": question.get("difficulty", "")
                    }
                    
                    vector_store.add_vector(vector, metadata)
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"处理问题 {question.get('id', 'unknown')} 时出错: {e}")
                    continue
    
    logger.info("向量化完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据集向量化工具")
    parser.add_argument("--questions", type=str, default="data/questions.json", help="问题集文件路径")
    parser.add_argument("--batch-size", type=int, default=10, help="批处理大小")
    parser.add_argument("--output", type=str, default="data/vector_store", help="向量存储输出目录")
    
    args = parser.parse_args()
    
    # 加载问题集
    questions = load_questions(args.questions)
    if not questions:
        logger.error("未能加载问题集，程序退出")
        return
    
    # 创建向量存储
    vector_store = VectorStore(args.output)
    
    # 开始向量化
    start_time = time.time()
    vectorize_questions(questions, vector_store, args.batch_size)
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    logger.info(f"向量化完成，总耗时: {elapsed_time:.2f}秒")
    logger.info(f"向量存储已保存在: {args.output}")

if __name__ == "__main__":
    main() 