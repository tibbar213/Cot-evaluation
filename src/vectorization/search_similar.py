"""
相似问题检索工具，用于查找与给定问题最相似的问题
"""

import logging
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import get_embedding
from vectorization.vector_store import VectorStore

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def search_similar_questions(
    query: str,
    vector_store: VectorStore,
    k: int = 3
) -> List[Dict[str, Any]]:
    """
    搜索与给定问题最相似的问题
    
    Args:
        query (str): 查询问题
        vector_store (VectorStore): 向量存储对象
        k (int): 返回的相似问题数量
        
    Returns:
        List[Dict[str, Any]]: 相似问题列表
    """
    try:
        # 获取查询问题的向量表示
        query_vector = get_embedding(query)
        
        # 搜索相似问题
        results = vector_store.search(query_vector, k)
        
        return results
    except Exception as e:
        logger.error(f"搜索相似问题时出错: {e}")
        return []

def print_results(results: List[Dict[str, Any]], query: str) -> None:
    """
    打印搜索结果
    
    Args:
        results (List[Dict[str, Any]]): 搜索结果
        query (str): 查询问题
    """
    print(f"\n查询问题: {query}")
    print("-" * 80)
    
    for result in results:
        print(f"\n{result['rank']}. 相似度: {1 - result['distance']:.4f}")  # 将距离转换为相似度
        print(f"问题: {result['question']}")
        print(f"答案: {result['answer']}")
        print(f"类别: {result.get('category', '未知')}")
        print(f"难度: {result.get('difficulty', '未知')}")
        print("-" * 80)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="相似问题检索工具")
    parser.add_argument("--query", type=str, required=True, help="查询问题")
    parser.add_argument("--k", type=int, default=3, help="返回的相似问题数量")
    parser.add_argument("--vector-store", type=str, default="data/vector_store", help="向量存储目录")
    
    args = parser.parse_args()
    
    # 创建向量存储对象
    vector_store = VectorStore(args.vector_store)
    
    # 搜索相似问题
    results = search_similar_questions(args.query, vector_store, args.k)
    
    # 打印结果
    if results:
        print_results(results, args.query)
    else:
        print("未找到相似问题")

if __name__ == "__main__":
    main() 