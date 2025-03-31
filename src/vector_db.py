"""
向量数据库接口，用于存储和检索向量化的问题
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import faiss
from pathlib import Path

from config import VECTOR_DB_PATH, QUESTIONS_PATH
from models import get_embedding

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDatabase:
    """向量数据库类，用于存储和检索向量化的问题"""
    
    def __init__(self, db_path: str = VECTOR_DB_PATH):
        """
        初始化向量数据库
        
        Args:
            db_path (str): 向量数据库存储路径
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.db_path / "faiss_index.bin"
        self.metadata_path = self.db_path / "metadata.json"
        
        self.index = None
        self.metadata = []
        
        # 加载或创建索引
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """加载或创建FAISS索引"""
        if self.index_path.exists() and self.metadata_path.exists():
            # 加载现有索引
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"已加载向量数据库，包含 {len(self.metadata)} 条记录")
            except Exception as e:
                logger.error(f"加载向量数据库时出错: {e}")
                self._create_new_index()
        else:
            # 创建新索引
            self._create_new_index()
    
    def _create_new_index(self):
        """创建新的FAISS索引"""
        try:
            # 创建一个空的元数据列表
            self.metadata = []
            
            # 确定向量维度
            dimension = 1024  # BAAI/bge-m3 的维度
            
            # 创建索引
            self.index = faiss.IndexFlatL2(dimension)
            logger.info(f"已创建新的向量数据库，维度: {dimension}")
        except Exception as e:
            logger.error(f"创建向量数据库时出错: {e}")
            raise
    
    def add_question(self, question: str, metadata: Dict[str, Any]) -> int:
        """
        添加问题到向量数据库
        
        Args:
            question (str): 问题文本
            metadata (Dict[str, Any]): 问题的元数据
            
        Returns:
            int: 添加的问题ID
        """
        try:
            # 获取问题的向量嵌入
            embedding = get_embedding(question)
            
            # 添加到索引
            embedding_np = np.array([embedding], dtype=np.float32)
            self.index.add(embedding_np)
            
            # 添加元数据
            question_id = len(self.metadata)
            metadata['id'] = question_id
            metadata['question'] = question
            self.metadata.append(metadata)
            
            # 保存索引和元数据
            self._save()
            
            logger.info(f"已添加问题到向量数据库，ID: {question_id}")
            return question_id
        
        except Exception as e:
            logger.error(f"添加问题到向量数据库时出错: {e}")
            raise
    
    def search(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        """
        搜索与查询最相似的问题
        
        Args:
            query (str): 查询文本
            k (int): 返回的最相似问题数量
            
        Returns:
            List[Dict[str, Any]]: 最相似问题的元数据列表
        """
        try:
            # 获取查询的向量嵌入
            query_embedding = get_embedding(query)
            
            # 转换为numpy数组
            query_embedding_np = np.array([query_embedding], dtype=np.float32)
            
            # 搜索最相似的向量
            k = min(k, len(self.metadata))  # 确保k不超过元数据长度
            if k == 0:
                return []
            
            distances, indices = self.index.search(query_embedding_np, k)
            
            # 获取对应的元数据
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['distance'] = float(distances[0][i])
                    results.append(result)
            
            logger.info(f"搜索完成，找到 {len(results)} 条结果")
            return results
        
        except Exception as e:
            logger.error(f"搜索向量数据库时出错: {e}")
            return []
    
    def _save(self):
        """保存索引和元数据"""
        try:
            # 保存索引
            faiss.write_index(self.index, str(self.index_path))
            
            # 保存元数据
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已保存向量数据库，包含 {len(self.metadata)} 条记录")
        
        except Exception as e:
            logger.error(f"保存向量数据库时出错: {e}")
            raise
    
    def load_questions_from_json(self, json_path: str = QUESTIONS_PATH) -> int:
        """
        从JSON文件加载问题到向量数据库
        
        Args:
            json_path (str): JSON文件路径
            
        Returns:
            int: 加载的问题数量
        """
        try:
            # 加载JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            # 添加问题到向量数据库
            count = 0
            for q in questions:
                metadata = {k: v for k, v in q.items() if k != 'question'}
                self.add_question(q['question'], metadata)
                count += 1
            
            logger.info(f"已从JSON文件加载 {count} 个问题到向量数据库")
            return count
        
        except Exception as e:
            logger.error(f"从JSON文件加载问题时出错: {e}")
            return 0
    
    def get_similar_questions(self, query: str, k: int = 2, exclude_exact_match: bool = True) -> List[Tuple[str, str]]:
        """
        获取与查询最相似的问题及其答案
        
        Args:
            query (str): 查询文本
            k (int): 返回的最相似问题数量
            exclude_exact_match (bool): 是否排除与查询几乎完全相同的问题（默认为True）
            
        Returns:
            List[Tuple[str, str]]: 最相似问题及其答案的元组列表
        """
        # 检索可能比所需结果多一个，以便在排除相似度最高的问题时仍有足够的结果
        actual_k = k + 1 if exclude_exact_match else k
        results = self.search(query, actual_k)
        
        # 如果需要排除与查询几乎完全相同的问题
        if exclude_exact_match and results:
            # 检查第一个结果的相似度是否非常高（距离非常小）
            if results and results[0].get('distance', 1.0) < 0.05:  # 距离阈值，可以根据需要调整
                logger.info(f"排除与查询几乎完全相同的问题: '{results[0].get('question', '')}'")
                results = results[1:]  # 排除第一个结果
        
        # 确保不超过请求的结果数量
        results = results[:k]
        
        return [(r['question'], r['answer']) for r in results if 'question' in r and 'answer' in r]
    
    def clear(self):
        """清空向量数据库"""
        self._create_new_index()
        if self.index_path.exists():
            os.remove(self.index_path)
        if self.metadata_path.exists():
            os.remove(self.metadata_path)
        logger.info("已清空向量数据库")
