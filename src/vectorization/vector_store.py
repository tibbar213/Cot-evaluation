"""
向量存储类，用于管理和检索向量数据
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import faiss

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """
    向量存储类，使用FAISS进行向量索引和检索
    """
    
    def __init__(self, store_dir: str):
        """
        初始化向量存储
        
        Args:
            store_dir (str): 存储目录路径
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化FAISS索引
        self.dimension = 1024  # BAAI/bge-m3的嵌入维度
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # 存储元数据
        self.metadata: List[Dict[str, Any]] = []
        
        # 如果存在已保存的索引，则加载
        self._load_existing_index()
    
    def _load_existing_index(self) -> None:
        """加载已存在的索引和元数据"""
        index_path = self.store_dir / "index.bin"
        metadata_path = self.store_dir / "metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            try:
                # 加载索引
                self.index = faiss.read_index(str(index_path))
                
                # 加载元数据
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                logger.info(f"已加载 {len(self.metadata)} 个向量")
            except Exception as e:
                logger.error(f"加载已存在的索引时出错: {e}")
    
    def save(self) -> None:
        """保存索引和元数据"""
        try:
            # 保存索引
            index_path = self.store_dir / "index.bin"
            faiss.write_index(self.index, str(index_path))
            
            # 保存元数据
            metadata_path = self.store_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已保存 {len(self.metadata)} 个向量")
        except Exception as e:
            logger.error(f"保存索引时出错: {e}")
            raise
    
    def add_vector(self, vector: List[float], metadata: Dict[str, Any]) -> None:
        """
        添加向量和元数据
        
        Args:
            vector (List[float]): 向量数据
            metadata (Dict[str, Any]): 元数据
        """
        try:
            # 将向量转换为numpy数组
            vector_array = np.array([vector], dtype=np.float32)
            
            # 添加到FAISS索引
            self.index.add(vector_array)
            
            # 保存元数据
            self.metadata.append(metadata)
            
            # 保存到磁盘
            self.save()
            
        except Exception as e:
            logger.error(f"添加向量时出错: {e}")
            logger.error(f"向量维度: {len(vector)}")
            logger.error(f"元数据: {metadata}")
            raise
    
    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索最相似的向量
        
        Args:
            query_vector (List[float]): 查询向量
            k (int): 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        try:
            # 将查询向量转换为numpy数组
            query_array = np.array([query_vector], dtype=np.float32)
            
            # 搜索最相似的向量
            distances, indices = self.index.search(query_array, k)
            
            # 获取结果
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):  # 确保索引有效
                    result = self.metadata[idx].copy()
                    result["distance"] = float(distance)
                    result["rank"] = i + 1
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"搜索向量时出错: {e}")
            return []
    
    def get_vector_count(self) -> int:
        """
        获取存储的向量数量
        
        Returns:
            int: 向量数量
        """
        return len(self.metadata) 