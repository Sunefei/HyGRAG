"""
HKGraphTreeDynamic: 增量更新版本的HKGraphTree

集成了EraRAG TreeGraphDynamic的增量更新机制到HKGraphTree的混合图架构中
"""

import asyncio
import re
import numpy as np
import random
import pickle
import os
from collections import defaultdict, deque
from typing import Any, List, Dict, Set, Tuple, Optional
from itertools import combinations
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import faiss

from Core.Graph.HKGraphTree import HKGraphTree
from Core.Common.Logger import logger
from Core.Common.Utils import clean_str, prase_json_from_response
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.Message import Message
from Core.Prompt import GraphPrompt
from Core.Prompt.Base import TextPrompt
from Core.Schema.EntityRelation import Entity, Relationship, HK_Node
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Common.Constants import (
    NODE_PATTERN,
    REL_PATTERN,
    GRAPH_FIELD_SEP
)
from Core.Storage.HKGraphTreeStorage import HKGraphTreeStorage
from Core.Utils.WAT import WATAnnotation
import requests
from Core.Common.Constants import GCUBE_TOKEN
from tqdm import tqdm


class HKNodeAux:
    """
    HK图节点的辅助信息类，用于增量更新管理
    类似于TreeGraphDynamic中的DynTreeNodeAux
    """
    def __init__(self, node_id: str, node_type: str, level: int = 0, 
                 parent: Optional[str] = None, children: Optional[Set[str]] = None,
                 update_flag: bool = False, valid_flag: bool = True):
        self.node_id = node_id
        self.node_type = node_type  # 'entity', 'chunk', 'community'
        self.level = level  # 层次级别
        self.parent = parent  # 父节点ID
        self.children = children or set()  # 子节点ID集合
        self.update_flag = update_flag  # 是否需要更新
        self.valid_flag = valid_flag  # 是否有效
        self.last_modified = None  # 最后修改时间
        self.signature = None  # LSH签名


class HKDynamicAux:
    """
    HKGraphTree的动态辅助结构，管理增量更新的元信息
    参考TreeGraphDynamic中的DynAux设计
    """
    def __init__(self, workspace, shape: Tuple[int, int], force: bool = False):
        self.workspace = workspace
        # 如果workspace是字符串路径，则不需要ns_clustering；如果是workspace对象，则创建ns_clustering
        if workspace and hasattr(workspace, 'make_for'):
            self.ns_clustering = workspace.make_for("ns_clustering")
        else:
            self.ns_clustering = None
        
        # 文件路径定义 - 使用workspace构建正确的路径
        if workspace:
            # 判断workspace是路径字符串还是workspace对象
            if isinstance(workspace, str):
                # 如果是字符串路径，直接使用
                base_path = workspace
            elif hasattr(workspace, 'root_path'):
                # 如果是workspace对象，使用其root_path
                base_path = workspace.root_path
            else:
                # 其他情况，使用当前目录
                base_path = "."
            
            self.signature_file = os.path.join(base_path, "hk_signatures.pkl")
            self.hyperplane_file = os.path.join(base_path, "hk_hyperplanes.npy")
            self.aux_data_file = os.path.join(base_path, "hk_aux_data.pkl")
        else:
            # 如果没有workspace，使用当前目录（兼容性）
            self.signature_file = "hk_signatures.pkl"
            self.hyperplane_file = "hk_hyperplanes.npy"
            self.aux_data_file = "hk_aux_data.pkl"
        
        # 如果强制重置，删除现有文件
        if force:
            for file_path in [self.signature_file, self.hyperplane_file, self.aux_data_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed existing file: {file_path}")
        
        # 核心数据结构
        self.node_aux = {}  # node_id -> HKNodeAux
        self.signature_map = {}  # node_id -> LSH signature
        self.hyperplanes = self.get_hyperplanes(shape)  # 固定的LSH超平面
        self.affected_entities = set()  # 需要更新的节点集合
        self.level_to_nodes = defaultdict(set)  # level -> set of node_ids
        self.node_to_level = {}  # node_id -> level
        
        # 增量更新相关
        self.incremental_mode = False  # 是否处于增量更新模式
        self.base_graph_loaded = False  # 基础图是否已加载
        self.last_update_timestamp = None
        
        logger.info(f"🔧 HKDynamicAux initialized with hyperplane shape: {shape}")

    def save_hyperplanes(self, hyperplanes: np.ndarray):
        """保存超平面到文件"""
        np.save(self.hyperplane_file, hyperplanes)
        logger.info(f"Saved hyperplanes to {self.hyperplane_file}")

    def load_hyperplanes(self) -> bool:
        """从文件加载超平面"""
        if os.path.exists(self.hyperplane_file):
            self.hyperplanes = np.load(self.hyperplane_file)
            logger.info(f"✅ Loaded hyperplanes from {self.hyperplane_file}")
            return True
        return False

    def get_hyperplanes(self, shape: Tuple[int, int], force: bool = False) -> np.ndarray:
        """
        获取LSH超平面，确保一致性
        """
        if os.path.exists(self.hyperplane_file) and not force:
            hp = np.load(self.hyperplane_file)
            logger.info("✅ Hyperplane loaded from existing file!")
        else:
            # 使用固定种子确保可重现性
            np.random.seed(42)
            hp = np.random.randn(*shape)
            np.save(self.hyperplane_file, hp)
            logger.info("❌ No existing hyperplane! Generated new hyperplane with fixed seed!")
        return hp

    def save_aux_data(self):
        """保存辅助数据到文件"""
        aux_data = {
            'node_aux': {node_id: {
                'node_type': aux.node_type,
                'level': aux.level,
                'parent': aux.parent,
                'children': list(aux.children),
                'update_flag': aux.update_flag,
                'valid_flag': aux.valid_flag,
                'signature': aux.signature
            } for node_id, aux in self.node_aux.items()},
            'signature_map': self.signature_map,
            'affected_entities': list(self.affected_entities),
            'level_to_nodes': {level: list(nodes) for level, nodes in self.level_to_nodes.items()},
            'node_to_level': self.node_to_level,
            'incremental_mode': self.incremental_mode,
            'base_graph_loaded': self.base_graph_loaded
        }
        
        with open(self.aux_data_file, 'wb') as f:
            pickle.dump(aux_data, f)
        logger.info(f"Saved auxiliary data to {self.aux_data_file}")

    def load_aux_data(self) -> bool:
        """从文件加载辅助数据"""
        if not os.path.exists(self.aux_data_file):
            return False
            
        try:
            with open(self.aux_data_file, 'rb') as f:
                aux_data = pickle.load(f)
            
            # 恢复node_aux
            self.node_aux = {}
            for node_id, data in aux_data.get('node_aux', {}).items():
                aux = HKNodeAux(
                    node_id=node_id,
                    node_type=data['node_type'],
                    level=data['level'],
                    parent=data['parent'],
                    children=set(data['children']),
                    update_flag=data['update_flag'],
                    valid_flag=data['valid_flag']
                )
                aux.signature = data.get('signature')
                self.node_aux[node_id] = aux
            
            # 恢复其他数据结构
            self.signature_map = aux_data.get('signature_map', {})
            self.affected_entities = set(aux_data.get('affected_entities', []))
            self.level_to_nodes = defaultdict(set)
            for level, nodes in aux_data.get('level_to_nodes', {}).items():
                self.level_to_nodes[int(level)] = set(nodes)
            self.node_to_level = aux_data.get('node_to_level', {})
            self.incremental_mode = aux_data.get('incremental_mode', False)
            self.base_graph_loaded = aux_data.get('base_graph_loaded', False)
            
            logger.info(f"✅ Loaded auxiliary data with {len(self.node_aux)} nodes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load auxiliary data: {e}")
            return False

    def add_node_aux(self, node_id: str, node_type: str, level: int = 0, 
                     parent: Optional[str] = None, children: Optional[Set[str]] = None):
        """添加节点辅助信息"""
        aux = HKNodeAux(
            node_id=node_id,
            node_type=node_type,
            level=level,
            parent=parent,
            children=children or set(),
            update_flag=True,  # 新节点默认需要更新
            valid_flag=True
        )
        self.node_aux[node_id] = aux
        self.level_to_nodes[level].add(node_id)
        self.node_to_level[node_id] = level
        
        # 标记为受影响的实体
        self.affected_entities.add(node_id)
        
        #logger.debug(f"Added node aux for {node_id} at level {level}")

    def update_node_level(self, node_id: str, new_level: int):
        """更新节点层级"""
        if node_id in self.node_aux:
            old_level = self.node_aux[node_id].level
            if old_level != new_level:
                # 从旧层级移除
                self.level_to_nodes[old_level].discard(node_id)
                # 添加到新层级
                self.level_to_nodes[new_level].add(node_id)
                self.node_to_level[node_id] = new_level
                self.node_aux[node_id].level = new_level
                self.node_aux[node_id].update_flag = True
                self.affected_entities.add(node_id)

    def set_parent_child_relationship(self, parent_id: str, child_id: str):
        """设置父子关系"""
        if parent_id in self.node_aux and child_id in self.node_aux:
            # 设置父子关系
            self.node_aux[child_id].parent = parent_id
            self.node_aux[parent_id].children.add(child_id)
            
            # 标记为受影响
            self.affected_entities.add(parent_id)
            self.affected_entities.add(child_id)

    def mark_node_invalid(self, node_id: str):
        """标记节点为无效"""
        if node_id in self.node_aux:
            self.node_aux[node_id].valid_flag = False
            self.affected_entities.add(node_id)
            
            # 从层级映射中移除
            level = self.node_aux[node_id].level
            self.level_to_nodes[level].discard(node_id)
            
            #logger.debug(f"Marked node {node_id} as invalid")

    def get_valid_nodes_at_level(self, level: int) -> List[str]:
        """获取指定层级的有效节点"""
        nodes = self.level_to_nodes.get(level, set())
        return [node_id for node_id in nodes 
                if node_id in self.node_aux and self.node_aux[node_id].valid_flag]

    def get_affected_nodes_at_level(self, level: int) -> List[str]:
        """获取指定层级需要更新的节点"""
        nodes = self.get_valid_nodes_at_level(level)
        return [node_id for node_id in nodes 
                if node_id in self.affected_entities]

    def clear_update_flags(self):
        """清除所有更新标志"""
        for aux in self.node_aux.values():
            aux.update_flag = False
        self.affected_entities.clear()

    def compute_signature(self, embedding: np.ndarray) -> int:
        """计算节点嵌入的LSH签名"""
        if self.hyperplanes is None:
            raise ValueError("Hyperplanes not initialized")
        
        projections = np.dot(embedding, self.hyperplanes.T)
        binary_hash = (projections > 0).astype(int)
        return int(''.join(map(str, binary_hash)), 2)

    def update_node_signature(self, node_id: str, embedding: np.ndarray):
        """更新节点的LSH签名"""
        signature = self.compute_signature(embedding)
        self.signature_map[node_id] = signature
        if node_id in self.node_aux:
            self.node_aux[node_id].signature = signature
        return signature

    def mark_node_affected(self, node_id: str):
        """标记节点为受影响"""
        self.affected_entities.add(node_id)
        if node_id in self.node_aux:
            self.node_aux[node_id].update_flag = True

    def get_statistics(self) -> Dict[str, Any]:
        """获取辅助结构的统计信息"""
        stats = {
            'total_nodes': len(self.node_aux),
            'valid_nodes': sum(1 for aux in self.node_aux.values() if aux.valid_flag),
            'affected_nodes': len(self.affected_entities),
            'level_distribution': {level: len(nodes) for level, nodes in self.level_to_nodes.items()},
            'node_type_distribution': defaultdict(int)
        }
        
        for aux in self.node_aux.values():
            if aux.valid_flag:
                stats['node_type_distribution'][aux.node_type] += 1
        
        return stats


class HKGraphTreeDynamic(HKGraphTree):
    """
    HKGraphTree的动态增量更新版本
    
    集成了EraRAG TreeGraphDynamic的增量更新机制：
    1. 固定LSH超平面确保签名一致性
    2. 细粒度的受影响节点追踪
    3. 层次化的局部重构
    4. 高效的增量嵌入更新
    """
    
    def __init__(self, config, embed_config, llm, encoder, **kwargs):
        super().__init__(config, embed_config, llm, encoder, **kwargs)
        
        # 增量更新配置
        self.enable_incremental_update = getattr(config, 'enable_incremental_update', True)
        self.incremental_batch_size = getattr(config, 'incremental_batch_size', 10)
        self.max_affected_ratio = getattr(config, 'max_affected_ratio', 0.5)  # 最大受影响节点比例
        self.enable_cross_chunk_connections = getattr(config, 'enable_cross_chunk_connections', True)  # 启用新旧chunk连接
        
        # 初始化动态辅助结构
        hyperplane_shape = (self.lsh_num_hyperplanes, self.cleora_dim)
        workspace = getattr(config, 'faiss_index_path', './faiss_index_temp/')
        self.aux = HKDynamicAux(workspace, hyperplane_shape, force=False)
        
        # 增量更新状态管理
        self.incremental_mode = False
        self.base_hierarchy_built = False
        
        logger.info(f"🚀 HKGraphTreeDynamic initialized with incremental update enabled: {self.enable_incremental_update}")

    async def _load_graph(self, force: bool = False) -> bool:
        """
        重写加载方法，支持增量更新模式
        """
        # 首先尝试加载基础图和层次结构
        base_loaded = await super()._load_graph(force)
        
        if base_loaded:
            # 尝试加载辅助数据
            aux_loaded = self.aux.load_aux_data()
            if aux_loaded:
                self.aux.base_graph_loaded = True
                self.base_hierarchy_built = True
                logger.info("✅ Successfully loaded base graph and auxiliary data for incremental updates")
                return True
            else:
                logger.warning("⚠️ Base graph loaded but auxiliary data missing - will need to rebuild aux structure")
                # 如果基础图存在但辅助数据缺失，需要重新构建辅助结构
                await self._rebuild_aux_structure()
                return True
        
        return False

    async def _rebuild_aux_structure(self):
        """
        从现有图结构重建辅助数据结构
        """
        logger.info("🔧 Rebuilding auxiliary structure from existing graph...")
        
        # 清理现有辅助数据
        self.aux.node_aux.clear()
        self.aux.signature_map.clear()
        self.aux.affected_entities.clear()
        self.aux.level_to_nodes.clear()
        self.aux.node_to_level.clear()
        
        # 重建基础节点的辅助信息
        all_nodes = await self._graph.get_nodes()
        for node_id in all_nodes:
            node_data = await self._graph.get_node(node_id)
            if node_data:
                # 确定节点类型
                if node_id.startswith('CHUNK_'):
                    node_type = 'chunk'
                elif node_id.startswith('COMMUNITY_'):
                    node_type = 'community'
                else:
                    node_type = 'entity'
                
                # 添加到辅助结构
                self.aux.add_node_aux(node_id, node_type, level=0)
                
                # 如果有嵌入，计算签名
                if node_id in self.node_embeddings:
                    embedding = self.node_embeddings[node_id]
                    self.aux.update_node_signature(node_id, embedding)
        
        # 重建层次结构的辅助信息
        if hasattr(self, 'hierarchy_levels'):
            for level, communities in self.hierarchy_levels.items():
                for community_data in communities:
                    # 从community_data字典中提取community_id
                    if isinstance(community_data, dict):
                        community_id = community_data.get('id')
                    else:
                        # 兼容可能的字符串格式
                        community_id = community_data
                    
                    if community_id and community_id not in self.aux.node_aux:
                        self.aux.add_node_aux(community_id, 'community', level=int(level)+1)
                    elif community_id:
                        self.aux.update_node_level(community_id, int(level)+1)
                    
                    # 建立父子关系
                    if community_id:
                        children = self.community_children.get(community_id, [])
                        for child_id in children:
                            if child_id in self.aux.node_aux:
                                self.aux.set_parent_child_relationship(community_id, child_id)
        
        # 保存重建的辅助数据
        self.aux.save_aux_data()
        self.aux.base_graph_loaded = True
        self.base_hierarchy_built = True
        
        stats = self.aux.get_statistics()
        logger.info(f"✅ Rebuilt auxiliary structure: {stats}")

    async def _build_graph(self, chunk_list: List[Any]):
        """
        重写图构建方法，确保在初始构建时也创建辅助数据结构
        """
        # 调用父类的构建方法
        await super()._build_graph(chunk_list)
        
        # 如果是初始构建（非增量模式），创建辅助数据结构
        if not self.incremental_mode and self.enable_incremental_update:
            logger.info("🔧 Creating auxiliary data structure for future incremental updates")
            await self._create_initial_aux_structure()
    
    async def _create_initial_aux_structure(self):
        """
        为初始构建的图创建辅助数据结构
        """
        logger.info("🛠️ Creating initial auxiliary structure")
        
        # 清理现有辅助数据
        self.aux.node_aux.clear()
        self.aux.signature_map.clear()
        self.aux.affected_entities.clear()
        self.aux.level_to_nodes.clear()
        self.aux.node_to_level.clear()
        
        # 为所有基础节点创建辅助信息
        all_nodes = await self._graph.get_nodes()
        for node_id in all_nodes:
            node_data = await self._graph.get_node(node_id)
            if node_data:
                # 确定节点类型
                if node_id.startswith('CHUNK_'):
                    node_type = 'chunk'
                elif node_id.startswith('COMMUNITY_'):
                    node_type = 'community'
                else:
                    node_type = 'entity'
                
                # 添加到辅助结构
                self.aux.add_node_aux(node_id, node_type, level=0)
                
                # 如果有嵌入，计算签名
                if node_id in self.node_embeddings:
                    embedding = self.node_embeddings[node_id]
                    self.aux.update_node_signature(node_id, embedding)
        
        # 为层次结构创建辅助信息
        if hasattr(self, 'hierarchy_levels'):
            for level, communities in self.hierarchy_levels.items():
                for community_data in communities:
                    # 从community_data字典中提取community_id
                    if isinstance(community_data, dict):
                        community_id = community_data.get('id')
                    else:
                        community_id = community_data
                    
                    if community_id:
                        # 更新节点层级信息
                        if community_id in self.aux.node_aux:
                            self.aux.update_node_level(community_id, int(level)+1)
                        else:
                            self.aux.add_node_aux(community_id, 'community', level=int(level)+1)
                        
                        # 建立父子关系
                        children = self.community_children.get(community_id, [])
                        for child_id in children:
                            if child_id in self.aux.node_aux:
                                self.aux.set_parent_child_relationship(community_id, child_id)
        
        # 清除更新标志（初始状态下所有节点都是"新"的，但不需要更新标志）
        self.aux.clear_update_flags()
        
        # 保存辅助数据
        self.aux.save_aux_data()
        self.aux.base_graph_loaded = True
        self.base_hierarchy_built = True
        
        stats = self.aux.get_statistics()
        logger.info(f"✅ Created initial auxiliary structure: {stats}")

    async def insert_incremental(self, new_chunk_list: List[Any]) -> bool:
        """
        增量插入新的文档块
        
        Args:
            new_chunk_list: 新的文档块列表，格式：[(chunk_key, TextChunk), ...]
            
        Returns:
            bool: 插入是否成功
        """
        if not self.enable_incremental_update:
            logger.warning("Incremental update is disabled, falling back to full rebuild")
            return await self._build_graph(new_chunk_list)
        
        if not self.base_hierarchy_built:
            logger.info("Base hierarchy not built, performing initial build...")
            return await self._build_graph(new_chunk_list)
        
        logger.info(f"🚀 Starting incremental update with {len(new_chunk_list)} new chunks")
        
        try:
            # 设置增量模式
            self.incremental_mode = True
            self.aux.incremental_mode = True
            
            # Step 1: 处理新的文档块，构建基础图部分
            await self._process_incremental_chunks(new_chunk_list)
            
            # Step 2: 更新Cleora嵌入
            await self._update_cleora_embeddings_incremental()
            
            # Step 3: 执行增量层次化聚类
            await self._update_hierarchy_incremental()
            
            # Step 4: 更新FAISS索引
            await self._update_faiss_indexes_incremental()
            
            # Step 5: 保存更新后的数据
            await self._save_incremental_updates()
            
            # 清理更新标志
            self.aux.clear_update_flags()
            
            stats = self.aux.get_statistics()
            logger.info(f"✅ Incremental update completed successfully: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Incremental update failed: {e}")
            self.incremental_mode = False
            self.aux.incremental_mode = False
            raise
        
        finally:
            self.incremental_mode = False
            self.aux.incremental_mode = False

    async def _process_incremental_chunks(self, new_chunk_list: List[Any]):
        """
        处理新的文档块，增量构建基础图
        """
        logger.info(f"📝 Processing {len(new_chunk_list)} new chunks for incremental update")
        
        # Step 1: 对新chunks进行实体关系抽取
        er_results = []
        passage_results = []
        
        logger.info("🛠️ Extracting entities and relationships from new chunks")
        
        # 使用并发控制处理chunk
        er_results, passage_results = await self._process_chunks_with_concurrency_control(new_chunk_list)
        
        # Step 2: 增量构建混合图
        await self._build_incremental_hybrid_graph(er_results, passage_results, new_chunk_list)

    async def _process_chunks_with_concurrency_control(self, chunk_list: List[Any]) -> Tuple[List[Dict], List[Dict]]:
        """
        使用并发控制处理chunks的实体关系抽取
        """
        # 使用与父类相同的并发控制参数
        max_concurrent = getattr(self, 'max_concurrent_summaries', 35)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _process_single_chunk(chunk_data):
            async with semaphore:
                try:
                    # 从字典中提取chunk信息
                    if isinstance(chunk_data, dict):
                        chunk_content = chunk_data.get('content', '')
                        # 使用内容hash作为chunk_key，与存储系统保持一致
                        from Core.Common.Utils import mdhash_id
                        chunk_key = mdhash_id(chunk_content.strip(), prefix="doc-")
                    else:
                        # 兼容原有的元组格式
                        chunk_key, chunk_info = chunk_data
                        chunk_content = chunk_info.content
                    
                    # 实体关系抽取
                    if self.extract_two_step:
                        entities = await self._named_entity_recognition(chunk_content)
                        triples = await self._openie_post_ner_extract(chunk_content, entities)
                    else:
                        content = await self._kg_agent(chunk_content)
                        entities, triples = await self._parse_kg_content(content)
                    
                    entities_dict, relationships_dict = await self._build_graph_from_tuples(entities, triples, chunk_key)
                    er_result = {
                        'chunk_key': chunk_key,
                        'entities': entities_dict,
                        'relationships': relationships_dict
                    }
                    
                    # 维基百科实体链接（如果启用）
                    if self.use_wat_linking:
                        wiki_entities = await self._extract_wiki_entities(chunk_content)
                        passage_result = {
                            'chunk_key': chunk_key,
                            'wiki_entities': wiki_entities
                        }
                    else:
                        passage_result = {
                            'chunk_key': chunk_key,
                            'wiki_entities': entities_dict
                        }
                    
                    return er_result, passage_result
                    
                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk_key}: {e}")
                    return None, None
        
        # 执行并发处理
        tasks = [_process_single_chunk(chunk_data) for chunk_data in chunk_list]
        logger.info(f"🔧 Processing {len(tasks)} chunks with max concurrency {max_concurrent}")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 分离结果
        er_results = []
        passage_results = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Chunk processing failed with exception: {result}")
                continue
            elif result and len(result) == 2:
                er_result, passage_result = result
                if er_result and passage_result:
                    er_results.append(er_result)
                    passage_results.append(passage_result)
        
        logger.info(f"✅ Successfully processed {len(er_results)} chunks")
        return er_results, passage_results

    async def _build_incremental_hybrid_graph(self, er_results: List[Dict], passage_results: List[Dict], chunk_list: List[Any]):
        """
        增量构建混合图，只添加新的节点和边
        """
        logger.info("🛠️ Building incremental hybrid graph")
        
        all_entities = defaultdict(list)
        all_relationships = defaultdict(list)
        chunk_entities_map = defaultdict(set)
        entity_chunks_map = defaultdict(set)
        wiki_entities_map = defaultdict(list)
        
        # 处理ER结果
        for er_result in er_results:
            chunk_key = er_result['chunk_key']
            entities = er_result['entities']
            relationships = er_result['relationships']
            
            for entity_name, entity_list in entities.items():
                all_entities[entity_name].extend(entity_list)
                chunk_entities_map[chunk_key].add(entity_name)
                entity_chunks_map[entity_name].add(chunk_key)
            
            for rel_key, rel_list in relationships.items():
                all_relationships[rel_key].extend(rel_list)
        
        # 处理passage结果
        for passage_result in passage_results:
            chunk_key = passage_result['chunk_key']
            wiki_entities = passage_result['wiki_entities']
            
            for wiki_entity, _ in wiki_entities.items():
                wiki_entities_map[wiki_entity].append(chunk_key)
        
        # 创建新的chunk节点
        new_chunk_nodes = defaultdict(list)
        for chunk_data in chunk_list:
            # 从字典中提取chunk信息
            if isinstance(chunk_data, dict):
                chunk_content = chunk_data.get('content', '')
                # 使用内容hash作为chunk_key，与存储系统保持一致
                from Core.Common.Utils import mdhash_id
                chunk_key = mdhash_id(chunk_content.strip(), prefix="doc-")
            else:
                # 兼容原有的元组格式
                chunk_key, chunk_info = chunk_data
                chunk_content = chunk_info.content
            
            chunk_node_id = f"CHUNK_{chunk_key}"
            chunk_entity = HK_Node(
                entity_name=chunk_node_id,
                entity_type="CHUNK",
                description=chunk_content,
                source_id=chunk_key
            )
            new_chunk_nodes[chunk_node_id].append(chunk_entity)
            
            # 添加到辅助结构
            self.aux.add_node_aux(chunk_node_id, 'chunk', level=0)
        
        # 只添加新的实体节点（检查是否已存在）
        new_entity_nodes = defaultdict(list)
        for entity_name, entity_list in all_entities.items():
            # 检查实体是否已存在
            if not await self._graph.has_node(entity_name):
                new_entity_nodes[entity_name] = entity_list
                # 添加到辅助结构
                self.aux.add_node_aux(entity_name, 'entity', level=0)
            else:
                # 如果实体已存在，标记为受影响（可能需要更新连接）
                self.aux.affected_entities.add(entity_name)
        
        # 创建新的关系
        new_entity_chunk_relationships = defaultdict(list)
        new_chunk_chunk_relationships = defaultdict(list)
        
        # 实体-chunk连接
        for entity_name, chunk_keys in entity_chunks_map.items():
            for chunk_key in chunk_keys:
                rel_key = (entity_name, f"CHUNK_{chunk_key}")
                relationship = Relationship(
                    src_id=entity_name,
                    tgt_id=f"CHUNK_{chunk_key}",
                    relation_name="BELONGS_TO",
                    description=f"Entity {entity_name} belongs to chunk {chunk_key}",
                    source_id=f"{entity_name}_{chunk_key}",
                    weight=1.0
                )
                new_entity_chunk_relationships[rel_key].append(relationship)
        
        # Chunk-chunk连接（基于共享实体）- 包括新chunk与现有chunk的完整连接
        chunk_pair_shared_entities = defaultdict(list)
        
        # Step 1: 收集所有涉及的实体
        all_entities_in_new_chunks = set(wiki_entities_map.keys())
        logger.info(f"🔍 Processing {len(all_entities_in_new_chunks)} entities for chunk-chunk connections")
        
        # Step 2: 为每个实体查询现有图中的相关chunk（使用并发控制）
        entity_to_existing_chunks = {}
        if self.enable_cross_chunk_connections:
            logger.info("🔗 Cross-chunk connections enabled, querying existing chunks")
            entity_to_existing_chunks = await self._get_existing_chunks_for_entities_batch(all_entities_in_new_chunks)
        else:
            logger.info("🚫 Cross-chunk connections disabled, skipping existing chunk queries")
        
        logger.info(f"📊 Found {len(entity_to_existing_chunks)} entities with existing chunk connections")
        
        # Step 3: 建立完整的chunk-chunk连接（新-新、新-旧）
        for wiki_entity, new_chunk_keys in wiki_entities_map.items():
            # 获取该实体相关的现有chunk
            existing_chunk_keys = entity_to_existing_chunks.get(wiki_entity, [])
            
            # 合并新旧chunk列表
            all_chunk_keys = list(set(new_chunk_keys + existing_chunk_keys))
            
            if len(all_chunk_keys) < 2:
                continue
            
            # 生成所有可能的chunk对
            for chunk1, chunk2 in combinations(all_chunk_keys, 2):
                # 确保至少有一个是新chunk（避免重复处理旧chunk之间的连接）
                if chunk1 in new_chunk_keys or chunk2 in new_chunk_keys:
                    chunk_pair = tuple(sorted([chunk1, chunk2]))
                    chunk_pair_shared_entities[chunk_pair].append(wiki_entity)
        
        logger.info(f"🔗 Generated {len(chunk_pair_shared_entities)} potential chunk-chunk connections")
        
        # Step 4: 应用共享实体阈值并创建连接
        shared_entity_threshold = getattr(self.config, 'shared_entity_threshold', 2)
        new_new_connections = 0  # 新chunk与新chunk的连接
        new_old_connections = 0  # 新chunk与旧chunk的连接
        
        for (chunk1, chunk2), shared_entities in chunk_pair_shared_entities.items():
            if len(shared_entities) < shared_entity_threshold:
                continue
            
            # 统计连接类型
            chunk1_is_new = any(chunk1 in chunk_keys for chunk_keys in wiki_entities_map.values())
            chunk2_is_new = any(chunk2 in chunk_keys for chunk_keys in wiki_entities_map.values())
            
            if chunk1_is_new and chunk2_is_new:
                new_new_connections += 1
            elif chunk1_is_new or chunk2_is_new:
                new_old_connections += 1
            
            rel_key = tuple(sorted([f"CHUNK_{chunk1}", f"CHUNK_{chunk2}"]))
            relationship = Relationship(
                src_id=rel_key[0],
                tgt_id=rel_key[1],
                relation_name="SHARED_ENTITY",
                description=f"Chunks connected through shared entities: {', '.join(shared_entities)}",
                source_id=GRAPH_FIELD_SEP.join([chunk1, chunk2] + shared_entities),
                weight=float(len(shared_entities))
            )
            new_chunk_chunk_relationships[rel_key].append(relationship)
        
        logger.info(f"📊 Chunk-chunk connection statistics:")
        logger.info(f"   🔗 New-New connections: {new_new_connections}")
        logger.info(f"   🔗 New-Old connections: {new_old_connections}")
        logger.info(f"   🔗 Total connections: {len(new_chunk_chunk_relationships)}")
        logger.info(f"   📏 Shared entity threshold: {shared_entity_threshold}")
        
        # 将新节点和边添加到图中
        logger.info("🛠️ Adding new nodes and edges to graph")
        
        # 添加新节点（使用并发控制）
        all_new_nodes = {**new_entity_nodes, **new_chunk_nodes}
        if all_new_nodes:
            await self._add_nodes_with_concurrency_control(all_new_nodes)
        
        # 添加新边（使用并发控制）
        all_new_edges = {**all_relationships, **new_entity_chunk_relationships, **new_chunk_chunk_relationships}
        if all_new_edges:
            await self._add_edges_with_concurrency_control(all_new_edges)
        
        logger.info(f"✅ Added {len(all_new_nodes)} new nodes and {len(all_new_edges)} new edges to graph")

    async def _get_existing_chunks_for_entity(self, entity_name: str) -> List[str]:
        """
        查询现有图中包含指定实体的chunk节点
        
        Args:
            entity_name: 实体名称
            
        Returns:
            List[str]: 包含该实体的现有chunk的key列表（不包含CHUNK_前缀）
        """
        try:
            # 检查实体是否存在于图中
            if not await self._graph.has_node(entity_name):
                return []
            
            # 通过实体节点查找其邻居chunk节点
            neighbors = await self._graph.neighbors(entity_name)
            existing_chunks = []
            
            for neighbor in neighbors:
                if neighbor.startswith('CHUNK_'):
                    # 提取chunk key（移除CHUNK_前缀）
                    chunk_key = neighbor.replace('CHUNK_', '')
                    existing_chunks.append(chunk_key)
            
            logger.debug(f"Entity '{entity_name}' connected to {len(existing_chunks)} existing chunks")
            return existing_chunks
            
        except Exception as e:
            logger.warning(f"Failed to get existing chunks for entity '{entity_name}': {e}")
            return []

    async def _get_existing_chunks_for_entities_batch(self, entity_names: set) -> Dict[str, List[str]]:
        """
        批量查询多个实体在现有图中的相关chunk节点（使用并发控制）
        
        Args:
            entity_names: 实体名称集合
            
        Returns:
            Dict[str, List[str]]: 实体名称 -> chunk key列表的映射
        """
        if not entity_names:
            return {}
        
        # 使用与父类相同的并发控制参数
        max_concurrent = getattr(self, 'max_concurrent_summaries', 35)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _get_single_entity_chunks(entity_name):
            async with semaphore:
                existing_chunks = await self._get_existing_chunks_for_entity(entity_name)
                return entity_name, existing_chunks
        
        # 执行并发查询
        tasks = [_get_single_entity_chunks(entity_name) for entity_name in entity_names]
        logger.info(f"🔧 Querying existing chunks for {len(tasks)} entities with max concurrency {max_concurrent}")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        entity_to_existing_chunks = {}
        successful_queries = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Failed to query entity chunks: {result}")
                continue
            
            entity_name, existing_chunks = result
            if existing_chunks:
                entity_to_existing_chunks[entity_name] = existing_chunks
                successful_queries += 1
                #logger.debug(f"Entity '{entity_name}' found in {len(existing_chunks)} existing chunks")
        
        logger.info(f"✅ Successfully queried {successful_queries}/{len(entity_names)} entities for existing chunk connections")
        return entity_to_existing_chunks

    async def _add_nodes_with_concurrency_control(self, nodes_dict: Dict):
        """
        使用并发控制添加节点
        """
        # 使用与父类相同的并发控制参数
        max_concurrent = getattr(self, 'max_concurrent_summaries', 35)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _add_single_node(key, value):
            async with semaphore:
                await self._merge_nodes_then_upsert(key, value)
        
        tasks = [_add_single_node(k, v) for k, v in nodes_dict.items()]
        logger.info(f"🔧 Adding {len(tasks)} nodes with max concurrency {max_concurrent}")
        await asyncio.gather(*tasks)

    async def _add_edges_with_concurrency_control(self, edges_dict: Dict):
        """
        使用并发控制添加边
        """
        # 使用与父类相同的并发控制参数
        max_concurrent = getattr(self, 'max_concurrent_summaries', 35)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _add_single_edge(key, value):
            async with semaphore:
                if isinstance(key, tuple) and len(key) == 2:
                    await self._merge_edges_then_upsert(key[0], key[1], value)
                else:
                    logger.warning(f"Invalid edge key format: {key}")
        
        tasks = [_add_single_edge(k, v) for k, v in edges_dict.items()]
        logger.info(f"🔧 Adding {len(tasks)} edges with max concurrency {max_concurrent}")
        await asyncio.gather(*tasks)

    async def _update_cleora_embeddings_incremental(self):
        """
        增量更新Cleora嵌入，只处理受影响的节点
        使用与原版HKGraphTree相同的Cleora算法
        """
        logger.info("🔄 Updating Cleora embeddings incrementally")
        
        # 获取所有受影响的节点
        affected_nodes = list(self.aux.affected_entities)
        if not affected_nodes:
            logger.info("No affected nodes found, skipping embedding update")
            return
        
        logger.info(f"Updating embeddings for {len(affected_nodes)} affected nodes")
        
        # Step 1: 为新节点生成初始文本嵌入
        new_nodes = []
        for node_id in affected_nodes:
            if node_id not in self.node_text_embeddings:
                try:
                    if node_id.startswith('CHUNK_'):
                        # Chunk节点：使用chunk内容
                        chunk_key = node_id.replace('CHUNK_', '')
                        node_data = await self._graph.get_node(node_id)
                        if node_data and 'description' in node_data:
                            text_embedding = await self._embed_text(node_data['description'])
                            self.node_text_embeddings[node_id] = np.array(text_embedding)
                    else:
                        # 实体节点：使用实体名称和描述
                        node_data = await self._graph.get_node(node_id)
                        if node_data:
                            text_content = node_data.get('entity_name', node_id)
                            if 'description' in node_data:
                                text_content += ": " + node_data['description']
                            text_embedding = await self._embed_text(text_content)
                            self.node_text_embeddings[node_id] = np.array(text_embedding)
                    
                    # 新节点初始使用文本嵌入作为节点嵌入
                    if node_id in self.node_text_embeddings:
                        self.node_embeddings[node_id] = self.node_text_embeddings[node_id].copy()
                        new_nodes.append(node_id)
                        
                except Exception as e:
                    logger.warning(f"Failed to generate text embedding for {node_id}: {e}")
        
        logger.info(f"Generated text embeddings for {len(new_nodes)} new nodes")
        
        # Step 2: 获取所有节点的邻接信息（只需要受影响节点的邻接信息）
        adj_list = {}
        for node_id in affected_nodes:
            try:
                neighbors = list(await self._graph.neighbors(node_id))
                adj_list[node_id] = neighbors
            except Exception as e:
                logger.warning(f"Failed to get neighbors for {node_id}: {e}")
                adj_list[node_id] = []
        
        # Step 3: 执行Cleora迭代（与原版算法相同）
        logger.info(f"Running Cleora iterations (iterations={self.cleora_iterations})")
        
        for iteration in range(self.cleora_iterations):
            logger.debug(f"Cleora iteration {iteration + 1}/{self.cleora_iterations}")
            
            # 创建临时存储用于更新后的嵌入
            updated_embeddings = {}
            
            for node_id in affected_nodes:
                if node_id not in self.node_embeddings:
                    continue
                    
                neighbors = adj_list.get(node_id, [])
                
                if neighbors:
                    # 收集邻居嵌入（包括自身）
                    embeddings_to_aggregate = []
                    
                    # 添加自身嵌入
                    embeddings_to_aggregate.append(self.node_embeddings[node_id])
                    
                    # 添加邻居嵌入
                    for neighbor_id in neighbors:
                        if neighbor_id in self.node_embeddings:
                            embeddings_to_aggregate.append(self.node_embeddings[neighbor_id])
                    
                    # 聚合：计算平均值（与原版相同）
                    if embeddings_to_aggregate:
                        aggregated = np.mean(np.vstack(embeddings_to_aggregate), axis=0)
                        updated_embeddings[node_id] = aggregated
                else:
                    # 没有邻居，保持当前嵌入
                    updated_embeddings[node_id] = self.node_embeddings[node_id]
            
            # 归一化并更新嵌入
            for node_id, embedding in updated_embeddings.items():
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    self.node_embeddings[node_id] = embedding / norm
                else:
                    # 避免零向量
                    self.node_embeddings[node_id] = embedding
        
        # Step 4: 更新受影响节点的LSH签名
        updated_count = 0
        for node_id in affected_nodes:
            if node_id in self.node_embeddings:
                self.aux.update_node_signature(node_id, self.node_embeddings[node_id])
                updated_count += 1
        
        logger.info(f"✅ Updated Cleora embeddings for {updated_count} nodes using {self.cleora_iterations} iterations")

    async def _update_hierarchy_incremental(self):
        """
        增量更新层次化聚类结构
        """
        logger.info("🔄 Updating hierarchy incrementally")
        
        # 计算受影响节点的比例
        total_nodes = len(self.aux.node_aux)
        affected_count = len(self.aux.affected_entities)
        affected_ratio = affected_count / total_nodes if total_nodes > 0 else 1.0
        
        logger.info(f"Affected ratio: {affected_ratio:.2%} ({affected_count}/{total_nodes})")
        
        # 如果受影响的节点太多，执行全量重构
        if affected_ratio > self.max_affected_ratio:
            logger.info(f"Affected ratio {affected_ratio:.2%} > threshold {self.max_affected_ratio:.2%}, performing full hierarchy rebuild")
            await self._rebuild_full_hierarchy()
            return
        
        # 执行增量层次化更新
        await self._incremental_hierarchy_update()

    async def _incremental_hierarchy_update(self):
        """
        执行增量层次化更新
        """
        logger.info("🔧 Performing incremental hierarchy update")
        
        # 从底层开始，逐层处理受影响的节点
        max_level = max(self.aux.level_to_nodes.keys()) if self.aux.level_to_nodes else 0
        
        for level in range(max_level + 1):#TODO
            affected_nodes = self.aux.get_affected_nodes_at_level(level)
            if not affected_nodes:
                continue
            
            logger.info(f"Processing level {level} with {len(affected_nodes)} affected nodes")
            
            if level == 0:
                # 底层：处理新加入的基础节点
                await self._process_level_0_incremental(affected_nodes)
            else:
                # 上层：重新聚类受影响的社区
                await self._process_upper_level_incremental(level, affected_nodes)
        
        # 检查是否需要创建新的顶层
        await self._check_and_create_new_top_level()

    async def _process_level_0_incremental(self, affected_nodes: List[str]):
        """
        处理第0层（基础层）的增量更新
        """
        logger.info(f"Processing {len(affected_nodes)} affected base nodes")
        
        # 获取这些节点的嵌入
        affected_embeddings = []
        affected_node_ids = []
        
        for node_id in affected_nodes:
            if node_id in self.node_embeddings:
                affected_embeddings.append(self.node_embeddings[node_id])
                affected_node_ids.append(node_id)
        
        if not affected_embeddings:
            logger.warning("No embeddings found for affected nodes")
            return
        
        affected_embeddings = np.array(affected_embeddings)
        
        # 尝试将新节点分配到现有的1级社区
        level_1_communities = self.aux.get_valid_nodes_at_level(1)
        
        assignments = {}  # node_id -> community_id
        unassigned_nodes = []
        communities_to_update = set()  # 需要重新生成摘要的社区
        
        for i, node_id in enumerate(affected_node_ids):
            node_embedding = affected_embeddings[i]
            node_signature = self.aux.signature_map.get(node_id)
            
            best_community = None
            best_similarity = -1
            
            # 查找最相似的且未满的社区
            for community_id in level_1_communities:
                if community_id in self.node_embeddings:
                    # 检查社区大小限制
                    current_size = len(self.community_children.get(community_id, set()))
                    if current_size >= self.lsh_max_cluster_size:
                        #logger.debug(f"Community {community_id} is full ({current_size}/{self.lsh_max_cluster_size}), skipping")
                        continue
                    
                    community_embedding = self.node_embeddings[community_id]
                    similarity = np.dot(node_embedding, community_embedding)
                    
                    # 检查LSH签名兼容性
                    community_signature = self.aux.signature_map.get(community_id)
                    if node_signature and community_signature:
                        # 计算汉明距离
                        hamming_distance = bin(node_signature ^ community_signature).count('1')
                        # 如果汉明距离过大，降低相似性
                        if hamming_distance > self.lsh_num_hyperplanes // 4:
                            similarity *= 0.5 #TODO
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_community = community_id
            
            # 如果找到合适的社区且相似性足够高
            if best_community and best_similarity > 0.5:
                assignments[node_id] = best_community
                #logger.debug(f"Assigned {node_id} to {best_community} (similarity: {best_similarity:.3f})")
                
                # 更新社区的子节点信息
                self.aux.set_parent_child_relationship(best_community, node_id)
                self.community_children[best_community].add(node_id)
                self.community_parents[node_id] = best_community
                
                # 标记社区需要更新
                self.aux.affected_entities.add(best_community)
                communities_to_update.add(best_community)
                
                # 关键修复：当社区接收新成员时，也需要向上传播影响
                await self._propagate_impact_to_parent(best_community)
            else:
                unassigned_nodes.append(node_id)
        
        logger.info(f"Assigned {len(assignments)} nodes to existing communities, {len(unassigned_nodes)} nodes need new communities")
        
        # 为接收了新成员的社区重新生成摘要（使用并发控制）
        if communities_to_update:
            logger.info(f"Updating summaries for {len(communities_to_update)} communities that received new members")
            await self._update_communities_with_concurrency_control(list(communities_to_update), 0)
        
        # 为未分配的节点创建新社区（如果数量足够）
        if len(unassigned_nodes) >= self.lsh_min_cluster_size:
            await self._create_new_communities_for_unassigned(unassigned_nodes, 0)

    async def _create_new_communities_for_unassigned(self, unassigned_nodes: List[str], level: int):
        """
        为未分配的节点创建新社区（使用并发控制）
        """
        logger.info(f"Creating new communities for {len(unassigned_nodes)} unassigned nodes at level {level}")
        
        # 获取未分配节点的嵌入
        embeddings = []
        valid_node_ids = []
        for node_id in unassigned_nodes:
            if node_id in self.node_embeddings:
                embeddings.append(self.node_embeddings[node_id])
                valid_node_ids.append(node_id)
        
        if not embeddings:
            logger.warning("No embeddings found for unassigned nodes")
            return
        
        embeddings = np.array(embeddings)
        
        # 对未分配节点进行LSH聚类
        clusters = await self._lsh_clustering(embeddings, valid_node_ids)
        
        # 筛选出满足最小大小要求的聚类
        valid_clusters = []
        current_level_communities = len(self.hierarchy_levels.get(level, []))
        for i, cluster_nodes in enumerate(clusters):
            if len(cluster_nodes) >= self.lsh_min_cluster_size:
                community_id = f"COMMUNITY_L{level}_C{current_level_communities + i}"
                valid_clusters.append((community_id, cluster_nodes, level))
            else:
                logger.debug(f"Cluster {i} too small ({len(cluster_nodes)} < {self.lsh_min_cluster_size}), skipping")
        
        if not valid_clusters:
            logger.info("No valid clusters created (all clusters too small)")
            return
        
        logger.info(f"Creating {len(valid_clusters)} new communities with concurrent processing")
        
        # 使用并发控制生成社区摘要和嵌入
        await self._create_communities_with_concurrency_control(valid_clusters, level)
        
        logger.info(f"✅ Successfully created {len(valid_clusters)} new communities at level {level}")

    async def _create_communities_with_concurrency_control(self, valid_clusters: List[Tuple[str, List[str], int]], level: int):
        """
        使用并发控制创建社区
        
        Args:
            valid_clusters: [(community_id, cluster_nodes, level), ...]
            level: 当前层级
        """
        # 使用与父类相同的并发控制参数
        max_concurrent = getattr(self, 'max_concurrent_summaries', 35)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _create_single_community(community_data):
            community_id, cluster_nodes, community_level = community_data
            async with semaphore:
                try:
                    # 生成社区摘要和嵌入
                    await self._generate_community_summary_and_embedding(community_id, cluster_nodes, community_level)
                    
                    # 更新层次结构 - 保持与原始格式一致的字典结构
                    if level not in self.hierarchy_levels:
                        self.hierarchy_levels[level] = []
                    
                    community_data = {
                        'id': community_id,
                        'nodes': cluster_nodes,
                        'level': community_level
                    }
                    self.hierarchy_levels[level].append(community_data)
                    
                    # 设置父子关系
                    self.community_children[community_id] = set(cluster_nodes)
                    for child_id in cluster_nodes:
                        self.aux.set_parent_child_relationship(community_id, child_id)
                        self.community_parents[child_id] = community_id
                    
                    # 添加到辅助结构
                    self.aux.add_node_aux(community_id, 'community', level=level+1)
                    
                    # 关键修复：新创建的社区也需要向上传播影响
                    await self._propagate_impact_to_parent(community_id)
                    
                    logger.info(f"Created new community {community_id} with {len(cluster_nodes)} members")
                    return community_id
                    
                except Exception as e:
                    logger.error(f"Failed to create community {community_id}: {e}")
                    return None
        
        # 创建并发任务
        tasks = [_create_single_community(community_data) for community_data in valid_clusters]
        logger.info(f"🔧 Creating {len(tasks)} communities with max concurrency {max_concurrent}")
        
        # 执行并发创建
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        successful_creations = 0
        failed_creations = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Community creation failed with exception: {result}")
                failed_creations += 1
            elif result is not None:
                successful_creations += 1
            else:
                failed_creations += 1
        
        logger.info(f"🎯 Community creation completed: {successful_creations} successful, {failed_creations} failed")

    async def _process_upper_level_incremental(self, level: int, affected_nodes: List[str]):
        """
        处理上层的增量更新
        """
        logger.info(f"Processing upper level {level} with {len(affected_nodes)} affected nodes")
        
        # Step 1: 重新生成受影响社区的摘要和嵌入（使用并发控制）
        await self._update_communities_with_concurrency_control(affected_nodes, level)
        
        # Step 2: 处理该层新增的社区节点，需要将它们聚类到更高层
        await self._process_new_communities_for_upper_clustering(level)

    async def _update_communities_with_concurrency_control(self, community_ids: List[str], level: int):
        """
        使用并发控制更新社区摘要和嵌入
        """
        if not community_ids:
            return
            
        # 使用与父类相同的并发控制参数
        max_concurrent = getattr(self, 'max_concurrent_summaries', 35)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _update_single_community(community_id):
            async with semaphore:
                if community_id in self.community_children:
                    children = self.community_children[community_id]
                    await self._generate_community_summary_and_embedding(community_id, children, level)
                    # 更新LSH签名
                    if community_id in self.node_embeddings:
                        self.aux.update_node_signature(community_id, self.node_embeddings[community_id])
                    
                    # 关键修复：标记该社区的父节点为受影响，确保影响向上传播
                    await self._propagate_impact_to_parent(community_id)
        
        tasks = [_update_single_community(community_id) for community_id in community_ids]
        logger.info(f"🔧 Updating {len(tasks)} communities with max concurrency {max_concurrent}")
        await asyncio.gather(*tasks)
        
        # 统计向上传播的影响
        propagated_parents = set()
        for community_id in community_ids:
            parent_id = await self._get_community_parent_id(community_id)
            if parent_id:
                propagated_parents.add(parent_id)
        
        if propagated_parents:
            logger.info(f"📈 Impact propagated to {len(propagated_parents)} parent communities at level {level + 1}")
        else:
            logger.debug(f"🔝 No parent communities to propagate impact to (reached top level)")

    async def _process_new_communities_for_upper_clustering(self, level: int):
        """
        处理该层新增的社区节点，将它们聚类到更高层社区中
        
        Args:
            level: 当前处理的层级
        """
        logger.info(f"Processing new communities at level {level} for upper clustering")
        
        # 获取该层所有社区节点（包括新增的）
        current_level_communities = self.aux.get_valid_nodes_at_level(level)
        
        if not current_level_communities:
            logger.info(f"No communities found at level {level}")
            return
        
        # 识别新增的社区（没有父节点的社区）
        new_communities = []
        for community_id in current_level_communities:
            parent_id = await self._get_community_parent_id(community_id)
            if not parent_id:  # 没有父节点说明是新增的社区
                new_communities.append(community_id)
        
        if not new_communities:
            logger.info(f"No new communities found at level {level} that need upper clustering")
            return
        
        logger.info(f"Found {len(new_communities)} new communities at level {level} that need upper clustering")
        
        # 获取更高层的现有社区
        upper_level = level + 1
        upper_level_communities = self.aux.get_valid_nodes_at_level(upper_level)
        
        # 尝试将新社区分配到现有的更高层社区
        assignments = {}  # community_id -> parent_community_id
        unassigned_communities = []
        
        for community_id in new_communities:
            if community_id not in self.node_embeddings:
                logger.warning(f"Community {community_id} has no embedding, skipping")
                continue
                
            community_embedding = self.node_embeddings[community_id]
            community_signature = self.aux.signature_map.get(community_id)
            
            best_parent = None
            best_similarity = -1
            
            # 查找最相似的且未满的上层社区
            for parent_community_id in upper_level_communities:
                if parent_community_id in self.node_embeddings:
                    # 检查父社区大小限制
                    current_size = len(self.community_children.get(parent_community_id, set()))
                    if current_size >= self.lsh_max_cluster_size:
                        #logger.debug(f"Parent community {parent_community_id} is full ({current_size}/{self.lsh_max_cluster_size}), skipping")
                        continue
                    
                    parent_embedding = self.node_embeddings[parent_community_id]
                    similarity = np.dot(community_embedding, parent_embedding)
                    
                    # 检查LSH签名兼容性
                    parent_signature = self.aux.signature_map.get(parent_community_id)
                    if community_signature and parent_signature:
                        hamming_distance = bin(community_signature ^ parent_signature).count('1')
                        if hamming_distance > self.lsh_num_hyperplanes // 4:
                            similarity *= 0.5
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_parent = parent_community_id
            
            # 如果找到合适的父社区且相似性足够高
            if best_parent and best_similarity > 0.5:  # 使用相同的阈值
                assignments[community_id] = best_parent
                # 设置父子关系
                self.aux.set_parent_child_relationship(best_parent, community_id)
                self.community_parents[community_id] = best_parent
                if best_parent in self.community_children:
                    self.community_children[best_parent].add(community_id)
                else:
                    self.community_children[best_parent] = {community_id}
                
                # 标记父社区需要更新
                self.aux.affected_entities.add(best_parent)
                # 向上传播影响
                await self._propagate_impact_to_parent(best_parent)
                
                logger.info(f"Assigned community {community_id} to parent {best_parent} (similarity: {best_similarity:.3f})")
            else:
                unassigned_communities.append(community_id)
        
        logger.info(f"Assigned {len(assignments)} communities to existing upper communities, {len(unassigned_communities)} communities need new upper communities")
        
        # 为未分配的社区创建新的更高层社区（如果数量足够）
        if len(unassigned_communities) >= self.lsh_min_cluster_size:
            await self._create_new_communities_for_unassigned(unassigned_communities, upper_level-1)
        elif unassigned_communities:
            # 如果未分配的社区数量不足以创建新社区，但又不为空
            # 可以考虑降低阈值重新分配，或者等待更多社区
            logger.info(f"Only {len(unassigned_communities)} unassigned communities, less than minimum cluster size {self.lsh_min_cluster_size}")

    async def _propagate_impact_to_parent(self, community_id: str):
        """
        将影响传播到父节点
        
        Args:
            community_id: 当前更新的社区ID
        """
        try:
            # 获取父节点ID
            parent_id = await self._get_community_parent_id(community_id)
            
            if parent_id:
                # 标记父节点为受影响
                self.aux.mark_node_affected(parent_id)
                
                logger.debug(f"Propagated impact from {community_id} to parent {parent_id}")
            else:
                logger.debug(f"Community {community_id} has no parent (top level)")
                
        except Exception as e:
            logger.warning(f"Failed to propagate impact from {community_id}: {e}")

    async def _get_community_parent_id(self, community_id: str) -> Optional[str]:
        """
        获取社区的父节点ID
        
        Args:
            community_id: 社区ID
            
        Returns:
            Optional[str]: 父节点ID，如果没有父节点则返回None
        """
        try:
            # 方法1：从辅助结构中获取
            if community_id in self.aux.node_aux:
                parent_id = self.aux.node_aux[community_id].parent
                if parent_id:
                    return parent_id
            
            # 方法2：从community_parents映射中获取
            parent_id = self.community_parents.get(community_id)
            if parent_id:
                return parent_id
            
            # 方法3：通过层次结构推断（如果前两种方法都失败）
            # 解析当前社区的层级
            level_match = re.search(r'COMMUNITY_L(\d+)_C\d+', community_id)
            if level_match:
                current_level = int(level_match.group(1))
                parent_level = current_level + 1
                
                # 查找可能的父节点
                parent_communities = self.aux.get_valid_nodes_at_level(parent_level)
                for parent_candidate in parent_communities:
                    if parent_candidate in self.community_children:
                        if community_id in self.community_children[parent_candidate]:
                            return parent_candidate
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get parent for community {community_id}: {e}")
            return None

    async def _check_and_create_new_top_level(self):
        """
        检查是否需要为新增的社区创建更高层的顶层
        """
        logger.debug("Checking if new top level is needed")
        
        # 获取当前最高层级
        if not self.hierarchy_levels:
            return
        
        max_level = max(int(level) for level in self.hierarchy_levels.keys())
        #top_level_communities = self.hierarchy_levels.get(str(max_level), [])
        top_level_communities = self.hierarchy_levels.get(max_level)
        logger.info(f"Top level has {len(top_level_communities)} communities")
        
        # 如果顶层社区数量过多，考虑创建新的顶层
        if len(top_level_communities) > self.lsh_max_cluster_size:
            logger.info(f"Top level has {len(top_level_communities)} communities, considering creating new top level")
            
            # 检查是否已达到最大层次限制
            if max_level >= self.max_hierarchy_levels - 1:
                logger.info(f"Already at maximum hierarchy levels ({self.max_hierarchy_levels}), not creating new top level")
                return
            
            # 获取顶层社区的嵌入
            top_communities_with_embeddings = []
            for community_data in top_level_communities:
                if isinstance(community_data, dict):
                    community_id = community_data.get('id')
                else:
                    community_id = community_data
                
                if community_id and community_id in self.node_embeddings:
                    top_communities_with_embeddings.append(community_id)
            
            if len(top_communities_with_embeddings) >= self.lsh_min_cluster_size:
                logger.info(f"Creating new top level with {len(top_communities_with_embeddings)} communities")
                
                # 对顶层社区进行聚类
                embeddings = np.array([self.node_embeddings[cid] for cid in top_communities_with_embeddings])
                clusters = await self._lsh_clustering(embeddings, top_communities_with_embeddings)
                
                new_level = max_level + 1
                new_communities = []
                
                # 为每个聚类创建新的顶层社区
                for cluster_id, cluster_nodes in enumerate(clusters):
                    if len(cluster_nodes) >= self.lsh_min_cluster_size:
                        new_community_id = f"COMMUNITY_L{new_level}_C{cluster_id}"
                        
                        # 生成社区摘要和嵌入
                        await self._generate_community_summary_and_embedding(new_community_id, cluster_nodes, new_level)
                        
                        # 创建社区数据
                        community_data = {
                            'id': new_community_id,
                            'nodes': cluster_nodes,
                            'level': new_level
                        }
                        new_communities.append(community_data)
                        
                        # 设置父子关系
                        self.community_children[new_community_id] = set(cluster_nodes)
                        for child_id in cluster_nodes:
                            self.community_parents[child_id] = new_community_id
                        
                        # 添加到辅助结构
                        self.aux.add_node_aux(new_community_id, 'community', level=new_level)
                        
                        logger.info(f"Created new top-level community {new_community_id} with {len(cluster_nodes)} children")
                
                # 更新层次结构
                if new_communities:
                    self.hierarchy_levels[new_level] = new_communities
                    logger.info(f"Created new hierarchy level {new_level} with {len(new_communities)} communities")
        else:
            logger.info(f"Top level has {len(top_level_communities)} communities, no new top level needed")

    async def _rebuild_full_hierarchy(self):
        """
        执行完整的层次结构重构
        """
        logger.info("🔄 Rebuilding full hierarchy due to large number of affected nodes")
        
        # 清除现有层次结构
        self.hierarchy_levels.clear()
        self.community_summaries.clear()
        self.community_children.clear()
        self.community_parents.clear()
        
        # 重新构建层次结构
        await self._build_hierarchy()

    async def _update_faiss_indexes_incremental(self):
        """
        增量更新FAISS索引
        """
        logger.info("🔄 Updating FAISS indexes incrementally")
        
        # 获取所有受影响的节点
        affected_nodes = list(self.aux.affected_entities)
        if not affected_nodes:
            logger.info("No affected nodes, skipping FAISS index update")
            return
        
        # 重新构建FAISS索引（简化实现）
        # 在实际实现中，应该只更新受影响的部分
        await self._build_faiss_index()
        
        logger.info("✅ FAISS indexes updated")

    async def _save_incremental_updates(self):
        """
        保存增量更新的结果
        """
        logger.info("💾 Saving incremental updates")
        
        # 保存图结构
        await self._graph.persist(force=True)
        
        # 保存层次结构数据
        await self._save_hierarchy_to_storage(force=True)
        
        # 保存辅助数据
        self.aux.save_aux_data()
        
        # 保存FAISS索引
        await self._save_faiss_indexes()
        
        logger.info("✅ Incremental updates saved successfully")

    def get_incremental_statistics(self) -> Dict[str, Any]:
        """
        获取增量更新的统计信息
        """
        stats = self.aux.get_statistics()
        stats.update({
            'incremental_mode': self.incremental_mode,
            'base_hierarchy_built': self.base_hierarchy_built,
            'enable_incremental_update': self.enable_incremental_update,
            'hierarchy_levels': len(self.hierarchy_levels),
            'total_communities': sum(len(communities) for communities in self.hierarchy_levels.values()),
            'total_node_embeddings': len(self.node_embeddings),
            'total_signatures': len(self.aux.signature_map)
        })
        return stats
