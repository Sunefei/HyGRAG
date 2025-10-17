import asyncio
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

from Core.Common.Logger import logger
from Core.Retriever.BaseRetriever import BaseRetriever
from Core.Retriever.RetrieverFactory import register_retriever_method
from Core.Common.Utils import truncate_list_by_token_size, min_max_normalize
from Core.Common.Constants import Retriever


class HKGraphTreeRetriever(BaseRetriever):
    """
    HKGraphTree专用检索器：从顶向下检索层次化社区、实体、文档块和关系
    
    核心思路：
    1. 从顶层社区开始，使用查询相似性筛选相关社区
    2. 逐层向下展开，获取子社区和成员节点
    3. 在底层收集相关的实体、文档块和关系
    4. 整合多层次信息形成最终检索结果
    """
    
    def __init__(self, **kwargs):
        config = kwargs.pop("config")
        super().__init__(config)
        self.mode_list = ["hk_tree_search", "hk_tree_top_down", "hk_tree_comprehensive", "hk_tree_true_search", "hk_tree_flat_search"]
        self.type = "hk_tree"
        
        # 设置必需的属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Ensure key attributes exist
        if not hasattr(self, 'graph'):
            logger.warning("Graph attribute not found during HKGraphTreeRetriever initialization")
        if not hasattr(self, 'doc_chunk'):
            logger.warning("Doc_chunk attribute not found during HKGraphTreeRetriever initialization")


    async def _compute_community_similarity(self, query_embedding: np.ndarray, 
                                          community: Dict, summary: str) -> float:
        """
        计算查询与社区的相似性分数
        优先使用已保存的 node_text_embeddings，避免重复计算
        
        Args:
            query_embedding: 查询embedding
            community: 社区信息
            summary: 社区摘要
            
        Returns:
            相似性分数
        """
        try:
            community_id = community['id']
            
            # 方法1：优先使用预计算的文本embedding（最常见且高效）
            if (hasattr(self.graph, 'node_text_embeddings') and 
                community_id in self.graph.node_text_embeddings):
                community_embedding = self.graph.node_text_embeddings[community_id]
                if isinstance(community_embedding, np.ndarray):
                    similarity = await self.compute_similarity(query_embedding, community_embedding, "cosine")
                    return float(similarity)
            
            
            
            # 方法3：基于社区摘要重新计算embedding（应该很少触发）
            if summary and summary.strip():
                logger.debug(f"⚠️ No cached embedding for community {community_id}, recalculating from summary...")
                summary_embedding = await self._get_query_embedding(summary)
                similarity = np.dot(query_embedding, summary_embedding)
                return float(similarity)
            
            # 方法4：如果没有摘要，返回低分
            logger.warning(f"No embedding or summary for community {community_id}")
            return 0.1
            
        except Exception as e:
            logger.warning(f"Failed to compute similarity for community {community.get('id', 'UNKNOWN')}: {e}")
            return 0.0

    async def _extract_entities_and_chunks_from_nodes(self, member_nodes: Set[str], 
                                                    query_embedding: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        从成员节点中提取并排序实体和文档块
        
        Args:
            member_nodes: 成员节点集合
            query_embedding: 查询embedding
            
        Returns:
            (实体列表, 文档块列表)
        """
        entities = []
        chunks = []
        
        for node_id in member_nodes:
            if node_id.startswith('CHUNK_'):
                # 处理文档块节点
                chunk_data = await self._get_chunk_data_with_similarity(node_id, query_embedding)
                if chunk_data:
                    chunks.append(chunk_data)
            elif not node_id.startswith('COMMUNITY_'):
                # 处理实体节点（跳过社区节点）
                entity_data = await self._get_entity_data_with_similarity(node_id, query_embedding)
                if entity_data:
                    entities.append(entity_data)
        
        # 按相似性分数排序
        entities.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        logger.debug(f"📋 Extracted and sorted {len(entities)} entities and {len(chunks)} chunks")
        return entities, chunks

    async def _get_chunk_data_with_similarity(self, chunk_node_id: str, query_embedding: np.ndarray) -> Dict[str, Any]:
        """
        获取文档块数据并计算相似性分数
        优先使用已保存的 node_text_embeddings，避免重复计算
        """
        try:
            chunk_key = chunk_node_id.replace('CHUNK_', '')
            chunk_content = None
            
            if hasattr(self, 'doc_chunk') and self.doc_chunk is not None:
                # 检查get_data_by_key是否是async方法
                try:
                    if hasattr(self.doc_chunk, 'get_data_by_key'):
                        # 尝试async调用
                        if asyncio.iscoroutinefunction(self.doc_chunk.get_data_by_key):
                            chunk_content = await self.doc_chunk.get_data_by_key(chunk_key)
                        else:
                            # 同步调用
                            chunk_content = self.doc_chunk.get_data_by_key(chunk_key)
                    else:
                        logger.warning(f"doc_chunk does not have get_data_by_key method")
                        return None
                    
                    if chunk_content:
                        # 优先使用已保存的文本嵌入
                        if (hasattr(self.graph, 'node_text_embeddings') and 
                            chunk_node_id in self.graph.node_text_embeddings):
                            chunk_embedding = self.graph.node_text_embeddings[chunk_node_id]
                            if isinstance(chunk_embedding, np.ndarray):
                                similarity_score = await self.compute_similarity(query_embedding, chunk_embedding, "cosine")
                            else:
                                similarity_score = 0.0
                        else:
                            # 回退：重新计算 embedding
                            content_embedding = await self._get_query_embedding(chunk_content)
                            similarity_score = np.dot(query_embedding, content_embedding)
                        
                        return {
                            'id': chunk_key,
                            'content': chunk_content,
                            'type': 'chunk',
                            'similarity_score': float(similarity_score)
                        }
                except Exception as chunk_error:
                    logger.warning(f"Error accessing chunk data for {chunk_key}: {chunk_error}")
                    return None
        except Exception as e:
            logger.warning(f"Failed to get chunk {chunk_node_id} with similarity: {e}")
        return None

    async def _get_entity_data_with_similarity(self, entity_node_id: str, query_embedding: np.ndarray) -> Dict[str, Any]:
        """
        获取实体数据并计算相似性分数
        优先使用已保存的 node_text_embeddings，避免重复计算
        """
        try:
            entity_data = await self.graph.get_node(entity_node_id)
            if entity_data:
                # 优先使用已保存的文本嵌入
                if (hasattr(self.graph, 'node_text_embeddings') and 
                    entity_node_id in self.graph.node_text_embeddings):
                    entity_embedding = self.graph.node_text_embeddings[entity_node_id]
                    if isinstance(entity_embedding, np.ndarray):
                        similarity_score = await self.compute_similarity(query_embedding, entity_embedding, "cosine")
                    else:
                        similarity_score = 0.0
                else:
                    # 回退：重新计算 embedding（基于名称和描述）
                    entity_text = f"{entity_data.get('entity_name', '')} {entity_data.get('description', '')}"
                    if entity_text.strip():
                        entity_embedding = await self._get_query_embedding(entity_text)
                        similarity_score = np.dot(query_embedding, entity_embedding)
                    else:
                        similarity_score = 0.0
                
                return {
                    'entity_name': entity_data.get('entity_name', entity_node_id),
                    'entity_type': entity_data.get('entity_type', 'UNKNOWN'),
                    'description': entity_data.get('description', ''),
                    'type': 'entity',
                    'similarity_score': float(similarity_score)
                }
        except Exception as e:
            logger.warning(f"Failed to get entity {entity_node_id} with similarity: {e}")
        return None

    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """
        获取查询的嵌入向量
        """
        try:
            embedding = None
            
            # 尝试使用HKGraphTree的embedding方法
            if hasattr(self.graph, '_embed_text'):
                try:
                    embedding = await self.graph._embed_text(query)
                    #logger.debug(f"Used graph._embed_text for query embedding")
                except Exception as e:
                    logger.warning(f"Failed to use graph._embed_text: {e}")
            
            # 回退到直接使用embedding_model
            if embedding is None and hasattr(self.graph, 'embedding_model'):
                try:
                    embedding = self.graph.embedding_model._get_text_embedding(query)
                    logger.debug(f"Used graph.embedding_model for query embedding")
                except Exception as e:
                    logger.warning(f"Failed to use graph.embedding_model: {e}")
            
            # 尝试使用entities_vdb的embedding模型
            if embedding is None and hasattr(self, 'entities_vdb') and hasattr(self.entities_vdb, 'embedding_model'):
                try:
                    embedding = self.entities_vdb.embedding_model._get_text_embedding(query)
                    logger.debug(f"Used entities_vdb.embedding_model for query embedding")
                except Exception as e:
                    logger.warning(f"Failed to use entities_vdb.embedding_model: {e}")
            
            # 最后的回退：随机嵌入
            if embedding is None:
                logger.warning("No embedding model found, using random embeddings")
                embedding = np.random.normal(0, 0.1, 128)
            
            # 确保embedding是numpy数组
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # 归一化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                # 如果norm为0，使用随机向量
                embedding = np.random.normal(0, 0.1, len(embedding))
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Critical error in _get_query_embedding: {e}")
            logger.warning("Using fallback random embeddings")
            embedding = np.random.normal(0, 0.1, 128)
            return embedding / np.linalg.norm(embedding)

    async def _get_chunk_data(self, chunk_node_id: str) -> Dict[str, Any]:
        """
        获取文档块数据
        """
        try:
            chunk_key = chunk_node_id.replace('CHUNK_', '')
            if hasattr(self, 'doc_chunk'):
                chunk_content = await self.doc_chunk.get_data_by_key(chunk_key)
                if chunk_content:
                    return {
                        'id': chunk_key,
                        'content': chunk_content,
                        'type': 'chunk'
                    }
        except Exception as e:
            logger.warning(f"Failed to get chunk {chunk_node_id}: {e}")
        return None

    async def _get_entity_data(self, entity_node_id: str) -> Dict[str, Any]:
        """
        获取实体数据
        """
        try:
            entity_data = await self.graph.get_node(entity_node_id)
            if entity_data:
                return {
                    'entity_name': entity_data.get('entity_name', entity_node_id),
                    'entity_type': entity_data.get('entity_type', 'UNKNOWN'),
                    'description': entity_data.get('description', ''),
                    'type': 'entity'
                }
        except Exception as e:
            logger.warning(f"Failed to get entity {entity_node_id}: {e}")
        return None

    async def _get_relationships_between_nodes(self, node_ids: Set[str]) -> List[Dict[str, Any]]:
        """
        获取节点间的关系 - 只考虑entity和entity之间的关系，跳过涉及chunk的关系
        """
        relationships = []
        node_ids_list = list(node_ids)
        
        try:
            logger.debug(f"Getting entity-entity relationships for {len(node_ids_list)} nodes")
            
            for i, src_node in enumerate(node_ids_list):
                try:
                    # 跳过chunk节点和community节点，只处理entity节点
                    if src_node.startswith('CHUNK_') or src_node.startswith('COMMUNITY_'):
                        continue
                    
                    # 检查graph是否有get_node_edges方法
                    if not hasattr(self.graph, 'get_node_edges'):
                        logger.warning("Graph does not have get_node_edges method")
                        break
                    
                    # 获取该节点的所有边
                    node_edges = await self.graph.get_node_edges(src_node)
                    if node_edges:
                        for edge_tuple in node_edges:
                            try:
                                if len(edge_tuple) >= 2:
                                    tgt_node = edge_tuple[1] if edge_tuple[0] == src_node else edge_tuple[0]
                                    
                                    # 只有当目标节点也是entity（不是chunk或community）且在节点集合中时才处理 #TODO 
                                    # if (tgt_node in node_ids and 
                                    #     not tgt_node.startswith('CHUNK_') and 
                                    #     not tgt_node.startswith('COMMUNITY_')):
                                    if (  not tgt_node.startswith('CHUNK_') and # 只有当目标节点也是entity（不是chunk或community）
                                        not tgt_node.startswith('COMMUNITY_')):
                                        
                                        # 获取边数据
                                        edge_data = await self.graph.get_edge(src_node, tgt_node)
                                        if edge_data:
                                            relationships.append({
                                                'src_id': edge_data.get('src_id', src_node),
                                                'tgt_id': edge_data.get('tgt_id', tgt_node),
                                                'relation_name': edge_data.get('relation_name', 'CONNECTED'),
                                                'description': edge_data.get('description', ''),
                                                'type': 'relationship'
                                            })
                            except Exception as edge_error:
                                logger.warning(f"Error processing edge {edge_tuple}: {edge_error}")
                                continue
                except Exception as node_error:
                    logger.warning(f"Error getting edges for node {src_node}: {node_error}")
                    continue
                    
        except Exception as e:
            logger.error(f"Critical error in _get_relationships_between_nodes: {e}")
        
        logger.debug(f"Found {len(relationships)} entity-entity relationships (skipped chunk-related edges)")
        return relationships

    async def _compute_text_similarity(self, query_embedding: np.ndarray, text: str) -> float:
        """
        计算查询嵌入与文本的相似性
        """
        try:
            if not text.strip():
                return 0.0
            
            # 获取文本嵌入
            text_embedding = await self._get_query_embedding(text)
            
            # 计算余弦相似性
            similarity = np.dot(query_embedding, text_embedding)
            return float(similarity)
        except Exception as e:
            logger.warning(f"Failed to compute text similarity: {e}")
            return 0.0

    async def _fallback_retrieval(self, query: str, seed_entities: List[Dict]) -> Dict[str, Any]:
        """
        当层次化检索不可用时的回退方法
        """
        logger.info("🔄 Using fallback retrieval method...")
        
        results = {
            'entities': [],
            'chunks': [],
            'relationships': [],
            'communities': [],
            'community_summaries': []
        }
        
        try:
            # 尝试基础的实体检索
            if hasattr(self, 'entities_vdb') and seed_entities:
                entity_results = []
                for seed_entity in seed_entities[:self.config.top_k]:
                    entity_results.append({
                        'entity_name': seed_entity.get('entity_name', 'UNKNOWN'),
                        'entity_type': seed_entity.get('entity_type', 'UNKNOWN'),
                        'description': seed_entity.get('description', ''),
                        'type': 'entity'
                    })
                results['entities'] = entity_results
            
            # 尝试基础的文档块检索
            if hasattr(self, 'doc_chunk'):
                # 获取一些示例文档块
                try:
                    sample_chunks = []
                    for i in range(min(self.config.top_k, 5)):
                        chunk_data = await self.doc_chunk.get_data_by_index(i)
                        if chunk_data:
                            sample_chunks.append({
                                'id': str(i),
                                'content': chunk_data,
                                'type': 'chunk'
                            })
                    results['chunks'] = sample_chunks
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Fallback retrieval failed: {e}")
        
        return results

    async def _seed_entity_based_retrieval(self, query: str, seed_entities: List[Dict]) -> Dict[str, Any]:
        """
        基于种子实体的检索
        """
        # 这里可以实现基于种子实体的传统检索逻辑
        # 作为层次化检索的补充
        return await self._fallback_retrieval(query, seed_entities)

    ### 
    @register_retriever_method(type="hk_tree", method_name="hk_tree_flat_search")
    async def _hk_tree_flat_search_retrieval(self, query: str, seed_entities: List[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        扁平化检索：直接在所有层次的所有节点中寻找top_k个最相关的节点
        不考虑层次结构，将所有节点（基础节点+社区节点）扁平化处理
        """
        #logger.info("🔍 Starting flat search across all hierarchy levels...")
        
        # 检查是否支持层次结构
        if not hasattr(self.graph, 'get_hierarchy_info'):
            logger.warning("Graph does not support hierarchy structure, falling back")
            return await self._fallback_retrieval(query, seed_entities)
        
        hierarchy_info = await self.graph.get_hierarchy_info()
        if not hierarchy_info or hierarchy_info.get('levels', 0) == 0:
            logger.warning("No hierarchy found, falling back")
            return await self._fallback_retrieval(query, seed_entities)
        
        try:
            # Step 1: 获取查询嵌入
            query_embedding = await self._get_query_embedding(query)
            
            # Step 2: 使用FAISS检索最相似的节点
            top_k = getattr(self.config, 'top_k', 5)
            extended_top_k = min(top_k * 200, 1000)  
            
            top_nodes = await self._faiss_search_top_nodes(query_embedding, top_k)
            #top_nodes = []
            
            if not top_nodes:
                logger.warning("No nodes found from FAISS search, falling back to manual search")
                # fallback
                all_nodes_with_scores = await self._collect_all_nodes_with_similarity(query_embedding, hierarchy_info)
                if not all_nodes_with_scores:
                    return await self._fallback_retrieval(query, seed_entities)
                sorted_nodes = sorted(all_nodes_with_scores, key=lambda x: x['similarity_score'], reverse=True)
                top_nodes = sorted_nodes[:extended_top_k]
            
            #logger.info(f"📊 From {len(all_nodes_with_scores)} total nodes, selected top {len(top_nodes)} for processing")
            
            # Step 4: 分类并构建最终结果
            results = await self._build_flat_search_results(top_nodes, query_embedding, top_k)
            
            # logger.info(f"🎯 Flat search completed: {len(results.get('communities', []))} communities, "
            #            f"{len(results.get('entities', []))} entities, {len(results.get('chunks', []))} chunks")
            
            return results
            
        except Exception as e:
            logger.error(f"Flat search failed: {e}")
            return await self._fallback_retrieval(query, seed_entities)
    
    async def _collect_all_nodes_with_similarity(self, query_embedding: np.ndarray, hierarchy_info: Dict) -> List[Dict]:
        """
        收集所有层次的所有节点并计算相似性分数
        """
        all_nodes_with_scores = []
        
        # 1. 收集所有基础节点（entities和chunks）
        base_nodes = []
        if hasattr(self.graph, 'node_embeddings'):
            for node_id in self.graph.node_embeddings.keys():
                if not node_id.startswith('COMMUNITY_'):
                    base_nodes.append(node_id)
        
        #logger.info(f"📋 Found {len(base_nodes)} base nodes (entities + chunks)")
        
        # 计算基础节点的相似性
        for node_id in base_nodes:
            try:
                similarity_score = await self._compute_node_similarity(query_embedding, node_id)
                
                node_info = {
                    'id': node_id,
                    'type': 'chunk' if node_id.startswith('CHUNK_') else 'entity',
                    'level': 0,  # 基础层
                    'similarity_score': similarity_score
                }
                all_nodes_with_scores.append(node_info)
                
            except Exception as e:
                logger.warning(f"Failed to compute similarity for base node {node_id}: {e}")
                continue
        
        # 2. 收集所有社区节点
        hierarchy_levels = hierarchy_info.get('hierarchy_levels', {})
        community_summaries = hierarchy_info.get('community_summaries', {})
        
        community_count = 0
        for level, communities in hierarchy_levels.items():
            for community in communities:
                community_id = community['id']
                try:
                    # 计算社区相似性
                    similarity_score = await self._compute_community_similarity(
                        query_embedding, community, community_summaries.get(community_id, '')
                    )
                    
                    community_info = {
                        'id': community_id,
                        'type': 'community',
                        'level': level,
                        'similarity_score': similarity_score,
                        'member_count': len(community.get('nodes', [])),
                        'summary': community_summaries.get(community_id, '')
                    }
                    all_nodes_with_scores.append(community_info)
                    community_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to compute similarity for community {community_id}: {e}")
                    continue
        
        #logger.info(f"📋 Found {community_count} community nodes across all levels")
        #logger.info(f"📊 Total nodes for flat search: {len(all_nodes_with_scores)}")
        
        return all_nodes_with_scores
    
    async def _compute_node_similarity(self, query_embedding: np.ndarray, node_id: str) -> float:
        """
        计算基础节点（entity或chunk）与查询的相似性
        优先使用已保存的 node_text_embeddings，避免重复计算
        """
        try:
            # 优先使用已保存的文本嵌入（这是最常见和高效的方式）
            if (hasattr(self.graph, 'node_text_embeddings') and 
                node_id in self.graph.node_text_embeddings):
                node_embedding = self.graph.node_text_embeddings[node_id]
                if isinstance(node_embedding, np.ndarray):
                    similarity = await self.compute_similarity(query_embedding, node_embedding, "cosine")
                    return float(similarity)
            
            # 回退方案：基于文本内容重新计算（应该很少触发）
            logger.debug(f"⚠️ No cached embedding for {node_id}, recalculating...")
            
            if node_id.startswith('CHUNK_'):
                # Chunk节点：使用chunk内容
                chunk_key = node_id.replace('CHUNK_', '')
                chunk_content = await self._get_chunk_content_for_similarity(chunk_key)
                if chunk_content:
                    content_embedding = await self._get_query_embedding(chunk_content)
                    similarity = np.dot(query_embedding, content_embedding)
                    return float(similarity)
            else:
                # Entity节点：使用实体名称和描述
                entity_data = await self.graph.get_node(node_id)
                if entity_data:
                    entity_text = f"{entity_data.get('entity_name', '')} {entity_data.get('description', '')}"
                    if entity_text.strip():
                        entity_embedding = await self._get_query_embedding(entity_text)
                        similarity = np.dot(query_embedding, entity_embedding)
                        return float(similarity)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Failed to compute similarity for node {node_id}: {e}")
            return 0.0

    async def compute_similarity(self, query_embedding: np.ndarray, 
                       node_embedding: np.ndarray, 
                       method: str = "cosine") -> float:
        """
        计算 query 和 node 的相似度
        
        参数:
            query_embedding: np.ndarray, 查询向量
            node_embedding: np.ndarray, 节点向量
            method: str, "cosine" 或 "l2"
        
        返回:
            similarity: float, 相似度分数
        """
        if not isinstance(query_embedding, np.ndarray) or not isinstance(node_embedding, np.ndarray):
            raise ValueError("Both embeddings must be numpy arrays.")

        if method == "cosine":
            # 归一化后点积 = 余弦相似度
            q_norm = query_embedding / np.linalg.norm(query_embedding)
            n_norm = node_embedding / np.linalg.norm(node_embedding)
            similarity = np.dot(q_norm, n_norm)
            return float(similarity)

        elif method == "l2":
            # L2 距离 -> 相似度 (值越大越相似)
            distance = np.linalg.norm(query_embedding - node_embedding)
            similarity = 1 / (1 + distance)
            return float(similarity)

        else:
            raise ValueError(f"Unknown method '{method}'. Use 'cosine' or 'l2'.")
    
    async def _get_chunk_content_for_similarity(self, chunk_key: str) -> str:
        """
        获取chunk内容用于相似性计算
        """
        try:
            if hasattr(self, 'doc_chunk') and self.doc_chunk is not None:
                if hasattr(self.doc_chunk, 'get_data_by_key'):
                    if asyncio.iscoroutinefunction(self.doc_chunk.get_data_by_key):
                        return await self.doc_chunk.get_data_by_key(chunk_key)
                    else:
                        return self.doc_chunk.get_data_by_key(chunk_key)
        except Exception as e:
            logger.warning(f"Failed to get chunk content for {chunk_key}: {e}")
        return ""
    
    async def _build_flat_search_results(self, top_nodes: List[Dict], query_embedding: np.ndarray, top_k: int) -> Dict[str, Any]:
        """
        从扁平化搜索的top节点构建最终结果
        """
        results = {
            'communities': [],
            'entities': [],
            'chunks': [],
            'relationships': [],
            'community_summaries': []
        }
        
        # 分类处理top节点
        selected_communities = []
        selected_entities = []  # 来自top_nodes的实体，受temp_entity_limit限制
        selected_chunks = []
        
        # 按类型和相似性分配节点 #TODO
        # community + chunk = top_k, entity = top_k, relationships = top_k
        max_community_limit = max(1, top_k // 5)  # 最大社区数量为1、1/5 
        community_limit = max_community_limit #TODO debug
        chunk_limit = top_k - max_community_limit  # 剩余全部给chunks
        entity_limit = top_k  # 实体数量为top_k
        temp_entity_limit = entity_limit * 3 #TODO relation的entity
        relationship_limit = top_k  # 关系数量为top_k
        
        for node in top_nodes:
            node_type = node['type']
            node_id = node['id']
            # Old method 1 community and 4 chunks
            # if node_type == 'community' and len(selected_communities) < community_limit:
            #     # 添加社区信息
            #     community_info = {
            #         'id': node_id,
            #         'level': node['level'],
            #         'member_count': node.get('member_count', 0),
            #         'summary': node.get('summary', ''),
            #         'similarity_score': node['similarity_score']
            #     }
            #     selected_communities.append(community_info)
            #     results['community_summaries'].append(community_info['summary'])
                
            #     # 收集社区成员节点
            #     if hasattr(self.graph, 'get_community_children'):
            #         try:
            #             children = await self.graph.get_community_children(node_id)
            #             all_member_nodes.update(children)
            #         except Exception as e:
            #             logger.warning(f"Failed to get children for community {node_id}: {e}")
                        
            # elif node_type == 'entity' and len(selected_entities) < temp_entity_limit: #原本为entity_limit
            #     # 添加实体信息
            #     entity_data = await self._get_entity_data_with_similarity(node_id, query_embedding)
            #     if entity_data:
            #         selected_entities.append(entity_data)
            #         all_member_nodes.add(node_id)
                    
            # elif node_type == 'chunk' and len(selected_chunks) < chunk_limit:
            #     # 添加chunk信息
            #     chunk_data = await self._get_chunk_data_with_similarity(node_id, query_embedding)
            #     if chunk_data:
            #         selected_chunks.append(chunk_data)
            #         all_member_nodes.add(node_id)

            # New method Free Community and chunks
                        
            if node_type == 'entity' and len(selected_entities) < temp_entity_limit: #原本为entity_limit
                # 添加实体信息
                entity_data = await self._get_entity_data_with_similarity(node_id, query_embedding)
                if entity_data:
                    selected_entities.append(entity_data)

            elif (node_type == 'community' or node_type == 'chunk') and len(selected_communities) + len(selected_chunks)< top_k:
                # 添加社区信息
                if node_type == 'community' and len(selected_communities) < max_community_limit:
                    community_info = {
                        'id': node_id,
                        'level': node['level'],
                        'member_count': node.get('member_count', 0),
                        'summary': node.get('summary', ''),
                        'similarity_score': node['similarity_score']
                    }
                    selected_communities.append(community_info)
                    results['community_summaries'].append(community_info['summary'])
                    
                elif node_type == 'chunk' and len(selected_chunks) < top_k:
                    # 添加chunk信息
                    chunk_data = await self._get_chunk_data_with_similarity(node_id, query_embedding)
                    if chunk_data:
                        selected_chunks.append(chunk_data)
            
        # 继续收集entity直到达到entity_limit
        for node in top_nodes:
            if len(selected_entities) >= temp_entity_limit:
                break
                
            node_type = node['type']
            node_id = node['id']
            
            if node_type == 'entity':
                # 检查是否已经添加过这个entity
                already_added = any(e.get('entity_name') == node_id for e in selected_entities)
                
                if not already_added:
                    entity_data = await self._get_entity_data_with_similarity(node_id, query_embedding)
                    if entity_data:
                        selected_entities.append(entity_data)
        
        # 如果community + chunk总数不够top_k，优先补充chunk
        community_chunk_total = len(selected_communities) + len(selected_chunks)
        if community_chunk_total < top_k:
            remaining_slots = top_k - community_chunk_total
            for node in top_nodes:
                if remaining_slots <= 0:
                    break
                    
                node_type = node['type']
                node_id = node['id']
                
                if node_type == 'chunk':
                    # 检查是否已经添加过这个chunk
                    already_added = any(c.get('id') == node_id.replace('CHUNK_', '') for c in selected_chunks)
                    
                    if not already_added:
                        chunk_data = await self._get_chunk_data_with_similarity(node_id, query_embedding)
                        if chunk_data:
                            selected_chunks.append(chunk_data)
                            remaining_slots -= 1
        
        # 从选中的社区中提取实体（不受temp_entity_limit限制）
        community_entities = []
        if selected_communities:
            logger.debug(f"🏘️ Extracting entities from {len(selected_communities)} selected communities")
            
            for community_info in selected_communities:
                community_id = community_info['id']
                try:
                    # 获取社区的子节点
                    if hasattr(self.graph, 'get_community_children'):
                        children = await self.graph.get_community_children(community_id)
                        
                        for child_node_id in children:
                            # 只处理实体节点（跳过chunk和community节点）
                            if (not child_node_id.startswith('CHUNK_') and 
                                not child_node_id.startswith('COMMUNITY_')):
                                
                                # 检查是否已经在selected_entities中
                                already_selected = any(e.get('entity_name') == child_node_id for e in selected_entities)
                                
                                if not already_selected:
                                    entity_data = await self._get_entity_data_with_similarity(child_node_id, query_embedding)
                                    if entity_data:
                                        # 标记这是来自社区的实体
                                        entity_data['source'] = 'community_member'
                                        community_entities.append(entity_data)
                                        
                except Exception as e:
                    logger.warning(f"Failed to extract entities from community {community_id}: {e}")
            
            logger.debug(f"✅ Extracted {len(community_entities)} entities from communities")
        
        # 合并两种来源的实体用于关系查找（去重）
        all_entities_for_relations = selected_entities.copy()  # 来自top_nodes的实体
        #community_entities = [] #TODO ablation
        # 添加来自社区的实体，去重
        for community_entity in community_entities:
            entity_name = community_entity.get('entity_name', '')
            already_exists = any(e.get('entity_name') == entity_name for e in all_entities_for_relations)
            if not already_exists:
                all_entities_for_relations.append(community_entity)
        
        # 按相似性排序结果
        selected_entities.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        community_entities.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        all_entities_for_relations.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        selected_chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        selected_communities.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # 填充结果 - 按照用户要求的数量分配
        # community + chunk = top_k, entity = top_k, relationships = top_k
        results['communities'] = selected_communities
        results['entities'] = selected_entities[:entity_limit]  # 只输出来自top_nodes的实体，限制为entity_limit
        results['chunks'] = selected_chunks

        # 用于关系计算的实体（包含两种来源的实体）
        all_entities_for_relations = all_entities_for_relations[:entity_limit*5] #TODO debug
        selected_entity_names = [d['entity_name'] for d in all_entities_for_relations]
        relationships = await self._get_relationships_between_nodes(selected_entity_names) #从合并后的实体中获取关系
        selected_relationships = await self._get_select_relationships_from_relationships(relationships, query_embedding, top_k)
        results['relationships'] = selected_relationships[:relationship_limit]

        
        # 确保 community + chunk 的总数不超过 top_k
        total_community_chunk = len(results['communities']) + len(results['chunks'])
        if total_community_chunk > top_k:
            # 如果超过了，优先保留community，然后调整chunk数量
            max_chunks = top_k - len(results['communities'])
            results['chunks'] = results['chunks'][:max_chunks]
        
        logger.info(f"📋 Flat search results: {len(results['communities'])} communities, "
                   f"{len(results['entities'])} entities (from top_nodes), {len(results['chunks'])} chunks, "
                   f"{len(results['relationships'])} relationships")
        logger.info(f"🔗 Relation calculation used {len(all_entities_for_relations)} total entities "
                   f"({len(selected_entities)} from top_nodes + {len(community_entities)} from communities)")
        # logger.info(f"📊 Distribution: communities+chunks={len(results['communities'])+len(results['chunks'])}/{top_k}, "
        #            f"entities={len(results['entities'])}/{top_k}, relationships={len(results['relationships'])}/{top_k}")
        
        return results

    async def _get_select_relationships_from_relationships(self, relationships: List[Dict], query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """
        从关系中选择top_k个最相关的关系
        
        Args:
            relationships: 候选关系列表
            query_embedding: 查询的embedding向量
            top_k: 选择的关系数量
            
        Returns:
            按相似性排序的top_k个关系列表
        """
        if not relationships:
            return []
        
        # 计算每个关系与查询的相似性分数
        relationship_scores = []
        
        for relationship in relationships:
            try:
                # 构建关系的文本表示：组合src_id, relation_name, tgt_id和description
                relation_text_parts = []
                
                # 添加源实体和目标实体
                src_id = relationship.get('src_id', '')
                tgt_id = relationship.get('tgt_id', '')
                relation_name = relationship.get('relation_name', '')
                description = relationship.get('description', '')
                
                # 构建关系文本
                if src_id and relation_name and tgt_id:
                    relation_text_parts.append(f"{src_id} {relation_name} {tgt_id}")
                
                # 添加描述信息
                if description and description.strip():
                    relation_text_parts.append(description.strip())
                
                # 如果没有有效的文本信息，使用默认值
                if not relation_text_parts:
                    relation_text_parts.append(f"{src_id} connected to {tgt_id}")
                
                # 组合所有文本
                relation_text = ". ".join(relation_text_parts)
                
                # 计算关系文本的embedding
                relation_embedding = await self._get_query_embedding(relation_text)
                
                # 计算与查询的余弦相似性
                similarity_score = np.dot(query_embedding, relation_embedding) #TODO 关系相似度选择
                
                # 创建带分数的关系副本
                scored_relationship = relationship.copy()
                scored_relationship['similarity_score'] = float(similarity_score)
                scored_relationship['relation_text'] = relation_text  # 保存用于调试
                
                relationship_scores.append(scored_relationship)
                
            except Exception as e:
                logger.warning(f"Failed to compute similarity for relationship {relationship}: {e}")
                # 如果计算失败，赋予低分数但仍保留
                fallback_relationship = relationship.copy()
                fallback_relationship['similarity_score'] = 0.0
                relationship_scores.append(fallback_relationship)
        
        # 按相似性分数降序排序
        relationship_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # 去重：用 (src_id, relation_name, tgt_id) 作为唯一键
        seen = set()
        unique_relationships = []
        for rel in relationship_scores:
            key = (rel.get('src_id', ''), rel.get('relation_name', ''), rel.get('tgt_id', ''))
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        # 选择top_k个关系
        selected_relationships = unique_relationships[:top_k]
        
        logger.debug(f"🔗 Selected {len(selected_relationships)} relationships from {len(relationships)} candidates")
        
        # 可选：记录top关系的信息（用于调试）
        # for i, rel in enumerate(selected_relationships[:3]):  # 只记录前3个
        #     logger.debug(f"   Top {i+1}: {rel.get('src_id', '')} -> {rel.get('tgt_id', '')} "
        #                  f"(relation: {rel.get('relation_name', '')}, score: {rel.get('similarity_score', 0):.3f})")
        
        return selected_relationships

    async def _faiss_search_top_nodes(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """
        使用FAISS检索最相似的节点，每种类型都检索top_k个
        
        Args:
            query_embedding: 查询的embedding向量
            top_k: 每种类型返回的节点数量
            
        Returns:
            按相似性排序的节点列表
        """
        try:
            # 检查graph是否有专用的FAISS搜索功能
            if not hasattr(self.graph, 'search_similar_entities'):
                logger.warning("Graph does not support specialized FAISS search methods")
                return []
            
            logger.info(f"🔍 Using FAISS to search for top {top_k} nodes of each type (chunk, entity, community)")
            
            all_faiss_results = []
            
            # 分别从三种专用索引中搜索，每种类型都搜索top_k个
            node_types = ['chunk', 'entity', 'community']
            
            for node_type in node_types:
                logger.debug(f"🔍 Searching {node_type} index for top {top_k} nodes")
                
                try:
                    # 使用专用的搜索方法
                    if node_type == 'chunk':
                        current_results = await self.graph.search_similar_chunks(
                            query_embedding=query_embedding,
                            top_k=top_k 
                        )
                    elif node_type == 'entity':
                        current_results = await self.graph.search_similar_entities(
                            query_embedding=query_embedding,
                            top_k=top_k*5
                        )
                    elif node_type == 'community':
                        current_results = await self.graph.search_similar_communities(
                            query_embedding=query_embedding,
                            top_k=top_k 
                        )
                    else:
                        continue
                    
                    if current_results:
                        all_faiss_results.extend(current_results)
                        logger.debug(f"✅ Found {len(current_results)} {node_type} nodes")
                    else:
                        logger.debug(f"⚠️ No results found for {node_type}")
                        
                except Exception as e:
                    logger.warning(f"Error searching {node_type} index: {e}")
                    continue
            
            if not all_faiss_results:
                logger.warning("FAISS search returned no results from any index")
                return []
            
            # 按相似度分数排序
            all_faiss_results.sort(key=lambda x: x[1], reverse=True)
            faiss_results = all_faiss_results

            if not faiss_results:
                logger.warning("FAISS search returned no results")
                return []
            
            # 获取节点详细信息
            node_ids = [node_id for node_id, score in faiss_results]
            node_info_dict = await self.graph.get_node_info_by_ids(node_ids)
            
            # 转换为与原格式兼容的结构
            top_nodes = []
            for node_id, similarity_score in faiss_results:
                node_info = node_info_dict.get(node_id, {})
                node_type = node_info.get('node_type', 'unknown')
                
                # 转换节点类型名称以保持兼容性
                if node_type == 'chunk':
                    formatted_type = 'chunk'
                elif node_type == 'entity':
                    formatted_type = 'entity'
                elif node_type == 'community':
                    formatted_type = 'community'
                else:
                    formatted_type = 'entity'  # 默认为entity
                
                # 构建节点信息
                node_data = {
                    'id': node_id,
                    'type': formatted_type,
                    'similarity_score': float(similarity_score)
                }
                
                # 添加额外信息
                if formatted_type == 'community':
                    node_data['level'] = node_info.get('level', 0)
                    node_data['member_count'] = len(node_info.get('children', []))
                    node_data['summary'] = node_info.get('content', '')
                elif formatted_type == 'chunk':
                    node_data['content'] = node_info.get('content', '')
                elif formatted_type == 'entity':
                    node_data['entity_name'] = node_info.get('entity_name', node_id)
                    node_data['description'] = node_info.get('description', '')
                
                top_nodes.append(node_data)
            
            chunk_count = sum(1 for n in top_nodes if n['type'] == 'chunk')
            entity_count = sum(1 for n in top_nodes if n['type'] == 'entity')  
            community_count = sum(1 for n in top_nodes if n['type'] == 'community')
            
            logger.info(f"✅ FAISS multi-index search found {len(top_nodes)} nodes (top {top_k} of each type): "
                       f"{chunk_count} chunks, {entity_count} entities, {community_count} communities")
            
            return top_nodes
            
        except Exception as e:
            logger.error(f"❌ FAISS search failed: {e}")
            logger.exception(e)
            return []