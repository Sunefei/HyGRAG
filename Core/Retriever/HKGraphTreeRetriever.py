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
    HKGraphTree specialized retriever: top-down retrieval of hierarchical communities, entities, chunks, and relations.

    Core approach:
    1. Start from top-level communities, filter relevant ones using query similarity
    2. Expand downward to retrieve sub-communities and member nodes
    3. Collect relevant entities, chunks, and relations at the base level
    4. Integrate multi-level information to form final retrieval results
    """

    def __init__(self, **kwargs):
        config = kwargs.pop("config")
        super().__init__(config)
        self.mode_list = ["hk_tree_search", "hk_tree_top_down", "hk_tree_comprehensive", "hk_tree_true_search", "hk_tree_flat_search"]
        self.type = "hk_tree"

        # Set required attributes
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
        Compute similarity between query and community.
        Prefer cached node_text_embeddings to avoid redundant computation.

        Args:
            query_embedding: Query embedding vector
            community: Community info dict
            summary: Community summary text

        Returns:
            Similarity score
        """
        try:
            community_id = community['id']
            
            # Method 1: Use precomputed text embedding (most common and efficient)
            if (hasattr(self.graph, 'node_text_embeddings') and 
                community_id in self.graph.node_text_embeddings):
                community_embedding = self.graph.node_text_embeddings[community_id]
                if isinstance(community_embedding, np.ndarray):
                    similarity = await self.compute_similarity(query_embedding, community_embedding, "cosine")
                    return float(similarity)
            
            
            
            # Fallback: recalculate embedding from community summary (rarely triggered)
            if summary and summary.strip():
                logger.debug(f"No cached embedding for community {community_id}, recalculating from summary...")
                summary_embedding = await self._get_query_embedding(summary)
                similarity = np.dot(query_embedding, summary_embedding)
                return float(similarity)

            # Fallback: return low score if no summary available
            logger.warning(f"No embedding or summary for community {community_id}")
            return 0.1
            
        except Exception as e:
            logger.warning(f"Failed to compute similarity for community {community.get('id', 'UNKNOWN')}: {e}")
            return 0.0

    async def _extract_entities_and_chunks_from_nodes(self, member_nodes: Set[str], 
                                                    query_embedding: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract and sort entities and chunks from member nodes.

        Args:
            member_nodes: Set of member node IDs
            query_embedding: Query embedding vector

        Returns:
            (entity list, chunk list)
        """
        entities = []
        chunks = []
        
        for node_id in member_nodes:
            if node_id.startswith('CHUNK_'):
                # Process chunk nodes
                chunk_data = await self._get_chunk_data_with_similarity(node_id, query_embedding)
                if chunk_data:
                    chunks.append(chunk_data)
            elif not node_id.startswith('COMMUNITY_'):
                # Process entity nodes (skip community nodes)
                entity_data = await self._get_entity_data_with_similarity(node_id, query_embedding)
                if entity_data:
                    entities.append(entity_data)

        # Sort by similarity score
        entities.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        logger.debug(f"📋 Extracted and sorted {len(entities)} entities and {len(chunks)} chunks")
        return entities, chunks

    async def _get_chunk_data_with_similarity(self, chunk_node_id: str, query_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Get chunk data and compute similarity score.
        Prefer cached node_text_embeddings to avoid redundant computation.
        """
        try:
            chunk_key = chunk_node_id.replace('CHUNK_', '')
            chunk_content = None

            if hasattr(self, 'doc_chunk') and self.doc_chunk is not None:
                # Check if get_data_by_key is async
                try:
                    if hasattr(self.doc_chunk, 'get_data_by_key'):
                        # Try async call
                        if asyncio.iscoroutinefunction(self.doc_chunk.get_data_by_key):
                            chunk_content = await self.doc_chunk.get_data_by_key(chunk_key)
                        else:
                            # Sync call
                            chunk_content = self.doc_chunk.get_data_by_key(chunk_key)
                    else:
                        logger.warning(f"doc_chunk does not have get_data_by_key method")
                        return None
                    
                    if chunk_content:
                        # Prefer cached text embeddings
                        if (hasattr(self.graph, 'node_text_embeddings') and
                            chunk_node_id in self.graph.node_text_embeddings):
                            chunk_embedding = self.graph.node_text_embeddings[chunk_node_id]
                            if isinstance(chunk_embedding, np.ndarray):
                                similarity_score = await self.compute_similarity(query_embedding, chunk_embedding, "cosine")
                            else:
                                similarity_score = 0.0
                        else:
                            # Fallback: recalculate embedding
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
        Get entity data and compute similarity score.
        Prefer cached node_text_embeddings to avoid redundant computation.
        """
        try:
            entity_data = await self.graph.get_node(entity_node_id)
            if entity_data:
                # Prefer cached text embeddings
                if (hasattr(self.graph, 'node_text_embeddings') and
                    entity_node_id in self.graph.node_text_embeddings):
                    entity_embedding = self.graph.node_text_embeddings[entity_node_id]
                    if isinstance(entity_embedding, np.ndarray):
                        similarity_score = await self.compute_similarity(query_embedding, entity_embedding, "cosine")
                    else:
                        similarity_score = 0.0
                else:
                    # Fallback: recalculate embedding (based on name and description)
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
        Get query embedding vector
        """
        try:
            embedding = None

            # Try using HKGraphTree's embedding method
            if hasattr(self.graph, '_embed_text'):
                try:
                    embedding = await self.graph._embed_text(query)
                    #logger.debug(f"Used graph._embed_text for query embedding")
                except Exception as e:
                    logger.warning(f"Failed to use graph._embed_text: {e}")

            # Fallback: use embedding_model directly
            if embedding is None and hasattr(self.graph, 'embedding_model'):
                try:
                    embedding = self.graph.embedding_model._get_text_embedding(query)
                    logger.debug(f"Used graph.embedding_model for query embedding")
                except Exception as e:
                    logger.warning(f"Failed to use graph.embedding_model: {e}")

            # Try using entities_vdb embedding model
            if embedding is None and hasattr(self, 'entities_vdb') and hasattr(self.entities_vdb, 'embedding_model'):
                try:
                    embedding = self.entities_vdb.embedding_model._get_text_embedding(query)
                    logger.debug(f"Used entities_vdb.embedding_model for query embedding")
                except Exception as e:
                    logger.warning(f"Failed to use entities_vdb.embedding_model: {e}")

            # Last resort: random embedding
            if embedding is None:
                logger.warning("No embedding model found, using random embeddings")
                embedding = np.random.normal(0, 0.1, 128)

            # Ensure embedding is a numpy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                # If norm is 0, use random vector
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
        Get chunk data
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
        Get entity data
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
        Get inter-node relationships - only entity-entity (skip chunk-related).
        """
        relationships = []
        node_ids_list = list(node_ids)

        try:
            logger.debug(f"Getting entity-entity relationships for {len(node_ids_list)} nodes")

            for i, src_node in enumerate(node_ids_list):
                try:
                    # Skip chunk and community nodes, only process entity nodes
                    if src_node.startswith('CHUNK_') or src_node.startswith('COMMUNITY_'):
                        continue

                    # Check if graph has get_node_edges method
                    if not hasattr(self.graph, 'get_node_edges'):
                        logger.warning("Graph does not have get_node_edges method")
                        break

                    # Get all edges for this node
                    node_edges = await self.graph.get_node_edges(src_node)
                    if node_edges:
                        for edge_tuple in node_edges:
                            try:
                                if len(edge_tuple) >= 2:
                                    tgt_node = edge_tuple[1] if edge_tuple[0] == src_node else edge_tuple[0]

                                    # Only process when target is also entity (not chunk or community) # TODO
                                    # if (tgt_node in node_ids and
                                    #     not tgt_node.startswith('CHUNK_') and
                                    #     not tgt_node.startswith('COMMUNITY_')):
                                    if (  not tgt_node.startswith('CHUNK_') and # Only when target is also entity (not chunk or community)
                                        not tgt_node.startswith('COMMUNITY_')):

                                        # Get edge data
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
        Compute similarity between query embedding and text
        """
        try:
            if not text.strip():
                return 0.0

            # Get text embedding
            text_embedding = await self._get_query_embedding(text)

            # Compute cosine similarity
            similarity = np.dot(query_embedding, text_embedding)
            return float(similarity)
        except Exception as e:
            logger.warning(f"Failed to compute text similarity: {e}")
            return 0.0

    async def _fallback_retrieval(self, query: str, seed_entities: List[Dict]) -> Dict[str, Any]:
        """
        Fallback method when hierarchical retrieval is unavailable
        """
        logger.info("Using fallback retrieval method...")
        
        results = {
            'entities': [],
            'chunks': [],
            'relationships': [],
            'communities': [],
            'community_summaries': []
        }
        
        try:
            # Try basic entity retrieval
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

            # Try basic chunk retrieval
            if hasattr(self, 'doc_chunk'):
                # Get some sample chunks
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
        Seed entity based retrieval
        """
        # Traditional retrieval logic based on seed entities
        # As a supplement to hierarchical retrieval
        return await self._fallback_retrieval(query, seed_entities)

    ### 
    @register_retriever_method(type="hk_tree", method_name="hk_tree_flat_search")
    async def _hk_tree_flat_search_retrieval(self, query: str, seed_entities: List[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Flat search: find top_k most relevant nodes across all hierarchy levels.
        Treats all nodes (base + community) as a flat collection regardless of hierarchy.
        """
        #logger.info("Starting flat search across all hierarchy levels...")

        # Check if hierarchy is supported
        if not hasattr(self.graph, 'get_hierarchy_info'):
            logger.warning("Graph does not support hierarchy structure, falling back")
            return await self._fallback_retrieval(query, seed_entities)
        
        hierarchy_info = await self.graph.get_hierarchy_info()
        if not hierarchy_info or hierarchy_info.get('levels', 0) == 0:
            logger.warning("No hierarchy found, falling back")
            return await self._fallback_retrieval(query, seed_entities)
        
        try:
            # Step 1: Get query embedding
            query_embedding = await self._get_query_embedding(query)

            # Step 2: Use FAISS to retrieve most similar nodes
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
            
            # Step 4: Classify and build final results
            results = await self._build_flat_search_results(top_nodes, query_embedding, top_k)
            
            # logger.info(f"🎯 Flat search completed: {len(results.get('communities', []))} communities, "
            #            f"{len(results.get('entities', []))} entities, {len(results.get('chunks', []))} chunks")
            
            return results
            
        except Exception as e:
            logger.error(f"Flat search failed: {e}")
            return await self._fallback_retrieval(query, seed_entities)
    
    async def _collect_all_nodes_with_similarity(self, query_embedding: np.ndarray, hierarchy_info: Dict) -> List[Dict]:
        """
        Collect all nodes across all levels and compute similarity scores
        """
        all_nodes_with_scores = []

        # 1. Collect all base nodes (entities and chunks)
        base_nodes = []
        if hasattr(self.graph, 'node_embeddings'):
            for node_id in self.graph.node_embeddings.keys():
                if not node_id.startswith('COMMUNITY_'):
                    base_nodes.append(node_id)
        
        #logger.info(f"📋 Found {len(base_nodes)} base nodes (entities + chunks)")
        
        # Compute similarity for base nodes
        for node_id in base_nodes:
            try:
                similarity_score = await self._compute_node_similarity(query_embedding, node_id)
                
                node_info = {
                    'id': node_id,
                    'type': 'chunk' if node_id.startswith('CHUNK_') else 'entity',
                    'level': 0,  # Base layer
                    'similarity_score': similarity_score
                }
                all_nodes_with_scores.append(node_info)
                
            except Exception as e:
                logger.warning(f"Failed to compute similarity for base node {node_id}: {e}")
                continue
        
        # 2. Collect all community nodes
        hierarchy_levels = hierarchy_info.get('hierarchy_levels', {})
        community_summaries = hierarchy_info.get('community_summaries', {})
        
        community_count = 0
        for level, communities in hierarchy_levels.items():
            for community in communities:
                community_id = community['id']
                try:
                    # Compute community similarity
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
        Compute similarity between base node (entity or chunk) and query
        Prefer saved node_text_embeddings to avoid recomputation
        """
        try:
            # Prefer saved text embeddings (most common and efficient)
            if (hasattr(self.graph, 'node_text_embeddings') and 
                node_id in self.graph.node_text_embeddings):
                node_embedding = self.graph.node_text_embeddings[node_id]
                if isinstance(node_embedding, np.ndarray):
                    similarity = await self.compute_similarity(query_embedding, node_embedding, "cosine")
                    return float(similarity)
            
            # Fallback: recompute from text content (rarely triggered)
            logger.debug(f"⚠️ No cached embedding for {node_id}, recalculating...")
            
            if node_id.startswith('CHUNK_'):
                # Chunk node: use chunk content
                chunk_key = node_id.replace('CHUNK_', '')
                chunk_content = await self._get_chunk_content_for_similarity(chunk_key)
                if chunk_content:
                    content_embedding = await self._get_query_embedding(chunk_content)
                    similarity = np.dot(query_embedding, content_embedding)
                    return float(similarity)
            else:
                # Entity node: use entity name and description
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
        Compute similarity between query and node
        
        Args:
            query_embedding: np.ndarray, query vector
            node_embedding: np.ndarray, node vector
            method: str, "cosine" or "l2"
        
        Returns:
            similarity: float, similarity score
        """
        if not isinstance(query_embedding, np.ndarray) or not isinstance(node_embedding, np.ndarray):
            raise ValueError("Both embeddings must be numpy arrays.")

        if method == "cosine":
            # Normalized dot product = cosine similarity
            q_norm = query_embedding / np.linalg.norm(query_embedding)
            n_norm = node_embedding / np.linalg.norm(node_embedding)
            similarity = np.dot(q_norm, n_norm)
            return float(similarity)

        elif method == "l2":
            # L2 distance -> similarity (higher = more similar)
            distance = np.linalg.norm(query_embedding - node_embedding)
            similarity = 1 / (1 + distance)
            return float(similarity)

        else:
            raise ValueError(f"Unknown method '{method}'. Use 'cosine' or 'l2'.")
    
    async def _get_chunk_content_for_similarity(self, chunk_key: str) -> str:
        """
        Get chunk content for similarity computation
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
        Build final results from flattened search top nodes
        """
        results = {
            'communities': [],
            'entities': [],
            'chunks': [],
            'relationships': [],
            'community_summaries': []
        }
        
        # Classify top nodes
        selected_communities = []
        selected_entities = []  # Entities from top_nodes, limited by temp_entity_limit
        selected_chunks = []
        
        # Assign nodes by type and similarity #TODO
        # community + chunk = top_k, entity = top_k, relationships = top_k
        max_community_limit = max(1, top_k // 5)  # Max community count 
        community_limit = max_community_limit #TODO debug
        chunk_limit = top_k - max_community_limit  # Remaining for chunks
        entity_limit = top_k  # Entity count = top_k
        temp_entity_limit = entity_limit * 3  # TODO: entities for relation lookup
        relationship_limit = top_k  # Relationship count = top_k
        
        for node in top_nodes:
            node_type = node['type']
            node_id = node['id']
            # Old method 1 community and 4 chunks
            # if node_type == 'community' and len(selected_communities) < community_limit:
            #     # Add community info
            #     community_info = {
            #         'id': node_id,
            #         'level': node['level'],
            #         'member_count': node.get('member_count', 0),
            #         'summary': node.get('summary', ''),
            #         'similarity_score': node['similarity_score']
            #     }
            #     selected_communities.append(community_info)
            #     results['community_summaries'].append(community_info['summary'])
                
            #     # Collect community member nodes
            #     if hasattr(self.graph, 'get_community_children'):
            #         try:
            #             children = await self.graph.get_community_children(node_id)
            #             all_member_nodes.update(children)
            #         except Exception as e:
            #             logger.warning(f"Failed to get children for community {node_id}: {e}")
                        
            # elif node_type == 'entity' and len(selected_entities) < temp_entity_limit:
            #     # Add entity info
            #     entity_data = await self._get_entity_data_with_similarity(node_id, query_embedding)
            #     if entity_data:
            #         selected_entities.append(entity_data)
            #         all_member_nodes.add(node_id)
                    
            # elif node_type == 'chunk' and len(selected_chunks) < chunk_limit:
            #     # Add chunk info
            #     chunk_data = await self._get_chunk_data_with_similarity(node_id, query_embedding)
            #     if chunk_data:
            #         selected_chunks.append(chunk_data)
            #         all_member_nodes.add(node_id)

            # New method Free Community and chunks
                        
            if node_type == 'entity' and len(selected_entities) < temp_entity_limit:
                # Add entity info
                entity_data = await self._get_entity_data_with_similarity(node_id, query_embedding)
                if entity_data:
                    selected_entities.append(entity_data)

            elif (node_type == 'community' or node_type == 'chunk') and len(selected_communities) + len(selected_chunks)< top_k:
                # Add community info
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
                    # Add chunk info
                    chunk_data = await self._get_chunk_data_with_similarity(node_id, query_embedding)
                    if chunk_data:
                        selected_chunks.append(chunk_data)
            
        # Continue collecting entities until entity_limit is reached
        for node in top_nodes:
            if len(selected_entities) >= temp_entity_limit:
                break
                
            node_type = node['type']
            node_id = node['id']
            
            if node_type == 'entity':
                # Check if this entity was already added
                already_added = any(e.get('entity_name') == node_id for e in selected_entities)
                
                if not already_added:
                    entity_data = await self._get_entity_data_with_similarity(node_id, query_embedding)
                    if entity_data:
                        selected_entities.append(entity_data)
        
        # If community + chunk total < top_k, supplement with chunks
        community_chunk_total = len(selected_communities) + len(selected_chunks)
        if community_chunk_total < top_k:
            remaining_slots = top_k - community_chunk_total
            for node in top_nodes:
                if remaining_slots <= 0:
                    break
                    
                node_type = node['type']
                node_id = node['id']
                
                if node_type == 'chunk':
                    # Check if this chunk was already added
                    already_added = any(c.get('id') == node_id.replace('CHUNK_', '') for c in selected_chunks)
                    
                    if not already_added:
                        chunk_data = await self._get_chunk_data_with_similarity(node_id, query_embedding)
                        if chunk_data:
                            selected_chunks.append(chunk_data)
                            remaining_slots -= 1
        
        # Extract entities from selected communities (not limited by temp_entity_limit)
        community_entities = []
        if selected_communities:
            logger.debug(f"🏘️ Extracting entities from {len(selected_communities)} selected communities")
            
            for community_info in selected_communities:
                community_id = community_info['id']
                try:
                    # Get community child nodes
                    if hasattr(self.graph, 'get_community_children'):
                        children = await self.graph.get_community_children(community_id)
                        
                        for child_node_id in children:
                            # Only process entity nodes (skip chunks and communities)
                            if (not child_node_id.startswith('CHUNK_') and 
                                not child_node_id.startswith('COMMUNITY_')):
                                
                                # Check if already in selected_entities
                                already_selected = any(e.get('entity_name') == child_node_id for e in selected_entities)
                                
                                if not already_selected:
                                    entity_data = await self._get_entity_data_with_similarity(child_node_id, query_embedding)
                                    if entity_data:
                                        # Mark as entity from community
                                        entity_data['source'] = 'community_member'
                                        community_entities.append(entity_data)
                                        
                except Exception as e:
                    logger.warning(f"Failed to extract entities from community {community_id}: {e}")
            
            logger.debug(f"✅ Extracted {len(community_entities)} entities from communities")
        
        # Merge entities from both sources for relation lookup (dedup)
        all_entities_for_relations = selected_entities.copy()  # Entities from top_nodes
        #community_entities = [] #TODO ablation
        # Add entities from communities, deduplicate
        for community_entity in community_entities:
            entity_name = community_entity.get('entity_name', '')
            already_exists = any(e.get('entity_name') == entity_name for e in all_entities_for_relations)
            if not already_exists:
                all_entities_for_relations.append(community_entity)
        
        # Sort results by similarity
        selected_entities.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        community_entities.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        all_entities_for_relations.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        selected_chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        selected_communities.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Fill results - allocate as per user requirements
        # community + chunk = top_k, entity = top_k, relationships = top_k
        results['communities'] = selected_communities
        results['entities'] = selected_entities[:entity_limit]  # Only output entities from top_nodes, limited
        results['chunks'] = selected_chunks

        # Entities for relation computation (from both sources)
        all_entities_for_relations = all_entities_for_relations[:entity_limit*5] #TODO debug
        selected_entity_names = [d['entity_name'] for d in all_entities_for_relations]
        relationships = await self._get_relationships_between_nodes(selected_entity_names)  # Get relations from merged entities
        selected_relationships = await self._get_select_relationships_from_relationships(relationships, query_embedding, top_k)
        results['relationships'] = selected_relationships[:relationship_limit]

        
        # Ensure community + chunk total does not exceed top_k
        total_community_chunk = len(results['communities']) + len(results['chunks'])
        if total_community_chunk > top_k:
            # If exceeded, prioritize communities, then adjust chunk count
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
        Select top_k most relevant relationships
        
        Args:
            relationships: candidate relationship list
            query_embedding: query embedding vector
            top_k: number of relationships to select
            
        Returns:
            List of top_k relationships sorted by similarity
        """
        if not relationships:
            return []
        
        # Compute similarity score for each relationship with the query
        relationship_scores = []
        
        for relationship in relationships:
            try:
                # Build text representation: combine src_id, relation_name, tgt_id, and description
                relation_text_parts = []
                
                # Add source and target entities
                src_id = relationship.get('src_id', '')
                tgt_id = relationship.get('tgt_id', '')
                relation_name = relationship.get('relation_name', '')
                description = relationship.get('description', '')
                
                # Build relation text
                if src_id and relation_name and tgt_id:
                    relation_text_parts.append(f"{src_id} {relation_name} {tgt_id}")
                
                # Add description
                if description and description.strip():
                    relation_text_parts.append(description.strip())
                
                # Use defaults if no valid text
                if not relation_text_parts:
                    relation_text_parts.append(f"{src_id} connected to {tgt_id}")
                
                # Combine all text
                relation_text = ". ".join(relation_text_parts)
                
                # Compute embedding for relation text
                relation_embedding = await self._get_query_embedding(relation_text)
                
                # Compute cosine similarity with query
                similarity_score = np.dot(query_embedding, relation_embedding)  # TODO: relation similarity selection
                
                # Create scored relationship copy
                scored_relationship = relationship.copy()
                scored_relationship['similarity_score'] = float(similarity_score)
                scored_relationship['relation_text'] = relation_text  # Saved for debugging
                
                relationship_scores.append(scored_relationship)
                
            except Exception as e:
                logger.warning(f"Failed to compute similarity for relationship {relationship}: {e}")
                # If computation fails, assign low score but retain
                fallback_relationship = relationship.copy()
                fallback_relationship['similarity_score'] = 0.0
                relationship_scores.append(fallback_relationship)
        
        # Sort by similarity score descending
        relationship_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Deduplicate: use (src_id, relation_name, tgt_id) as unique key
        seen = set()
        unique_relationships = []
        for rel in relationship_scores:
            key = (rel.get('src_id', ''), rel.get('relation_name', ''), rel.get('tgt_id', ''))
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        # Select top_k relationships
        selected_relationships = unique_relationships[:top_k]
        
        logger.debug(f"🔗 Selected {len(selected_relationships)} relationships from {len(relationships)} candidates")
        
        # Optional: log top relations info (for debugging)
        # for i, rel in enumerate(selected_relationships[:3]):  # Only log first 3
        #     logger.debug(f"   Top {i+1}: {rel.get('src_id', '')} -> {rel.get('tgt_id', '')} "
        #                  f"(relation: {rel.get('relation_name', '')}, score: {rel.get('similarity_score', 0):.3f})")
        
        return selected_relationships

    async def _faiss_search_top_nodes(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """
        Use FAISS to retrieve most similar nodes, top_k per type
        
        Args:
            query_embedding: query embedding vector
            top_k: number of nodes to return per type
            
        Returns:
            List of nodes sorted by similarity
        """
        try:
            # Check if graph has dedicated FAISS search
            if not hasattr(self.graph, 'search_similar_entities'):
                logger.warning("Graph does not support specialized FAISS search methods")
                return []
            
            logger.info(f"🔍 Using FAISS to search for top {top_k} nodes of each type (chunk, entity, community)")
            
            all_faiss_results = []
            
            # Search from three dedicated indexes, top_k per type
            node_types = ['chunk', 'entity', 'community']
            
            for node_type in node_types:
                logger.debug(f"🔍 Searching {node_type} index for top {top_k} nodes")
                
                try:
                    # Use dedicated search methods
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
            
            # Sort by similarity score
            all_faiss_results.sort(key=lambda x: x[1], reverse=True)
            faiss_results = all_faiss_results

            if not faiss_results:
                logger.warning("FAISS search returned no results")
                return []
            
            # Get detailed node info
            node_ids = [node_id for node_id, score in faiss_results]
            node_info_dict = await self.graph.get_node_info_by_ids(node_ids)
            
            # Convert to compatible structure
            top_nodes = []
            for node_id, similarity_score in faiss_results:
                node_info = node_info_dict.get(node_id, {})
                node_type = node_info.get('node_type', 'unknown')
                
                # Convert node type name for compatibility
                if node_type == 'chunk':
                    formatted_type = 'chunk'
                elif node_type == 'entity':
                    formatted_type = 'entity'
                elif node_type == 'community':
                    formatted_type = 'community'
                else:
                    formatted_type = 'entity'
                
                # Build node info
                node_data = {
                    'id': node_id,
                    'type': formatted_type,
                    'similarity_score': float(similarity_score)
                }
                
                # Add extra info
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