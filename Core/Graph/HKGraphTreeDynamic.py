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
    Auxiliary information class of HK graph nodes, used for incremental update management
    """
    def __init__(self, node_id: str, node_type: str, level: int = 0, 
                 parent: Optional[str] = None, children: Optional[Set[str]] = None,
                 update_flag: bool = False, valid_flag: bool = True):
        self.node_id = node_id
        self.node_type = node_type  # 'entity', 'chunk', 'community'
        self.level = level  
        self.parent = parent  
        self.children = children or set()  
        self.update_flag = update_flag  
        self.valid_flag = valid_flag  
        self.last_modified = None  
        self.signature = None  


class HKDynamicAux:
    """
    Dynamic auxiliary structure of HKGraphTree, managing metadata for incremental updates
    """
    def __init__(self, workspace, shape: Tuple[int, int], force: bool = False):
        self.workspace = workspace
        if workspace and hasattr(workspace, 'make_for'):
            self.ns_clustering = workspace.make_for("ns_clustering")
        else:
            self.ns_clustering = None
        
        if workspace:
            if isinstance(workspace, str):
                base_path = workspace
            elif hasattr(workspace, 'root_path'):
                base_path = workspace.root_path
            else:
                base_path = "."
            
            self.signature_file = os.path.join(base_path, "hk_signatures.pkl")
            self.hyperplane_file = os.path.join(base_path, "hk_hyperplanes.npy")
            self.aux_data_file = os.path.join(base_path, "hk_aux_data.pkl")
        else:
            self.signature_file = "hk_signatures.pkl"
            self.hyperplane_file = "hk_hyperplanes.npy"
            self.aux_data_file = "hk_aux_data.pkl"
        
        if force:
            for file_path in [self.signature_file, self.hyperplane_file, self.aux_data_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed existing file: {file_path}")
        
        self.node_aux = {}  
        self.signature_map = {}  
        self.hyperplanes = self.get_hyperplanes(shape)  
        self.affected_entities = set()  
        self.level_to_nodes = defaultdict(set)  
        self.node_to_level = {}  
        
        self.incremental_mode = False  
        self.base_graph_loaded = False  
        self.last_update_timestamp = None
        
        self.use_member_based_signature = True  
        self.signature_aggregation_method = 'average'  
        
        logger.info(f"🔧 HKDynamicAux initialized with hyperplane shape: {shape}")
        logger.info(f"🔧 Community signature method: {'member-based' if self.use_member_based_signature else 'embedding-based'}")

    def save_hyperplanes(self, hyperplanes: np.ndarray):
        np.save(self.hyperplane_file, hyperplanes)
        logger.info(f"Saved hyperplanes to {self.hyperplane_file}")

    def load_hyperplanes(self) -> bool:
        if os.path.exists(self.hyperplane_file):
            self.hyperplanes = np.load(self.hyperplane_file)
            logger.info(f"✅ Loaded hyperplanes from {self.hyperplane_file}")
            return True
        return False

    def get_hyperplanes(self, shape: Tuple[int, int], force: bool = False) -> np.ndarray:
        """
        Get LSH hyperplanes, ensuring consistency
        """
        if os.path.exists(self.hyperplane_file) and not force:
            hp = np.load(self.hyperplane_file)
            logger.info("✅ Hyperplane loaded from existing file!")
        else:
            np.random.seed(42)
            hp = np.random.randn(*shape)
            np.save(self.hyperplane_file, hp)
            logger.info("❌ No existing hyperplane! Generated new hyperplane with fixed seed!")
        return hp

    def save_aux_data(self):
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
        if not os.path.exists(self.aux_data_file):
            return False
            
        try:
            with open(self.aux_data_file, 'rb') as f:
                aux_data = pickle.load(f)
            
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
        aux = HKNodeAux(
            node_id=node_id,
            node_type=node_type,
            level=level,
            parent=parent,
            children=children or set(),
            update_flag=True,  
            valid_flag=True
        )
        self.node_aux[node_id] = aux
        self.level_to_nodes[level].add(node_id)
        self.node_to_level[node_id] = level
        
        self.affected_entities.add(node_id)
        
    def update_node_level(self, node_id: str, new_level: int):
        if node_id in self.node_aux:
            old_level = self.node_aux[node_id].level
            if old_level != new_level:
                self.level_to_nodes[old_level].discard(node_id)
                self.level_to_nodes[new_level].add(node_id)
                self.node_to_level[node_id] = new_level
                self.node_aux[node_id].level = new_level
                self.node_aux[node_id].update_flag = True
                self.affected_entities.add(node_id)

    def set_parent_child_relationship(self, parent_id: str, child_id: str):
        if parent_id in self.node_aux and child_id in self.node_aux:
            self.node_aux[child_id].parent = parent_id
            self.node_aux[parent_id].children.add(child_id)
            
            self.affected_entities.add(parent_id)
            self.affected_entities.add(child_id)

    def mark_node_invalid(self, node_id: str):
        if node_id in self.node_aux:
            self.node_aux[node_id].valid_flag = False
            self.affected_entities.add(node_id)
            
            level = self.node_aux[node_id].level
            self.level_to_nodes[level].discard(node_id)
            
            #logger.debug(f"Marked node {node_id} as invalid")

    def get_valid_nodes_at_level(self, level: int) -> List[str]:
        nodes = self.level_to_nodes.get(level, set())
        return [node_id for node_id in nodes 
                if node_id in self.node_aux and self.node_aux[node_id].valid_flag]

    def get_affected_nodes_at_level(self, level: int) -> List[str]:
        nodes = self.get_valid_nodes_at_level(level)
        return [node_id for node_id in nodes 
                if node_id in self.affected_entities]

    def clear_update_flags(self):
        for aux in self.node_aux.values():
            aux.update_flag = False
        self.affected_entities.clear()

    def compute_signature(self, embedding: np.ndarray) -> int:
        if self.hyperplanes is None:
            raise ValueError("Hyperplanes not initialized")
        
        projections = np.dot(embedding, self.hyperplanes.T)
        binary_hash = (projections > 0).astype(int)
        return int(''.join(map(str, binary_hash)), 2)

    def update_node_signature(self, node_id: str, embedding: np.ndarray):
        signature = self.compute_signature(embedding)
        self.signature_map[node_id] = signature
        if node_id in self.node_aux:
            self.node_aux[node_id].signature = signature
        return signature

    def compute_community_signature_from_members(self, community_id: str, member_nodes: List[str]) -> int:
        """
        Determines the community signature by averaging the signatures of all community members.

        Args:
        community_id: Community ID
        member_nodes: List of community member nodes

        Returns:
        int: Calculated community signature
        """
        if not member_nodes:
            logger.warning(f"No members found for community {community_id}")
            return 0
        
        # Collect signatures of all valid members
        member_signatures = []
        valid_members = []
        
        for member_id in member_nodes:
            if member_id in self.signature_map:
                signature = self.signature_map[member_id]
                member_signatures.append(signature)
                valid_members.append(member_id)
        
        if not member_signatures:
            logger.warning(f"No valid signatures found for community {community_id} members")
            return 0
        
        if self.signature_aggregation_method == 'voting':
            avg_signature = self._compute_voting_signature(member_signatures)
        else:  
            avg_signature = self._compute_average_signature(member_signatures)
        
        logger.debug(f"Computed community signature for {community_id} from {len(valid_members)} members using {self.signature_aggregation_method} method")
        return avg_signature

    def _compute_average_signature(self, signatures: List[int]) -> int:
        """
        Calculates the average of signatures

        Args:
        signatures: List of signatures

        Returns:
        int: Average signature
        """
        if not signatures:
            return 0
        
        num_hyperplanes = self.hyperplanes.shape[0]
        signature_bits = []
        
        for signature in signatures:
            bits = [(signature >> i) & 1 for i in range(num_hyperplanes)]
            signature_bits.append(bits)
        
        avg_bits = []
        for i in range(num_hyperplanes):
            bit_values = [bits[i] for bits in signature_bits]
            avg_bit = sum(bit_values) / len(bit_values)
            avg_bits.append(1 if avg_bit >= 0.5 else 0)

        avg_signature = 0
        for i, bit in enumerate(avg_bits):
            avg_signature |= (bit << i)
        
        return avg_signature

    def _compute_voting_signature(self, signatures: List[int]) -> int:

        if not signatures:
            return 0
        
        num_hyperplanes = self.hyperplanes.shape[0]
        voting_bits = []
        
        for i in range(num_hyperplanes):
            bit_votes = []
            for signature in signatures:
                bit = (signature >> i) & 1
                bit_votes.append(bit)
            
            vote_count = sum(bit_votes)
            majority_bit = 1 if vote_count > len(bit_votes) / 2 else 0
            voting_bits.append(majority_bit)
        
        voting_signature = 0
        for i, bit in enumerate(voting_bits):
            voting_signature |= (bit << i)
        
        return voting_signature

    def update_community_signature_smart(self, community_id: str, member_nodes: List[str], 
                                       community_embedding: np.ndarray = None):
        """
        update community signatures: select calculation method based on configuration

        Args:

        community_id: Community ID

        member_nodes: List of community member nodes

        community_embedding: Community embedding vector (optional)
        """
        # Method 1: Calculation based on member signatures (if enabled)
        if self.use_member_based_signature and member_nodes:
            try:
                signature = self.compute_community_signature_from_members(community_id, member_nodes)
                if signature != 0:  # 确保计算成功
                    self.signature_map[community_id] = signature
                    if community_id in self.node_aux:
                        self.node_aux[community_id].signature = signature
                    logger.debug(f"Updated {community_id} signature using member-based method")
                    return signature
            except Exception as e:
                logger.warning(f"Member-based signature calculation failed for {community_id}: {e}")
        
        # Method 2: Alternative use of embedded signature
        if community_embedding is not None:
            signature = self.compute_signature(community_embedding)
            self.signature_map[community_id] = signature
            if community_id in self.node_aux:
                self.node_aux[community_id].signature = signature
            logger.debug(f"Updated {community_id} signature using embedding-based method")
            return signature
        
        logger.warning(f"Failed to update signature for {community_id}")
        return 0

    def mark_node_affected(self, node_id: str):
        self.affected_entities.add(node_id)
        if node_id in self.node_aux:
            self.node_aux[node_id].update_flag = True

    def get_statistics(self) -> Dict[str, Any]:
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

    def get_signature_statistics(self) -> Dict[str, Any]:
        """Get signature calculation statistics"""
        member_based_count = 0
        embedding_based_count = 0
        
        # Count the calculation methods of community signatures
        for node_id, aux in self.node_aux.items():
            if node_id.startswith('COMMUNITY_') and node_id in self.signature_map:
                # Check if there is member information to determine the calculation method
                if node_id in self.community_children and self.community_children[node_id]:
                    member_based_count += 1
                else:
                    embedding_based_count += 1
        
        return {
            'total_signatures': len(self.signature_map),
            'member_based_signatures': member_based_count,
            'embedding_based_signatures': embedding_based_count,
            'use_member_based_signature': self.use_member_based_signature,
            'signature_aggregation_method': self.signature_aggregation_method,
            'hyperplane_count': self.hyperplanes.shape[0] if self.hyperplanes is not None else 0
        }


class HKGraphTreeDynamic(HKGraphTree):
    """
    Dynamic incremental update version of HKGraphTree

    Integrates the incremental update mechanism of EraRAG TreeGraphDynamic:
    1. Fixed LSH hyperplanes ensure signature consistency
    2. Fine-grained tracking of affected nodes
    3. Hierarchical local reconstruction
    4. Efficient incremental embedding updates
    """
    
    def __init__(self, config, embed_config, llm, encoder, **kwargs):
        super().__init__(config, embed_config, llm, encoder, **kwargs)
        
        # Incremental update configuration
        self.enable_incremental_update = getattr(config, 'enable_incremental_update', True)
        self.incremental_batch_size = getattr(config, 'incremental_batch_size', 10)
        self.max_affected_ratio = getattr(config, 'max_affected_ratio', 0.5)  # Maximum affected node ratio
        self.enable_cross_chunk_connections = getattr(config, 'enable_cross_chunk_connections', True)  # Enable new and old chunk connections
        
        # Initialize dynamic auxiliary structure
        hyperplane_shape = (self.lsh_num_hyperplanes, self.cleora_dim)
        workspace = getattr(config, 'faiss_index_path', './faiss_index_temp/')
        self.aux = HKDynamicAux(workspace, hyperplane_shape, force=False)
        
        # Incremental update state management
        self.incremental_mode = False
        self.base_hierarchy_built = False
        
        logger.info(f"🚀 HKGraphTreeDynamic initialized with incremental update enabled: {self.enable_incremental_update}")

    async def _load_graph(self, force: bool = False) -> bool:
        """
        Override load method, supporting incremental update mode
        """
        # First try to load the base graph and hierarchy
        base_loaded = await super()._load_graph(force)
        
        if base_loaded:
            # Try to load auxiliary data
            aux_loaded = self.aux.load_aux_data()
            if aux_loaded:
                self.aux.base_graph_loaded = True
                self.base_hierarchy_built = True
                logger.info("✅ Successfully loaded base graph and auxiliary data for incremental updates")
                return True
            else:
                logger.warning("⚠️ Base graph loaded but auxiliary data missing - will need to rebuild aux structure")
                # If the base graph exists but the auxiliary data is missing, the auxiliary structure needs to be rebuilt
                await self._rebuild_aux_structure()
                return True
        
        return False

    async def _rebuild_aux_structure(self):
        """
        Rebuild the auxiliary data structure from the existing graph
        """
        logger.info("🔧 Rebuilding auxiliary structure from existing graph...")
        
        # Clean up existing auxiliary data
        self.aux.node_aux.clear()
        self.aux.signature_map.clear()
        self.aux.affected_entities.clear()
        self.aux.level_to_nodes.clear()
        self.aux.node_to_level.clear()
        
        # Rebuild the auxiliary information of the base nodes
        all_nodes = await self._graph.get_nodes()
        for node_id in all_nodes:
            node_data = await self._graph.get_node(node_id)
            if node_data:
                # Determine the node type
                if node_id.startswith('CHUNK_'):
                    node_type = 'chunk'
                elif node_id.startswith('COMMUNITY_'):
                    node_type = 'community'
                else:
                    node_type = 'entity'
                
                # Add to the auxiliary structure
                self.aux.add_node_aux(node_id, node_type, level=0)
                
                # If there is an embedding, calculate the signature
                if node_id in self.node_embeddings:
                    embedding = self.node_embeddings[node_id]
                    self.aux.update_node_signature(node_id, embedding)
        
        # Rebuild the auxiliary information of the hierarchy
        if hasattr(self, 'hierarchy_levels'):
            for level, communities in self.hierarchy_levels.items():
                for community_data in communities:
                    # Extract the community_id from the community_data dictionary
                    if isinstance(community_data, dict):
                        community_id = community_data.get('id')
                    else:
                        # Compatible with possible string formats
                        community_id = community_data
                    
                    if community_id and community_id not in self.aux.node_aux:
                        self.aux.add_node_aux(community_id, 'community', level=int(level)+1)
                    elif community_id:
                        self.aux.update_node_level(community_id, int(level)+1)
                    
                    # Establish parent-child relationship
                    if community_id:
                        children = self.community_children.get(community_id, [])
                        for child_id in children:
                            if child_id in self.aux.node_aux:
                                self.aux.set_parent_child_relationship(community_id, child_id)
        
        # Save the rebuilt auxiliary data
        self.aux.save_aux_data()
        self.aux.base_graph_loaded = True
        self.base_hierarchy_built = True
        
        stats = self.aux.get_statistics()
        logger.info(f"✅ Rebuilt auxiliary structure: {stats}")

    async def _build_graph(self, chunk_list: List[Any]):
        """
        Override the graph build method, ensuring that the auxiliary data structure is also created during the initial build
        """

        await super()._build_graph(chunk_list)
        
        if not self.incremental_mode and self.enable_incremental_update:
            logger.info("🔧 Creating auxiliary data structure for future incremental updates")
            await self._create_initial_aux_structure()
    
    async def _create_initial_aux_structure(self):
        """
        Create the auxiliary data structure for the initial build graph
        """
        logger.info("🛠️ Creating initial auxiliary structure")
        
        # Clean up existing auxiliary data
        self.aux.node_aux.clear()
        self.aux.signature_map.clear()
        self.aux.affected_entities.clear()
        self.aux.level_to_nodes.clear()
        self.aux.node_to_level.clear()
        
        # Create the auxiliary information for all base nodes
        all_nodes = await self._graph.get_nodes()
        for node_id in all_nodes:
            node_data = await self._graph.get_node(node_id)
            if node_data:
                # Determine the node type
                if node_id.startswith('CHUNK_'):
                    node_type = 'chunk'
                elif node_id.startswith('COMMUNITY_'):
                    node_type = 'community'
                else:
                    node_type = 'entity'
                
                # Add to the auxiliary structure
                self.aux.add_node_aux(node_id, node_type, level=0)
                
                # If there is an embedding, calculate the signature
                if node_id in self.node_embeddings:
                    embedding = self.node_embeddings[node_id]
                    self.aux.update_node_signature(node_id, embedding)
        
        # Create the auxiliary information for the hierarchy
        if hasattr(self, 'hierarchy_levels'):
            for level, communities in self.hierarchy_levels.items():
                for community_data in communities:
                    # Extract the community_id from the community_data dictionary
                    if isinstance(community_data, dict):
                        community_id = community_data.get('id')
                    else:
                        community_id = community_data
                    
                    if community_id:
                        # Update the node level information
                        if community_id in self.aux.node_aux:
                            self.aux.update_node_level(community_id, int(level)+1)
                        else:
                            self.aux.add_node_aux(community_id, 'community', level=int(level)+1)
                        
                        # Establish parent-child relationship
                        children = self.community_children.get(community_id, [])
                        for child_id in children:
                            if child_id in self.aux.node_aux:
                                self.aux.set_parent_child_relationship(community_id, child_id)
        
        # Clear the update flags (all nodes are "new" in the initial state, but no update flags are needed)
        self.aux.clear_update_flags()
        
        # Save the auxiliary data
        self.aux.save_aux_data()
        self.aux.base_graph_loaded = True
        self.base_hierarchy_built = True
        
        stats = self.aux.get_statistics()
        logger.info(f"✅ Created initial auxiliary structure: {stats}")

    async def insert_incremental(self, new_chunk_list: List[Any]) -> bool:
        """
        Incremental insertion of new chunks
        
        Args:
        new_chunk_list: List of new chunks, format: [(chunk_key, TextChunk), ...]
            
        Returns:
        bool: Whether the insertion is successful
        """
        if not self.enable_incremental_update:
            logger.warning("Incremental update is disabled, falling back to full rebuild")
            return await self._build_graph(new_chunk_list)
        
        if not self.base_hierarchy_built:
            logger.info("Base hierarchy not built, performing initial build...")
            return await self._build_graph(new_chunk_list)
        
        logger.info(f"🚀 Starting incremental update with {len(new_chunk_list)} new chunks")
        
        try:
            # Set incremental mode
            self.incremental_mode = True
            self.aux.incremental_mode = True
            
            # Step 1: Process new chunks, build the base graph part
            await self._process_incremental_chunks(new_chunk_list)
            
            # Step 2: Update Cleora embeddings
            await self._update_cleora_embeddings_incremental()
            
            # Step 3: Execute incremental hierarchical clustering
            await self._update_hierarchy_incremental()
            
            # Step 4: Update FAISS indexes
            await self._update_faiss_indexes_incremental()
            
            # Step 5: Save the updated data
            await self._save_incremental_updates()
            
            # Clear update flags
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
        Process new chunks, incrementally build the base graph
        """
        logger.info(f"📝 Processing {len(new_chunk_list)} new chunks for incremental update")
        
        # Step 1: Extract entities and relationships from new chunks
        er_results = []
        passage_results = []
        
        logger.info("🛠️ Extracting entities and relationships from new chunks")
        
        # Use concurrency control to process chunks
        er_results, passage_results = await self._process_chunks_with_concurrency_control(new_chunk_list)
        
        # Step 2: build hybrid Graph
        await self._build_incremental_hybrid_graph(er_results, passage_results, new_chunk_list)

    async def _process_chunks_with_concurrency_control(self, chunk_list: List[Any]) -> Tuple[List[Dict], List[Dict]]:

        max_concurrent = getattr(self, 'max_concurrent_summaries', 35)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _process_single_chunk(chunk_data):
            async with semaphore:
                try:

                    if isinstance(chunk_data, dict):
                        chunk_content = chunk_data.get('content', '')
                        from Core.Common.Utils import mdhash_id
                        chunk_key = mdhash_id(chunk_content.strip(), prefix="doc-")
                    else:
                        chunk_key, chunk_info = chunk_data
                        chunk_content = chunk_info.content
                    
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
        
        tasks = [_process_single_chunk(chunk_data) for chunk_data in chunk_list]
        logger.info(f"🔧 Processing {len(tasks)} chunks with max concurrency {max_concurrent}")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
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

        logger.info("🛠️ Building incremental hybrid graph")
        
        all_entities = defaultdict(list)
        all_relationships = defaultdict(list)
        chunk_entities_map = defaultdict(set)
        entity_chunks_map = defaultdict(set)
        wiki_entities_map = defaultdict(list)
        
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
        
        for passage_result in passage_results:
            chunk_key = passage_result['chunk_key']
            wiki_entities = passage_result['wiki_entities']
            
            for wiki_entity, _ in wiki_entities.items():
                wiki_entities_map[wiki_entity].append(chunk_key)
        
        new_chunk_nodes = defaultdict(list)
        for chunk_data in chunk_list:
            if isinstance(chunk_data, dict):
                chunk_content = chunk_data.get('content', '')
                from Core.Common.Utils import mdhash_id
                chunk_key = mdhash_id(chunk_content.strip(), prefix="doc-")
            else:
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
            
            self.aux.add_node_aux(chunk_node_id, 'chunk', level=0)
        
        new_entity_nodes = defaultdict(list)
        for entity_name, entity_list in all_entities.items():
            if not await self._graph.has_node(entity_name):
                new_entity_nodes[entity_name] = entity_list
                self.aux.add_node_aux(entity_name, 'entity', level=0)
            else:
                self.aux.affected_entities.add(entity_name)
        
        new_entity_chunk_relationships = defaultdict(list)
        new_chunk_chunk_relationships = defaultdict(list)
        
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
        
        chunk_pair_shared_entities = defaultdict(list)
        
        all_entities_in_new_chunks = set(wiki_entities_map.keys())
        logger.info(f"🔍 Processing {len(all_entities_in_new_chunks)} entities for chunk-chunk connections")
        
        entity_to_existing_chunks = {}
        if self.enable_cross_chunk_connections:
            logger.info("🔗 Cross-chunk connections enabled, querying existing chunks")
            entity_to_existing_chunks = await self._get_existing_chunks_for_entities_batch(all_entities_in_new_chunks)
        else:
            logger.info("🚫 Cross-chunk connections disabled, skipping existing chunk queries")
        
        logger.info(f"📊 Found {len(entity_to_existing_chunks)} entities with existing chunk connections")
        
        for wiki_entity, new_chunk_keys in wiki_entities_map.items():
            # Get the existing chunks related to the entity
            existing_chunk_keys = entity_to_existing_chunks.get(wiki_entity, [])
            
            # Merge the new and old chunk lists
            all_chunk_keys = list(set(new_chunk_keys + existing_chunk_keys))
            
            if len(all_chunk_keys) < 2:
                continue
            
            for chunk1, chunk2 in combinations(all_chunk_keys, 2):
                if chunk1 in new_chunk_keys or chunk2 in new_chunk_keys:
                    chunk_pair = tuple(sorted([chunk1, chunk2]))
                    chunk_pair_shared_entities[chunk_pair].append(wiki_entity)
        
        logger.info(f"🔗 Generated {len(chunk_pair_shared_entities)} potential chunk-chunk connections")
        
        shared_entity_threshold = getattr(self.config, 'shared_entity_threshold', 2)
        new_new_connections = 0  
        new_old_connections = 0  
        
        for (chunk1, chunk2), shared_entities in chunk_pair_shared_entities.items():
            if len(shared_entities) < shared_entity_threshold:
                continue
            
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
        
        logger.info("🛠️ Adding new nodes and edges to graph")
        
        all_new_nodes = {**new_entity_nodes, **new_chunk_nodes}
        if all_new_nodes:
            await self._add_nodes_with_concurrency_control(all_new_nodes)
        
        all_new_edges = {**all_relationships, **new_entity_chunk_relationships, **new_chunk_chunk_relationships}
        if all_new_edges:
            await self._add_edges_with_concurrency_control(all_new_edges)
        
        logger.info(f"✅ Added {len(all_new_nodes)} new nodes and {len(all_new_edges)} new edges to graph")

    async def _get_existing_chunks_for_entity(self, entity_name: str) -> List[str]:
        """
        Query the chunk nodes containing the specified entity in the existing graph
        
        Args:
            entity_name: Entity name
            
        Returns:
            List[str]: List of chunk keys containing the entity (without the CHUNK_ prefix)
        """
        try:
            # Check if the entity exists in the graph
            if not await self._graph.has_node(entity_name):
                return []
            
            # Find the neighbor chunk nodes through the entity node
            neighbors = await self._graph.neighbors(entity_name)
            existing_chunks = []
            
            for neighbor in neighbors:
                if neighbor.startswith('CHUNK_'):
                    # Extract the chunk key (remove the CHUNK_ prefix)
                    chunk_key = neighbor.replace('CHUNK_', '')
                    existing_chunks.append(chunk_key)
            
            logger.debug(f"Entity '{entity_name}' connected to {len(existing_chunks)} existing chunks")
            return existing_chunks
            
        except Exception as e:
            logger.warning(f"Failed to get existing chunks for entity '{entity_name}': {e}")
            return []

    async def _get_existing_chunks_for_entities_batch(self, entity_names: set) -> Dict[str, List[str]]:
        """
        Batch query the related chunk nodes of multiple entities in the existing graph (using concurrency control)
        
        Args:
            entity_names: Entity name set
            
        Returns:
            Dict[str, List[str]]: Entity name -> chunk key list mapping
        """
        if not entity_names:
            return {}
        
        # Use the same concurrency control parameters as the parent class
        max_concurrent = getattr(self, 'max_concurrent_summaries', 35)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _get_single_entity_chunks(entity_name):
            async with semaphore:
                existing_chunks = await self._get_existing_chunks_for_entity(entity_name)
                return entity_name, existing_chunks
        
        # Execute concurrent queries
        tasks = [_get_single_entity_chunks(entity_name) for entity_name in entity_names]
        logger.info(f"🔧 Querying existing chunks for {len(tasks)} entities with max concurrency {max_concurrent}")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process the results
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
        Use concurrency control to add nodes
        """
        # Use the same concurrency control parameters as the parent class
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
        Use concurrency control to add edges
        """
        # Use the same concurrency control parameters as the parent class
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
        Incremental update Cleora embeddings, only process the affected nodes   
        """
        logger.info("🔄 Updating Cleora embeddings incrementally")
        
        # Get all affected nodes
        affected_nodes = list(self.aux.affected_entities)
        if not affected_nodes:
            logger.info("No affected nodes found, skipping embedding update")
            return
        
        logger.info(f"Updating embeddings for {len(affected_nodes)} affected nodes")
        
        # Step 1: Generate initial text embeddings for new nodes
        new_nodes = []
        for node_id in affected_nodes:
            if node_id not in self.node_text_embeddings:
                try:
                    if node_id.startswith('CHUNK_'):
                        # Chunk node: use chunk content
                        chunk_key = node_id.replace('CHUNK_', '')
                        node_data = await self._graph.get_node(node_id)
                        if node_data and 'description' in node_data:
                            text_embedding = await self._embed_text(node_data['description'])
                            self.node_text_embeddings[node_id] = np.array(text_embedding)
                    else:
                        # Entity node: use entity name and description
                        node_data = await self._graph.get_node(node_id)
                        if node_data:
                            text_content = node_data.get('entity_name', node_id)
                            if 'description' in node_data:
                                text_content += ": " + node_data['description']
                            text_embedding = await self._embed_text(text_content)
                            self.node_text_embeddings[node_id] = np.array(text_embedding)
                    
                    # New nodes initially use text embeddings as node embeddings
                    if node_id in self.node_text_embeddings:
                        self.node_embeddings[node_id] = self.node_text_embeddings[node_id].copy()
                        new_nodes.append(node_id)
                        
                except Exception as e:
                    logger.warning(f"Failed to generate text embedding for {node_id}: {e}")
        
        logger.info(f"Generated text embeddings for {len(new_nodes)} new nodes")
        
        # Step 2: Get the adjacency information of all nodes (only the adjacency information of the affected nodes is needed)
        adj_list = {}
        for node_id in affected_nodes:
            try:
                neighbors = list(await self._graph.neighbors(node_id))
                adj_list[node_id] = neighbors
            except Exception as e:
                logger.warning(f"Failed to get neighbors for {node_id}: {e}")
                adj_list[node_id] = []
        
        # Step 3: Execute Cleora iterations (the same as the original algorithm)
        logger.info(f"Running Cleora iterations (iterations={self.cleora_iterations})")
        
        for iteration in range(self.cleora_iterations):
            logger.debug(f"Cleora iteration {iteration + 1}/{self.cleora_iterations}")
            
            # Create temporary storage for the updated embeddings
            updated_embeddings = {}
            
            for node_id in affected_nodes:
                if node_id not in self.node_embeddings:
                    continue
                    
                neighbors = adj_list.get(node_id, [])
                
                if neighbors:
                    # Collect neighbor embeddings (including itself)
                    embeddings_to_aggregate = []
                    
                    # Add itself embedding
                    embeddings_to_aggregate.append(self.node_embeddings[node_id])
                    
                    # Add neighbor embeddings
                    for neighbor_id in neighbors:
                        if neighbor_id in self.node_embeddings:
                            embeddings_to_aggregate.append(self.node_embeddings[neighbor_id])
                    
                    # Aggregate: calculate the average (the same as the original algorithm)
                    if embeddings_to_aggregate:
                        aggregated = np.mean(np.vstack(embeddings_to_aggregate), axis=0)
                        updated_embeddings[node_id] = aggregated
                else:
                    # No neighbors, keep the current embeddings
                    updated_embeddings[node_id] = self.node_embeddings[node_id]
            
            # Normalize and update the embeddings
            for node_id, embedding in updated_embeddings.items():
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    self.node_embeddings[node_id] = embedding / norm
                else:
                    # Avoid zero vectors
                    self.node_embeddings[node_id] = embedding
        
        # Step 4: Update the LSH signatures of the affected nodes
        updated_count = 0
        for node_id in affected_nodes:
            if node_id in self.node_embeddings:
                # For community nodes, use the smart signature update method
                if node_id.startswith('COMMUNITY_'):
                    # Get the community members
                    member_nodes = list(self.community_children.get(node_id, []))
                    self.aux.update_community_signature_smart(node_id, member_nodes, 
                                                            self.node_embeddings[node_id])
                else:
                    # For base nodes, use the traditional method
                    self.aux.update_node_signature(node_id, self.node_embeddings[node_id])
                updated_count += 1
        
        logger.info(f"✅ Updated Cleora embeddings for {updated_count} nodes using {self.cleora_iterations} iterations")

    async def _update_hierarchy_incremental(self):
        """
        Incremental update hierarchical clustering structure
        """
        logger.info("🔄 Updating hierarchy incrementally")
        
        # Calculate the ratio of affected nodes
        total_nodes = len(self.aux.node_aux)
        affected_count = len(self.aux.affected_entities)
        affected_ratio = affected_count / total_nodes if total_nodes > 0 else 1.0
        
        logger.info(f"Affected ratio: {affected_ratio:.2%} ({affected_count}/{total_nodes})")
        
        # If the affected nodes are too many, perform a full rebuild
        if affected_ratio > self.max_affected_ratio:
            logger.info(f"Affected ratio {affected_ratio:.2%} > threshold {self.max_affected_ratio:.2%}, performing full hierarchy rebuild")
            await self._rebuild_full_hierarchy()
            return
        
        # Execute incremental hierarchical update
        await self._incremental_hierarchy_update()

    async def _incremental_hierarchy_update(self):
        """
        Execute incremental hierarchical update
        """
        logger.info("🔧 Performing incremental hierarchy update")
        
        # Start from the bottom layer, process the affected nodes layer by layer
        max_level = max(self.aux.level_to_nodes.keys()) if self.aux.level_to_nodes else 0
        
        for level in range(max_level + 1):#TODO
            affected_nodes = self.aux.get_affected_nodes_at_level(level)
            if not affected_nodes:
                continue
            
            logger.info(f"Processing level {level} with {len(affected_nodes)} affected nodes")
            
            if level == 0:
                # Bottom layer: process the new added base nodes
                await self._process_level_0_incremental(affected_nodes)
            else:
                # Upper layer: re-cluster the affected communities
                await self._process_upper_level_incremental(level, affected_nodes)
        
        # Check if a new top level needs to be created
        await self._check_and_create_new_top_level()

    async def _process_level_0_incremental(self, affected_nodes: List[str]):
        """
        Process the incremental update of the 0th layer (base layer)
        """
        logger.info(f"Processing {len(affected_nodes)} affected base nodes")
        
        # Get the embeddings of these nodes
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
        
        # Try to assign the new nodes to the existing 1st level communities
        level_1_communities = self.aux.get_valid_nodes_at_level(1)
        
        assignments = {}  # node_id -> community_id
        unassigned_nodes = []
        communities_to_update = set()  # Communities that need to be re-generated summaries
        
        for i, node_id in enumerate(affected_node_ids):
            node_embedding = affected_embeddings[i]
            node_signature = self.aux.signature_map.get(node_id)
            
            best_community = None
            best_similarity = -1
            
            # Find the most similar and not full communities
            for community_id in level_1_communities:
                if community_id in self.node_embeddings:
                    # Check the community size limit
                    current_size = len(self.community_children.get(community_id, set()))
                    if current_size >= self.lsh_max_cluster_size:
                        #logger.debug(f"Community {community_id} is full ({current_size}/{self.lsh_max_cluster_size}), skipping")
                        continue
                    
                    community_embedding = self.node_embeddings[community_id]
                    similarity = np.dot(node_embedding, community_embedding)
                    
                    # Check the LSH signature compatibility
                    community_signature = self.aux.signature_map.get(community_id)
                    if node_signature and community_signature:
                        # Calculate the Hamming distance
                        hamming_distance = bin(node_signature ^ community_signature).count('1')
                        # If the Hamming distance is too large, reduce the similarity
                        if hamming_distance > self.lsh_num_hyperplanes // 4:
                            similarity *= 0.5 #TODO
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_community = community_id
            
            # If a suitable community is found and the similarity is high enough
            if best_community and best_similarity > 0.5:
                assignments[node_id] = best_community
                #logger.debug(f"Assigned {node_id} to {best_community} (similarity: {best_similarity:.3f})")
                
                # Update the child nodes of the community
                self.aux.set_parent_child_relationship(best_community, node_id)
                self.community_children[best_community].add(node_id)
                self.community_parents[node_id] = best_community
                
                # Mark the community as needing update
                self.aux.affected_entities.add(best_community)
                communities_to_update.add(best_community)
                
                # Critical fix: When a community receives new members, the impact also needs to be propagated upwards
                await self._propagate_impact_to_parent(best_community)
            else:
                unassigned_nodes.append(node_id)
        
        logger.info(f"Assigned {len(assignments)} nodes to existing communities, {len(unassigned_nodes)} nodes need new communities")
        
        # Re-generate summaries for communities that received new members (using concurrency control)
        if communities_to_update:
            logger.info(f"Updating summaries for {len(communities_to_update)} communities that received new members")
            await self._update_communities_with_concurrency_control(list(communities_to_update), 0)
        
        # Create new communities for unassigned nodes (if the number is enough)
        if len(unassigned_nodes) >= self.lsh_min_cluster_size:
            await self._create_new_communities_for_unassigned(unassigned_nodes, 0)

    async def _create_new_communities_for_unassigned(self, unassigned_nodes: List[str], level: int):
        """
        Create new communities for unassigned nodes (using concurrency control)
        """
        logger.info(f"Creating new communities for {len(unassigned_nodes)} unassigned nodes at level {level}")
        
        # Get the embeddings of the unassigned nodes
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
        
        # Perform LSH clustering on the unassigned nodes
        clusters = await self._lsh_clustering(embeddings, valid_node_ids)
        
        # Filter out clusters that meet the minimum size requirements
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
        
        # Generate community summaries and embeddings using concurrency control
        await self._create_communities_with_concurrency_control(valid_clusters, level)
        
        logger.info(f"✅ Successfully created {len(valid_clusters)} new communities at level {level}")

    async def _create_communities_with_concurrency_control(self, valid_clusters: List[Tuple[str, List[str], int]], level: int):
        """
        Use concurrency control to create communities
        
        Args:
            valid_clusters: [(community_id, cluster_nodes, level), ...]
            level: Current level
        """
        max_concurrent = getattr(self, 'max_concurrent_summaries', 35)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _create_single_community(community_data):
            community_id, cluster_nodes, community_level = community_data
            async with semaphore:
                try:
                    # Generate community summaries and embeddings
                    await self._generate_community_summary_and_embedding(community_id, cluster_nodes, community_level)
                    
                    # Calculate the community signature using member signatures (new)
                    self.aux.update_community_signature_smart(community_id, cluster_nodes, 
                                                            self.node_embeddings.get(community_id))
                    
                    # Update the hierarchy - keep the same dictionary structure as the original
                    if level not in self.hierarchy_levels:
                        self.hierarchy_levels[level] = []
                    
                    community_data = {
                        'id': community_id,
                        'nodes': cluster_nodes,
                        'level': community_level
                    }
                    self.hierarchy_levels[level].append(community_data)
                    
                    # Set parent-child relationships
                    self.community_children[community_id] = set(cluster_nodes)
                    for child_id in cluster_nodes:
                        self.aux.set_parent_child_relationship(community_id, child_id)
                        self.community_parents[child_id] = community_id
                    
                    # Add to the auxiliary structure
                    self.aux.add_node_aux(community_id, 'community', level=level+1)
                    
                    # Critical fix: Newly created communities also need to propagate impact upwards
                    await self._propagate_impact_to_parent(community_id)
                    
                    logger.info(f"Created new community {community_id} with {len(cluster_nodes)} members")
                    return community_id
                    
                except Exception as e:
                    logger.error(f"Failed to create community {community_id}: {e}")
                    return None
        
        # Create concurrent tasks
        tasks = [_create_single_community(community_data) for community_data in valid_clusters]
        logger.info(f"🔧 Creating {len(tasks)} communities with max concurrency {max_concurrent}")
        
        # Execute concurrent creation
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count the results
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
        Process the incremental update of the upper level
        """
        logger.info(f"Processing upper level {level} with {len(affected_nodes)} affected nodes")
        
        # Step 1: Re-generate the summaries and embeddings of the affected communities (using concurrency control)
        await self._update_communities_with_concurrency_control(affected_nodes, level)
        
        # Step 2: Process the new communities at this layer, need to cluster them into higher layers
        await self._process_new_communities_for_upper_clustering(level)

    async def _update_communities_with_concurrency_control(self, community_ids: List[str], level: int):
        """
        Update the summaries and embeddings of the communities (using concurrency control)
        """
        if not community_ids:
            return
            
        # Use the same concurrency control parameters as the parent class
        max_concurrent = getattr(self, 'max_concurrent_summaries', 35)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _update_single_community(community_id):
            async with semaphore:
                if community_id in self.community_children:
                    children = self.community_children[community_id]
                    await self._generate_community_summary_and_embedding(community_id, children, level)
                    # Update the community signature using member signatures (modified)
                    self.aux.update_community_signature_smart(community_id, list(children), 
                                                            self.node_embeddings.get(community_id))
                    
                    # Critical fix: Mark the parent node of the community as affected, ensure the impact is propagated upwards
                    await self._propagate_impact_to_parent(community_id)
        
        tasks = [_update_single_community(community_id) for community_id in community_ids]
        logger.info(f"🔧 Updating {len(tasks)} communities with max concurrency {max_concurrent}")
        await asyncio.gather(*tasks)
        
        # Count the impact propagated upwards
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
        Process the new communities at this layer, cluster them into higher layer communities
        
        Args:
            level: Current level
        """
        logger.info(f"Processing new communities at level {level} for upper clustering")
        
        # Get all communities at this layer (including the new ones)
        current_level_communities = self.aux.get_valid_nodes_at_level(level)
        
        if not current_level_communities:
            logger.info(f"No communities found at level {level}")
            return
        
        # Identify the new communities (communities without parent nodes)
        new_communities = []
        for community_id in current_level_communities:
            parent_id = await self._get_community_parent_id(community_id)
            if not parent_id:  # No parent node means it is a new community
                new_communities.append(community_id)
        
        if not new_communities:
            logger.info(f"No new communities found at level {level} that need upper clustering")
            return
        
        logger.info(f"Found {len(new_communities)} new communities at level {level} that need upper clustering")
        
        # Get the existing communities at the higher level
        upper_level = level + 1
        upper_level_communities = self.aux.get_valid_nodes_at_level(upper_level)
        
        # Try to assign the new communities to the existing higher layer communities
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
            
            # Find the most similar and not full higher layer communities
            for parent_community_id in upper_level_communities:
                if parent_community_id in self.node_embeddings:
                    # Check the parent community size limit
                    current_size = len(self.community_children.get(parent_community_id, set()))
                    if current_size >= self.lsh_max_cluster_size:
                        #logger.debug(f"Parent community {parent_community_id} is full ({current_size}/{self.lsh_max_cluster_size}), skipping")
                        continue
                    
                    parent_embedding = self.node_embeddings[parent_community_id]
                    similarity = np.dot(community_embedding, parent_embedding)
                    
                    # Check the LSH signature compatibility
                    parent_signature = self.aux.signature_map.get(parent_community_id)
                    if community_signature and parent_signature:
                        hamming_distance = bin(community_signature ^ parent_signature).count('1')
                        if hamming_distance > self.lsh_num_hyperplanes // 4:
                            similarity *= 0.5
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_parent = parent_community_id
            
            # If a suitable parent community is found and the similarity is high enough
            if best_parent and best_similarity > 0.5:  # 使用相同的阈值
                assignments[community_id] = best_parent
                # Set parent-child relationships
                self.aux.set_parent_child_relationship(best_parent, community_id)
                self.community_parents[community_id] = best_parent
                if best_parent in self.community_children:
                    self.community_children[best_parent].add(community_id)
                else:
                    self.community_children[best_parent] = {community_id}
                
                # Mark the parent community as needing update
                self.aux.affected_entities.add(best_parent)
                # Propagate the impact upwards
                await self._propagate_impact_to_parent(best_parent)
                
                logger.info(f"Assigned community {community_id} to parent {best_parent} (similarity: {best_similarity:.3f})")
            else:
                unassigned_communities.append(community_id)
        
        logger.info(f"Assigned {len(assignments)} communities to existing upper communities, {len(unassigned_communities)} communities need new upper communities")
        
        # Create new higher layer communities for unassigned communities (if the number is enough)
        if len(unassigned_communities) >= self.lsh_min_cluster_size:
            await self._create_new_communities_for_unassigned(unassigned_communities, upper_level-1)
        elif unassigned_communities:
            # If the number of unassigned communities is not enough to create new communities, but又不为空
            # Consider reducing the threshold to re-assign, or waiting for more communities
            logger.info(f"Only {len(unassigned_communities)} unassigned communities, less than minimum cluster size {self.lsh_min_cluster_size}")

    async def _propagate_impact_to_parent(self, community_id: str):
        """
        Propagate the impact to the parent node
        
        Args:
            community_id: Current updated community ID
        """
        try:
            # Get the parent node ID
            parent_id = await self._get_community_parent_id(community_id)
            
            if parent_id:
                # Mark the parent node as affected
                self.aux.mark_node_affected(parent_id)
                
                logger.debug(f"Propagated impact from {community_id} to parent {parent_id}")
            else:
                logger.debug(f"Community {community_id} has no parent (top level)")
                
        except Exception as e:
            logger.warning(f"Failed to propagate impact from {community_id}: {e}")

    async def _get_community_parent_id(self, community_id: str) -> Optional[str]:
        """
        Get the parent node ID of the community
        
        Args:
            community_id: Community ID
            
        Returns:
            Optional[str]: Parent node ID, if there is no parent node, return None
        """
        try:
            # Method 1: Get the parent node ID from the auxiliary structure
            if community_id in self.aux.node_aux:
                parent_id = self.aux.node_aux[community_id].parent
                if parent_id:
                    return parent_id
            
            # Method 2: Get the parent node ID from the community_parents mapping
            parent_id = self.community_parents.get(community_id)
            if parent_id:
                return parent_id
            
            # Method 3: Infer the parent node ID through the hierarchy (if the first two methods fail)
            # Parse the level of the current community
            level_match = re.search(r'COMMUNITY_L(\d+)_C\d+', community_id)
            if level_match:
                current_level = int(level_match.group(1))
                parent_level = current_level + 1
                
                # Find possible parent nodes
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
        Check if a new top level is needed for the new communities
        """
        logger.debug("Checking if new top level is needed")
        
        # Get the current highest level
        if not self.hierarchy_levels:
            return
        
        max_level = max(int(level) for level in self.hierarchy_levels.keys())
        #top_level_communities = self.hierarchy_levels.get(str(max_level), [])
        top_level_communities = self.hierarchy_levels.get(max_level)
        logger.info(f"Top level has {len(top_level_communities)} communities")
        
        # If the number of top level communities is too many, consider creating a new top level
        if len(top_level_communities) > self.lsh_max_cluster_size:
            logger.info(f"Top level has {len(top_level_communities)} communities, considering creating new top level")
            
            # Check if the maximum hierarchy limit has been reached
            if max_level >= self.max_hierarchy_levels - 1:
                logger.info(f"Already at maximum hierarchy levels ({self.max_hierarchy_levels}), not creating new top level")
                return
            
            # Get the embeddings of the top level communities
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
                
                # Cluster the top level communities
                embeddings = np.array([self.node_embeddings[cid] for cid in top_communities_with_embeddings])
                clusters = await self._lsh_clustering(embeddings, top_communities_with_embeddings)
                
                new_level = max_level + 1
                new_communities = []
                
                # Create new top level communities for each cluster
                for cluster_id, cluster_nodes in enumerate(clusters):
                    if len(cluster_nodes) >= self.lsh_min_cluster_size:
                        new_community_id = f"COMMUNITY_L{new_level}_C{cluster_id}"
                        
                        # Generate community summaries and embeddings
                        await self._generate_community_summary_and_embedding(new_community_id, cluster_nodes, new_level)
                        
                        # Calculate the community signature using member signatures (new)
                        self.aux.update_community_signature_smart(new_community_id, cluster_nodes, 
                                                                self.node_embeddings.get(new_community_id))
                        
                        # Create community data
                        community_data = {
                            'id': new_community_id,
                            'nodes': cluster_nodes,
                            'level': new_level
                        }
                        new_communities.append(community_data)
                        
                        # Set parent-child relationships
                        self.community_children[new_community_id] = set(cluster_nodes)
                        for child_id in cluster_nodes:
                            self.community_parents[child_id] = new_community_id
                        
                        # Add to the auxiliary structure
                        self.aux.add_node_aux(new_community_id, 'community', level=new_level)
                        
                        logger.info(f"Created new top-level community {new_community_id} with {len(cluster_nodes)} children")
                
                # Update the hierarchy
                if new_communities:
                    self.hierarchy_levels[new_level] = new_communities
                    logger.info(f"Created new hierarchy level {new_level} with {len(new_communities)} communities")
        else:
            logger.info(f"Top level has {len(top_level_communities)} communities, no new top level needed")

    async def _rebuild_full_hierarchy(self):
        """
        Execute a complete hierarchy reconstruction
        """
        logger.info("🔄 Rebuilding full hierarchy due to large number of affected nodes")
        
        # Clear the existing hierarchy
        self.hierarchy_levels.clear()
        self.community_summaries.clear()
        self.community_children.clear()
        self.community_parents.clear()
        
        # Rebuild the hierarchy
        await self._build_hierarchy()

    async def _update_faiss_indexes_incremental(self):
        """
        Incrementally update the FAISS indexes
        """
        logger.info("🔄 Updating FAISS indexes incrementally")
        
        # Get all affected nodes
        affected_nodes = list(self.aux.affected_entities)
        if not affected_nodes:
            logger.info("No affected nodes, skipping FAISS index update")
            return
        
        await self._build_faiss_index()
        
        logger.info("✅ FAISS indexes updated")

    async def _save_incremental_updates(self):

        logger.info("💾 Saving incremental updates")
        
        # Save the graph structure
        await self._graph.persist(force=True)
        
        # Save the hierarchy data
        await self._save_hierarchy_to_storage(force=True)
        
        # Save the auxiliary data
        self.aux.save_aux_data()
        
        # Save the FAISS indexes
        await self._save_faiss_indexes()
        
        logger.info("✅ Incremental updates saved successfully")

    def get_incremental_statistics(self) -> Dict[str, Any]:
        """
        Get the statistics of the incremental updates
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
