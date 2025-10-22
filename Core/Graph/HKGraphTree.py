import asyncio
import re
import numpy as np
import random
from collections import defaultdict, deque
from typing import Any, List, Dict, Set, Tuple
from itertools import combinations
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import faiss
import pickle
import os


from Core.Graph.BaseGraph import BaseGraph
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


class HKGraphTree(BaseGraph):

    def __init__(self, config, embed_config, llm, encoder, **kwargs):
        super().__init__(config, llm, encoder)
        self._graph = HKGraphTreeStorage()
        self.embedding_model = get_rag_embedding(embed_config.embedding.api_type, embed_config)  # Embedding model
        
        # Set additional attributes from kwargs
        self.doc_chunk = kwargs.get('doc_chunk', None)
        
        #config=config.graph
        # Configuration for different extraction methods
        self.use_wat_linking = getattr(config, 'use_wat_linking', False)
        self.extract_two_step = getattr(config, 'extract_two_step', True)
        self.prior_prob = getattr(config, 'prior_prob', 0.8)
        
        # Cleora and LSH configuration
        self.cleora_dim = getattr(config, 'cleora_dim', 1024)
        self.cleora_iterations = getattr(config, 'cleora_iterations', 2)
        self.lsh_num_hyperplanes = getattr(config, 'lsh_num_hyperplanes', 16)
        self.lsh_min_cluster_size = getattr(config, 'lsh_min_cluster_size', 5)
        self.lsh_max_cluster_size = getattr(config, 'lsh_max_cluster_size', 50)
        self.max_hierarchy_levels = getattr(config, 'max_hierarchy_levels', 4)
        self.community_summary_length = getattr(config, 'community_summary_length', 300)
        self.max_concurrent_summaries = getattr(config, 'max_concurrent_summaries', 35)  # 控制生成摘要并发度
        
        # Initialize hierarchy storage
        self.node_embeddings = {}
        self.node_text_embeddings = {} #embedding of original text embedding
        self.hierarchy_levels = {}  # level -> communities
        self.community_summaries = {}
        self.community_children = {}  # community_id -> set of child community/node IDs
        self.community_parents = {}   # community_id -> parent community ID
        self.random_seed = getattr(config, 'random_seed', 42)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # FAISS configuration
        self.faiss_index_type = getattr(config, 'faiss_index_type', 'HNSW')  # HNSW, IVF, Flat
        self.faiss_hnsw_m = getattr(config, 'faiss_hnsw_m', 64)  # HNSW parameter M
        self.faiss_hnsw_ef_construction = getattr(config, 'faiss_hnsw_ef_construction', 500)  # HNSW ef_construction
        self.faiss_hnsw_ef_search = getattr(config, 'faiss_hnsw_ef_search', 200)  # HNSW ef_search
        
        # FAISS data structures - separate indexes for each node type
        self.faiss_indexes = {
            'entity': None,
            'chunk': None,
            'community': None
        }
        self.faiss_id_to_node = {
            'entity': {},
            'chunk': {},
            'community': {}
        }  # faiss_id -> node_id mapping for each type
        self.node_to_faiss_id = {
            'entity': {},
            'chunk': {},
            'community': {}
        }  # node_id -> faiss_id mapping for each type
        self.faiss_index_path = getattr(config, 'faiss_index_path', './faiss_index_th3/Popqa')
        
        logger.info(f"🔧 HKGraphTree initialized with doc_chunk: {self.doc_chunk is not None}")
        logger.info(f"🔧 FAISS config: {self.faiss_index_type}, M={self.faiss_hnsw_m}, ef_construction={self.faiss_hnsw_ef_construction}")

    async def _load_graph(self, force: bool = False) -> bool:

        base_loaded = await self._graph.load_graph(force)
        
        if not base_loaded:
            logger.info("Base graph not loaded, will need to build from scratch")
            return False

        hierarchy_loaded = await self._load_hierarchy_from_storage()
        
        faiss_loaded = await self._load_faiss_indexes()
        
        if hierarchy_loaded and faiss_loaded:
            logger.info("✅ Successfully loaded base graph, hierarchy data, and FAISS index")
            return True
        elif hierarchy_loaded:
            logger.warning("⚠️ Base graph and hierarchy loaded but FAISS index missing - may need to rebuild")
            return base_loaded
        else:
            logger.warning("⚠️ Base graph loaded but hierarchy data missing - may need to rebuild")
            return base_loaded

    async def _load_hierarchy_from_storage(self) -> bool:

        try:
            hierarchy_data = self._graph.hierarchy_data
            if hierarchy_data is None:
                logger.info("No hierarchy data found in storage")
                return False
            
            self.node_embeddings = hierarchy_data.get('node_embeddings', {})
            self.node_text_embeddings = hierarchy_data.get('node_text_embeddings', {})
            self.hierarchy_levels = hierarchy_data.get('hierarchy_levels', {})
            self.community_summaries = hierarchy_data.get('community_summaries', {})
            self.community_children = hierarchy_data.get('community_children', {})
            self.community_parents = hierarchy_data.get('community_parents', {})
            
            logger.info(f"✅ Loaded hierarchy data with {len(self.hierarchy_levels)} levels, {len(self.node_embeddings)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load hierarchy data: {e}")
            return False

    async def _save_hierarchy_to_storage(self, force: bool = False):

        try:

            hierarchy_data = {
                'node_embeddings': self.node_embeddings,
                'node_text_embeddings': self.node_text_embeddings,
                'hierarchy_levels': self.hierarchy_levels,
                'community_summaries': self.community_summaries,
                'community_children': self.community_children,
                'community_parents': self.community_parents,
                'config': {  
                    'cleora_dim': self.cleora_dim,
                    'max_hierarchy_levels': self.max_hierarchy_levels,
                    'random_seed': self.random_seed
                }
            }

            await self._graph.persist_hierarchy(hierarchy_data, force=force)
            logger.info(f"✅ Saved hierarchy data with {len(self.hierarchy_levels)} levels")
            
        except Exception as e:
            logger.error(f"Failed to save hierarchy data: {e}")
            raise

    async def _build_graph(self, chunk_list: List[Any]):

        try:
            logger.info("Building Hierarchical HK Graph with Cleora + LSH")
            
            if self.hierarchy_levels and len(self.hierarchy_levels) > 0:
                logger.info("✅ Hierarchy already loaded, skipping rebuild")
                return
            
            logger.info("No cached hierarchy found, building from scratch...")
            
            # Step 1: Build base HK Graph
            logger.info("🏗️ Step 1: Building base HK Graph")
            await self._build_base_hk_graph(chunk_list)
            
            # Step 2: Convert to homogeneous graph representation
            logger.info("🏗️ Step 2: Converting to homogeneous graph")
            homogeneous_graph = await self._convert_to_homogeneous_graph()
            
            # Step 3: Generate Cleora embeddings
            logger.info("🏗️ Step 3: Generating Cleora embeddings")
            await self._generate_cleora_embeddings(homogeneous_graph)
            
            # Step 4: Build hierarchical communities using LSH
            logger.info("🏗️ Step 4: Building hierarchical communities with LSH")
            await self._build_hierarchical_communities()
            
            # Step 5: Build FAISS index
            logger.info("🏗️ Step 5: Building FAISS index")
            await self._build_faiss_index()
            
            # Step 6: Save hierarchy and FAISS index to storage
            logger.info("🏗️ Step 6: Saving hierarchy data and FAISS index to storage")
            await self._save_hierarchy_to_storage(force=True)
            await self._save_faiss_indexes()
            
        except Exception as e:
            logger.exception(f"Error building HK graph tree: {e}")
        finally:
            logger.info("✅ HK Graph Tree construction finished")

    async def _build_base_hk_graph(self, chunk_list: List[Any]):
        """
        Build the base HK graph using the original approach.
        """
        # Step 1: Extract entities and relationships from each chunk (ER Graph approach)
        logger.info("⚙️ Extracting entities and relationships from chunks")
        er_results = await asyncio.gather(
            *[self._extract_entity_relationship(chunk) for chunk in chunk_list]
        )
        
        # Step 2: Extract entity linking information for passage connections (Passage Graph approach)
        logger.info("⚙️ Extracting entity linking for passage connections")
        passage_results = await asyncio.gather(
            *[self._extract_passage_entities(chunk) for chunk in chunk_list]
        )
        
        # Step 3: Build the hybrid graph
        logger.info("⚙️ Building hybrid graph structure")
        await self._build_hybrid_graph(er_results, passage_results, chunk_list)

    async def _extract_entity_relationship(self, chunk_key_pair: tuple[str, TextChunk]) -> Dict[str, Any]:
        """
        Extract entities and relationships from a chunk (ER Graph approach).
        """
        chunk_key, chunk_info = chunk_key_pair
        chunk_content = chunk_info.content
        
        if self.extract_two_step:
            # Two-step extraction: NER + OpenIE
            entities = await self._named_entity_recognition(chunk_content)
            triples = await self._openie_post_ner_extract(chunk_content, entities)
            maybe_nodes, maybe_edges = await self._build_graph_from_tuples(entities, triples, chunk_key)
        else:
            # One-step extraction using KG Agent
            graph_element = await self._kg_agent(chunk_content)
            maybe_nodes, maybe_edges = await self._build_graph_by_regular_matching(graph_element, chunk_key)
        
        return {
            'chunk_key': chunk_key,
            'chunk_content': chunk_content,
            'entities': maybe_nodes,
            'relationships': maybe_edges
        }

    async def _extract_passage_entities(self, chunk_key_pair: tuple[str, TextChunk]) -> Dict[str, Any]:
        """
        Extract entity linking information for passage connections (Passage Graph approach).
        """
        chunk_key, chunk_info = chunk_key_pair
        chunk_content = chunk_info.content
        
        if self.use_wat_linking:
            # Use WAT entity linking system
            wat_annotations = await self._wat_entity_linking(chunk_content)
            wiki_entities = await self._build_wiki_entities_from_wat(wat_annotations, chunk_key)
        else:
            # Use simple entity extraction
            wiki_entities = await self._simple_entity_extraction(chunk_content, chunk_key)
        
        return {
            'chunk_key': chunk_key,
            'wiki_entities': wiki_entities
        }

    async def _build_hybrid_graph(self, er_results: List[Dict], passage_results: List[Dict], chunk_list: List[Any]):
        """
        Build the hybrid graph combining all three types of connections.
        """
        all_entities = defaultdict(list)
        all_relationships = defaultdict(list)
        chunk_entities_map = defaultdict(set)  # chunk_key -> set of entities
        entity_chunks_map = defaultdict(set)   # entity -> set of chunk_keys
        wiki_entities_map = defaultdict(list)  # wiki_entity -> list of chunk_keys
        
        # Step 1: Process ER results - Entity-Entity connections
        logger.info("🛠️  Processing Entity-Entity connections")
        for er_result in er_results:
            chunk_key = er_result['chunk_key']
            entities = er_result['entities']
            relationships = er_result['relationships']
            
            # Add entities
            for entity_name, entity_list in entities.items():
                all_entities[entity_name].extend(entity_list)
                chunk_entities_map[chunk_key].add(entity_name)
                entity_chunks_map[entity_name].add(chunk_key)
            
            # Add relationships
            for rel_key, rel_list in relationships.items():
                all_relationships[rel_key].extend(rel_list)
        
        # Step 2: Process passage results - prepare for Chunk-Chunk connections
        logger.info("🛠️  Processing Passage entity linking")
        for passage_result in passage_results:
            chunk_key = passage_result['chunk_key']
            wiki_entities = passage_result['wiki_entities']
            
            for wiki_entity, _ in wiki_entities.items():
                wiki_entities_map[wiki_entity].append(chunk_key)
        
        # Step 3: Create Chunk nodes
        logger.info("🛠️ Creating Chunk nodes")
        chunk_nodes = defaultdict(list)
        for chunk_key, chunk_info in chunk_list:
            chunk_entity = HK_Node(
                entity_name=f"CHUNK_{chunk_key}",
                entity_type="CHUNK",
                description=chunk_info.content,  # 保留完整的chunk内容
                source_id=chunk_key
            )
            chunk_nodes[f"CHUNK_{chunk_key}"].append(chunk_entity)
        
        # Step 4: Create Entity-Chunk connections
        logger.info("🛠️ Creating Entity-Chunk connections")
        entity_chunk_relationships = defaultdict(list)
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
                entity_chunk_relationships[rel_key].append(relationship)
        
        # Step 5: Create Chunk-Chunk connections based on shared entities
        logger.info("🛠️ Creating Chunk-Chunk connections")
        chunk_chunk_relationships = defaultdict(list)
        
        # 首先统计每对chunk之间的共享实体
        chunk_pair_shared_entities = defaultdict(list)  # (chunk1, chunk2) -> [shared_entities]
        
        for wiki_entity, chunk_keys in wiki_entities_map.items():
            if len(chunk_keys) < 2:
                continue
            
            chunk_keys_set = set(chunk_keys)
            for chunk1, chunk2 in combinations(chunk_keys_set, 2):
                chunk_pair = tuple(sorted([chunk1, chunk2]))
                chunk_pair_shared_entities[chunk_pair].append(wiki_entity)
        
        # 设置共享实体阈值，可通过配置文件设置
        shared_entity_threshold = getattr(self.config, 'shared_entity_threshold', 2)
        logger.info(f"Using shared entity threshold: {shared_entity_threshold}")
        
        # 只为共享实体数量 >= 阈值的chunk pair创建连接
        for (chunk1, chunk2), shared_entities in chunk_pair_shared_entities.items():
            if len(shared_entities) < shared_entity_threshold:
                continue
            
            rel_key = tuple(sorted([f"CHUNK_{chunk1}", f"CHUNK_{chunk2}"]))
            
            # 创建包含所有共享实体信息的关系
            relationship = Relationship(
                src_id=rel_key[0],
                tgt_id=rel_key[1],
                relation_name="SHARED_ENTITY",
                description=f"Chunks connected through shared entities: {', '.join(shared_entities)}",
                source_id=GRAPH_FIELD_SEP.join([chunk1, chunk2] + shared_entities),
                weight=float(len(shared_entities))  # 权重设为共享实体数量
            )
            chunk_chunk_relationships[rel_key].append(relationship)
            
        logger.info(f"Created {len(chunk_chunk_relationships)} chunk-chunk connections with threshold {shared_entity_threshold}")
        
        # Step 6: Merge all nodes and edges into the graph
        logger.info("🛠️  Merging all nodes and edges into the graph")
        
        # Merge entity nodes
        all_nodes = {**all_entities, **chunk_nodes}
        await asyncio.gather(*[self._merge_nodes_then_upsert(k, v) for k, v in all_nodes.items()])
        
        # Merge all relationships
        all_edges = {**all_relationships, **entity_chunk_relationships, **chunk_chunk_relationships}
        await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in all_edges.items()])
        
        # 分别统计节点和边的数量
        num_entities = len(all_entities)
        num_chunks = len(chunk_nodes)
        num_relationships = len(all_relationships)
        num_entity_chunk_relationships = len(entity_chunk_relationships)
        num_chunk_chunk_relationships = len(chunk_chunk_relationships)

        logger.info(
            f"✅ HK Graph built with {num_entities + num_chunks} nodes "
            f"({num_entities} entities, {num_chunks} chunks) and "
            f"{num_relationships + num_entity_chunk_relationships + num_chunk_chunk_relationships} edges "
            f"({num_relationships} relationships, {num_entity_chunk_relationships} entity-chunk, {num_chunk_chunk_relationships} chunk-chunk)"
        )

    # ===== Entity Relationship Extraction Methods (from ER Graph) =====
    
    async def _named_entity_recognition(self, passage: str):
        """Named Entity Recognition using LLM."""
        ner_messages = GraphPrompt.NER.format(user_input=passage)
        entities = await self.llm.aask(ner_messages, format="json")
        
        if 'named_entities' not in entities:
            entities = []
        else:
            entities = entities['named_entities']
        return entities

    async def _openie_post_ner_extract(self, chunk, entities):
        """Open Information Extraction after NER."""
        named_entity_json = {"named_entities": entities}
        openie_messages = GraphPrompt.OPENIE_POST_NET.format(
            passage=chunk,
            named_entity_json=str(named_entity_json)
        )
        triples = await self.llm.aask(openie_messages, format="json")
        
        try:
            triples = triples["triples"]
        except:
            return []
        return triples

    async def _kg_agent(self, chunk_info):
        """Knowledge Graph Agent for one-step extraction."""
        knowledge_graph_prompt = TextPrompt(GraphPrompt.KG_AGNET)
        knowledge_graph_generation = knowledge_graph_prompt.format(task=chunk_info)
        
        knowledge_graph_generation_msg = Message(role="Graphify", content=knowledge_graph_generation)
        content = await self.llm.aask(knowledge_graph_generation_msg.content)
        return content

    @staticmethod
    async def _build_graph_from_tuples(entities, triples, chunk_key):
        """Build graph from entity and triple tuples."""
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)

        for _entity in entities:
            entity_name = clean_str(_entity)
            if entity_name == '':
                logger.warning(f"entity name is not valid, entity is: {_entity}, so skip it")
                continue
            entity = HK_Node(entity_name=entity_name, source_id=chunk_key)
            maybe_nodes[entity_name].append(entity)

        for triple in triples:
            if isinstance(triple[0], list): 
                triple = triple[0]
            if len(triple) != 3:
                logger.warning(f"triples length is not 3, triple is: {triple}, len is {len(triple)}, so skip it")
                continue
            
            src_entity = clean_str(triple[0])
            tgt_entity = clean_str(triple[2])
            relation_name = clean_str(triple[1])
            
            if src_entity == '' or tgt_entity == '' or relation_name == '':
                logger.warning(f"triple is not valid, since we have empty entity or relation, triple is: {triple}, so skip it")
                continue
            
            if isinstance(src_entity, str) and isinstance(tgt_entity, str) and isinstance(relation_name, str):
                relationship = Relationship(
                    src_id=src_entity,
                    tgt_id=tgt_entity,
                    weight=1.0,
                    source_id=chunk_key,
                    relation_name=relation_name
                )
                maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        return dict(maybe_nodes), dict(maybe_edges)

    @staticmethod
    async def _build_graph_by_regular_matching(content: str, chunk_key):
        """Build graph using regular expression matching."""
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)

        # Extract nodes
        matches = re.findall(NODE_PATTERN, content)
        for match in matches:
            entity_name, entity_type = match
            entity_name = clean_str(entity_name)
            entity_type = clean_str(entity_type)
            if entity_name not in maybe_nodes:
                entity = HK_Node(entity_name=entity_name, entity_type=entity_type, source_id=chunk_key)
                maybe_nodes[entity_name].append(entity)

        # Extract relationships
        matches = re.findall(REL_PATTERN, content)
        for match in matches:
            src_id, _, tgt_id, _, rel_type = match
            src_id = clean_str(src_id)
            tgt_id = clean_str(tgt_id)
            rel_type = clean_str(rel_type)
            if src_id in maybe_nodes and tgt_id in maybe_nodes:
                relationship = Relationship(
                    src_id=clean_str(src_id), 
                    tgt_id=clean_str(tgt_id), 
                    source_id=chunk_key,
                    relation_name=clean_str(rel_type)
                )
                maybe_edges[(src_id, tgt_id)].append(relationship)

        return maybe_nodes, maybe_edges

    # ===== Passage Entity Extraction Methods (from Passage Graph) =====
    
    @staticmethod
    async def _wat_entity_linking(text: str):
        """WAT entity linking system."""
        wat_url = 'https://wat.d4science.org/wat/tag/tag'
        payload = [("gcube-token", GCUBE_TOKEN),
                   ("text", text),
                   ("lang", 'en'),
                   ("tokenizer", "nlp4j"),
                   ('debug', 9),
                   ("method",
                    "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,voting:relatedness=lm,ranker:model=0046.model,confidence:model=pruner-wiki.linear")]
        
        retry_count = 3
        for attempt in range(retry_count):
            try:
                response = requests.get(wat_url, params=payload)
                return [WATAnnotation(**annotation) for annotation in response.json()['annotations']]
            except requests.exceptions.RequestException as e:
                logger.error(f"Retry attempt {attempt + 1} failed: {e}")
                if attempt == retry_count - 1:
                    logger.error("All retry attempts failed. Exiting.")
        return []

    async def _build_wiki_entities_from_wat(self, wat_annotations, chunk_key):
        """Build wiki entities from WAT annotations."""
        wiki_entities = defaultdict(set)
        for wiki in wat_annotations:
            if wiki.wiki_title != '' and wiki.prior_prob > self.prior_prob:
                wiki_entities[wiki.wiki_title].add(chunk_key)
        return dict(wiki_entities)

    async def _simple_entity_extraction(self, chunk_content: str, chunk_key: str):
        """Simple entity extraction when WAT is not available."""
        # Use NER as fallback for entity extraction
        entities = await self._named_entity_recognition(chunk_content)
        wiki_entities = defaultdict(set)
        for entity in entities:
            wiki_entities[entity].add(chunk_key)
        return dict(wiki_entities)

    @property
    def entity_metakey(self):
        return "index"

    # ===== New Methods for Hierarchical Graph Construction =====

    async def _convert_to_homogeneous_graph(self) -> nx.Graph:
        """
        Convert the heterogeneous HK graph to a homogeneous graph for Cleora embedding.
        All nodes (entities, chunks) are treated as the same type.
        """
        logger.info("Converting heterogeneous HK graph to homogeneous representation")
        
        # Create a new homogeneous graph
        homogeneous_graph = nx.Graph()
        
        # Get all nodes and edges from the original graph
        nodes = await self._graph.get_nodes()
        edges = await self._graph.edges()
        
        # Add all nodes to homogeneous graph
        for node_id in nodes:
            node_data = await self._graph.get_node(node_id)
            if node_data:
                # Simplify node data for homogeneous representation
                simplified_data = {
                    'id': node_id,
                    'type': 'CHUNK' if node_id.startswith('CHUNK_') else 'ENTITY',
                    'name': node_data.get('entity_name', node_id),
                    'description': node_data.get('description', ''),
                }
                homogeneous_graph.add_node(node_id, **simplified_data)
        
        # Add all edges with uniform weights
        for edge in edges:
            src, tgt = edge
            edge_data = await self._graph.get_edge(src, tgt)
            weight = edge_data.get('weight', 1.0) if edge_data else 1.0
            homogeneous_graph.add_edge(src, tgt, weight=weight)
        
        logger.info(f"Created homogeneous graph with {homogeneous_graph.number_of_nodes()} nodes and {homogeneous_graph.number_of_edges()} edges")
        return homogeneous_graph

    async def _generate_cleora_embeddings(self, graph: nx.Graph):
        """
        Generate Cleora embeddings for all nodes in the graph.
        Cleora aggregates neighbor information iteratively.
        """
        logger.info(f"Generating Cleora embeddings (dim={self.cleora_dim}, iterations={self.cleora_iterations})")
        
        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Initialize embeddings using text content of nodes
        logger.info("Initializing node embeddings using text content")
        embeddings = await self._initialize_text_embeddings(nodes, graph) #TODO start with original text embedding

        # Store node embeddings for original text
        for i, node in enumerate(nodes):
            self.node_text_embeddings[node] = embeddings[i]
        
        # Get adjacency information
        adj_list = {}
        for node in nodes:
            neighbors = list(graph.neighbors(node))
            adj_list[node] = neighbors
        
        # Cleora iterations
        for iteration in range(self.cleora_iterations):
            logger.info(f"Cleora iteration {iteration + 1}/{self.cleora_iterations}")
            new_embeddings = np.zeros_like(embeddings)
            
            for i, node in enumerate(nodes):
                neighbors = adj_list[node]
                if neighbors:
                    # Aggregate neighbor embeddings
                    neighbor_indices = [node_to_idx[neighbor] for neighbor in neighbors]
                    neighbor_embeddings = embeddings[neighbor_indices]
                    
                    # Simple aggregation: mean of neighbors + self
                    self_embedding = embeddings[i]
                    aggregated = np.mean(np.vstack([self_embedding[np.newaxis, :], neighbor_embeddings]), axis=0)
                    new_embeddings[i] = aggregated
                else:
                    # No neighbors, keep current embedding
                    new_embeddings[i] = embeddings[i]
            
            # Normalize embeddings
            norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = new_embeddings / norms
        
        # Store node embeddings after Cleora iterations
        for i, node in enumerate(nodes):
            self.node_embeddings[node] = embeddings[i]
        
        logger.info(f"Generated Cleora embeddings for {len(self.node_embeddings)} nodes")

    async def _build_hierarchical_communities(self):
        """
        Build hierarchical communities using LSH clustering.
        """
        logger.info("🏗️ Building hierarchical communities with LSH clustering")
        
        # Debug: Check initial state
        logger.info(f"📊 Initial state:")
        logger.info(f"  - Available embeddings: {len(self.node_embeddings)}")
        logger.info(f"  - LSH min cluster size: {self.lsh_min_cluster_size}")
        logger.info(f"  - LSH max cluster size: {self.lsh_max_cluster_size}")
        logger.info(f"  - Max hierarchy levels: {self.max_hierarchy_levels}")
        logger.info(f"  - Num lsh hyperplanes: {self.lsh_num_hyperplanes}")
        
        # Start with all base nodes
        current_level_nodes = list(self.node_embeddings.keys())
        logger.info(f"📝 Starting with {len(current_level_nodes)} base nodes")
        
        if len(current_level_nodes) == 0:
            logger.error("❌ No node embeddings available! Cannot build hierarchy.")
            return
            
        if len(current_level_nodes) <= self.lsh_min_cluster_size:
            logger.warning(f"⚠️ Only {len(current_level_nodes)} nodes available, less than min cluster size {self.lsh_min_cluster_size}")

            if len(current_level_nodes) >= 2:
                logger.info("🔧 Creating basic hierarchy for small dataset")
                basic_community = {
                    'id': 'COMMUNITY_L0_C0',
                    'nodes': current_level_nodes,
                    'level': 0
                }
                self.hierarchy_levels[0] = [basic_community]

                try:
                    await self._generate_community_summary_and_embedding('COMMUNITY_L0_C0', current_level_nodes, 0)
                    logger.info("✅ Created basic hierarchy for small dataset")
                except Exception as e:
                    logger.error(f"❌ Failed to create basic hierarchy: {e}")
            return
        
        level = 0
        
        while len(current_level_nodes) > self.lsh_min_cluster_size and level < self.max_hierarchy_levels:
            logger.info(f"🔍 Processing hierarchy level {level} with {len(current_level_nodes)} nodes")
            
            # Get embeddings for current level nodes
            try:
                if level == 0:
                    # Base level: use Cleora embeddings
                    embeddings = np.array([self.node_embeddings[node] for node in current_level_nodes])
                else:
                    # Higher levels: use community embeddings
                    embeddings = np.array([self.node_embeddings[node] for node in current_level_nodes])
                
                logger.info(f"📈 Got embeddings shape: {embeddings.shape}")
            except Exception as e:
                logger.error(f"❌ Failed to get embeddings for level {level}: {e}")
                break
            
            # Perform LSH clustering
            try:
                clusters = await self._lsh_clustering(embeddings, current_level_nodes)
                logger.info(f"🔗 LSH created {len(clusters)} clusters")
            except Exception as e:
                logger.error(f"❌ LSH clustering failed for level {level}: {e}")
                break
            
            # Create community nodes for this level
            communities = []
            next_level_nodes = []
            
            # First pass: Create community structure and relationships
            valid_communities = []
            for cluster_id, cluster_nodes in enumerate(clusters):
                logger.debug(f"  📦 Cluster {cluster_id}: {len(cluster_nodes)} nodes")
                if len(cluster_nodes) >= self.lsh_min_cluster_size:
                    # Create community node
                    community_id = f"COMMUNITY_L{level}_C{cluster_id}"
                    community_data = {
                        'id': community_id,
                        'nodes': cluster_nodes,
                        'level': level
                    }
                    communities.append(community_data)
                    valid_communities.append((community_id, cluster_nodes, level))
                    
                    # Build parent-child relationships
                    self.community_children[community_id] = set(cluster_nodes)
                    
                    # Establish parent-child relationships
                    for member_node in cluster_nodes:
                        self.community_parents[member_node] = community_id
                else:
                    logger.info(f"  ⚠️ Cluster {cluster_id} too small ({len(cluster_nodes)} < {self.lsh_min_cluster_size}), skipping")
            
            # Second pass: Generate community summaries and embeddings concurrently with controlled concurrency
            if valid_communities:
                concurrent_limit = min(self.max_concurrent_summaries, len(valid_communities))
                logger.info(f"🚀 Generating summaries for {len(valid_communities)} communities with max concurrency {concurrent_limit}")
                
                # Execute tasks with controlled concurrency
                summary_results = await self._generate_summaries_with_limited_concurrency(valid_communities)
                
                # Process results
                for i, ((community_id, cluster_nodes, level), result) in enumerate(zip(valid_communities, summary_results)):
                    if isinstance(result, Exception):
                        logger.error(f"  ❌ Failed to create community {community_id}: {result}")
                    else:
                        next_level_nodes.append(community_id)
                        logger.debug(f"  ✅ Created community {community_id} with {len(cluster_nodes)} children")
                
                logger.info(f"🎯 Completed concurrent summary generation for level {level} (concurrency: {concurrent_limit})")
            
            # Store this level
            if communities:
                self.hierarchy_levels[level] = communities
                logger.info(f"✅ Stored level {level} with {len(communities)} communities")
            else:
                logger.warning(f"⚠️ No valid communities created for level {level}")
                break
            
            # Prepare for next level
            current_level_nodes = next_level_nodes
            level += 1
            
            if len(current_level_nodes) == 0:
                logger.info(f"🏁 No more nodes for next level, stopping at level {level-1}")
                break
        
        logger.info(f"🎯 Built {len(self.hierarchy_levels)} levels of hierarchy")
        
        # Print comprehensive hierarchy statistics
        self._print_hierarchy_summary()
        
        logger.info(f"📊 Parent-child relationships: {len(self.community_children)} communities with children")
        logger.info(f"📊 Child-parent relationships: {len(self.community_parents)} nodes with parents")

    def _print_hierarchy_summary(self):
        """
        Print comprehensive hierarchy statistics similar to TreeGraphLSH.
        """
        if not self.hierarchy_levels:
            logger.info("⚠️ No hierarchy levels found!")
            return
            
        logger.info(f"\n" + "="*60)
        logger.info(f"📊 HKGraphTree structure summary")
        logger.info(f"="*60)
        
        total_communities = 0
        total_nodes_in_communities = 0
        level_stats = []
        
        for level, communities in self.hierarchy_levels.items():
            level_community_count = len(communities)
            level_node_sizes = []
            
            for community in communities:
                node_count = len(community['nodes'])
                level_node_sizes.append(node_count)
                total_nodes_in_communities += node_count
            
            total_communities += level_community_count
            
            level_stats.append({
                'level': level,
                'communities': level_community_count,
                'min_size': min(level_node_sizes) if level_node_sizes else 0,
                'max_size': max(level_node_sizes) if level_node_sizes else 0,
                'avg_size': np.mean(level_node_sizes) if level_node_sizes else 0,
                'median_size': np.median(level_node_sizes) if level_node_sizes else 0,
                'total_nodes': sum(level_node_sizes)
            })
            
            logger.info(f"\n🏗️  Level {level}:")
            logger.info(f"   📦 Number of communities: {level_community_count}")
            logger.info(f"   📏 Community size: Min={min(level_node_sizes) if level_node_sizes else 0}, "
                      f"Max={max(level_node_sizes) if level_node_sizes else 0}, "
                      f"Avg={np.mean(level_node_sizes):.1f}, "
                      f"Median={np.median(level_node_sizes):.1f}")
            logger.info(f"   📊 Total nodes in this level: {sum(level_node_sizes)}")
            
            # Show first few communities as examples
            for i, community in enumerate(communities[:3]):
                logger.info(f"     - {community['id']}: {len(community['nodes'])} nodes")
            if len(communities) > 3:
                logger.info(f"     - ... (and {len(communities)-3} more communities)")
        
        # Overall statistics
        logger.info(f"\n🎯 Overall Statistics:")
        logger.info(f"   📊 Total levels: {len(self.hierarchy_levels)}")
        logger.info(f"   📊 Total communities: {total_communities}")
        logger.info(f"   📊 Total embeddings: {len(self.node_embeddings)}")
        logger.info(f"   📊 Total summaries: {len(self.community_summaries)}")
        
        # Hierarchy compression ratio
        if level_stats:
            base_level_nodes = level_stats[0]['total_nodes']
            top_level_communities = level_stats[-1]['communities'] if len(level_stats) > 1 else base_level_nodes
            compression_ratio = base_level_nodes / top_level_communities if top_level_communities > 0 else 1
            logger.info(f"   📈 Hierarchy compression ratio: {compression_ratio:.1f}:1 (from {base_level_nodes} base nodes to {top_level_communities} top-level communities)")
        
        
        logger.info(f"="*60)


    async def _generate_summaries_with_limited_concurrency(self, valid_communities):
        """
        Generate community summaries with controlled concurrency using semaphore.
        """
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_concurrent_summaries)
        
        async def _process_single_community(community_data):
            """Process a single community with semaphore control."""
            community_id, cluster_nodes, level = community_data
            async with semaphore:
                return await self._generate_community_summary_and_embedding(community_id, cluster_nodes, level)
        
        # Create tasks with semaphore control
        tasks = [
            _process_single_community(community_data)
            for community_data in valid_communities
        ]
        
        # Execute all tasks with controlled concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def _analyze_bucket_distribution(self, buckets):
        """
        Analyze bucket distribution statistics.
        """
        size_counts = defaultdict(int)
        total_items = 0
        
        for items in buckets.values():
            size = len(items)
            size_counts[size] += 1
            total_items += size
        
        if not size_counts:
            return {}
    
        sorted_sizes = sorted(size_counts.items())

        return {
            'total_buckets': len(buckets),
            'size_distribution': dict(sorted_sizes),
            'max_size': max(size_counts.keys()) if size_counts else 0,
            'min_size': min(size_counts.keys()) if size_counts else 0,
            'avg_size': round(total_items / len(buckets), 2) if buckets else 0
        }

    def _print_cluster_stats(self, clusters, stage_name="Final Clustering"):
        """
        Print final clustering statistics.
        """
        if not clusters:
            logger.info(f"{stage_name} - No clustering results")
            return
            
        sizes = [len(c) for c in clusters]
        logger.info(f"\n=== {stage_name} Result Statistics ===")
        logger.info(f"🎯 Total number of clusters: {len(clusters)}")
        logger.info(f"📏 Cluster size distribution:")
        logger.info(f"   - Max cluster size: {max(sizes)} nodes")
        logger.info(f"   - Min cluster size: {min(sizes)} nodes")
        logger.info(f"   - Average cluster size: {np.mean(sizes):.1f} nodes")
        logger.info(f"   - Median cluster size: {np.median(sizes):.1f} nodes")
        
        # Cluster size distribution statistics
        size_dist = defaultdict(int)
        for size in sizes:
            size_dist[size] += 1
        
        logger.info(f"\n📊 Cluster size frequency distribution:")
        for size, count in sorted(size_dist.items())[:10]:  # Show the top 10 most common sizes
            logger.info(f"   - Size {size}: {count} clusters")
        
        if len(size_dist) > 10:
            logger.info(f"   - ... (and {len(size_dist)-10} other sizes)")

    async def _lsh_clustering(self, embeddings: np.ndarray, node_ids: List[str]) -> List[List[str]]:
        """
        Perform LSH clustering on embeddings with detailed statistics.
        """
        logger.info(f"LSH clustering {len(embeddings)} embeddings")
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        n_samples, dim = embeddings.shape
        logger.info(f"📐 Embedding Dimension: {dim}, Number of Hyperplanes: {self.lsh_num_hyperplanes}")
        
        # Generate random hyperplanes for LSH
        hyperplanes = np.random.randn(self.lsh_num_hyperplanes, dim)
        
        def get_bucket_id(vec):
            projections = np.dot(vec, hyperplanes.T)
            binary_hash = (projections > 0).astype(int)
            return int(''.join(map(str, binary_hash)), 2)
        
        # Create LSH buckets
        buckets = defaultdict(list)
        for i, embedding in enumerate(embeddings):
            bucket_id = get_bucket_id(embedding)
            buckets[bucket_id].append(node_ids[i])
        
        # Print initial bucket statistics
        self._print_bucket_stats(buckets, "Initial LSH Buckets")
        
        # Process buckets to create final clusters
        clusters = []
        sorted_buckets = sorted(buckets.items(), key=lambda x: x[0])
        queue = deque([(bid, nodes) for bid, nodes in sorted_buckets])
        current_cluster = []
        
        while queue:
            bid, nodes = queue.popleft()
            
            # If bucket is too large, split it
            if len(nodes) >= self.lsh_max_cluster_size:
                clusters.append(nodes[:self.lsh_max_cluster_size])
                remaining = nodes[self.lsh_max_cluster_size:]
                if remaining:
                    queue.appendleft((bid, remaining))
                continue
            
            # Try to merge with current cluster
            if len(current_cluster) + len(nodes) <= self.lsh_max_cluster_size:
                current_cluster.extend(nodes)
            else:
                # Finalize current cluster if it meets minimum size
                if len(current_cluster) >= self.lsh_min_cluster_size:
                    clusters.append(current_cluster)
                
                # Start new cluster
                current_cluster = nodes.copy()
        
        # Handle final cluster
        if len(current_cluster) >= self.lsh_min_cluster_size:
            clusters.append(current_cluster)
        
        # Print final clustering statistics
        self._print_cluster_stats(clusters, f"LSH clustering (minimum size ≥{self.lsh_min_cluster_size})")
        
        logger.info(f"✅ LSH clustering completed, generated {len(clusters)} clusters")
        return clusters

    async def _generate_community_summary_and_embedding(self, community_id: str, member_nodes: List[str], level: int):
        """
        Generate summary and embedding for a community.
        Enhanced to include chunk content, entity names, and entity-entity relationships.
        """
        # Collect different types of content
        chunk_contents = []
        entity_names = []
        entity_relationships = []
        member_embeddings = []
        
        # Step 1: Collect chunk contents and entity names
        for node_id in member_nodes:
            if level == 0:
                # Base level: get node content
                if node_id.startswith('CHUNK_'):
                    # Chunk node - collect original content
                    chunk_key = node_id.replace('CHUNK_', '')
                    chunk_data = await self._get_chunk_original_content(chunk_key, node_id)
                    if chunk_data:
                        chunk_contents.append(f"Document Chunk {chunk_key}: {chunk_data}")  #TODO [:1200]
                else:
                    # Entity node - collect name and description
                    entity_data = await self._get_entity_name_and_desc(node_id)
                    if entity_data:
                        entity_names.append(entity_data)
                
                # Get node embedding
                if node_id in self.node_embeddings:
                    member_embeddings.append(self.node_embeddings[node_id])
            
            else:
                # Higher level: get community summaries
                if node_id in self.community_summaries:
                    # For higher levels, we use the existing summaries
                    chunk_contents.append(self.community_summaries[node_id])
                
                if node_id in self.node_embeddings:
                    member_embeddings.append(self.node_embeddings[node_id])
        
        # Step 2: Collect entity-entity relationships (only for level 0)
        if level == 0:
            entity_relationships = await self._collect_entity_entity_relationships(member_nodes)
        
        # Step 3: Generate enhanced summary using LLM
        summary = await self._generate_enhanced_community_summary(
            community_id, chunk_contents, entity_names, entity_relationships, level
        )
        
        # Step 4: Generate community embedding based on community summary text
        # (Similar to how leaf nodes get embeddings from their text content)
        if community_id in self.community_summaries:
            community_summary = self.community_summaries[community_id]
            if community_summary and community_summary.strip():
                try:
                    # Generate embedding from community summary text using embedding model
                    community_embedding = await self._embed_text(community_summary)
                    
                    # Ensure it's a numpy array and normalize
                    if not isinstance(community_embedding, np.ndarray):
                        community_embedding = np.array(community_embedding)
                    
                    # Normalize the embedding
                    norm = np.linalg.norm(community_embedding)
                    if norm > 0:
                        community_embedding = community_embedding / norm
                        self.node_embeddings[community_id] = community_embedding
                        self.node_text_embeddings[community_id] = community_embedding
                        logger.info(f"Generated text-based embedding for {community_id} from summary ({len(community_summary)} chars)")
                    else:
                        logger.warning(f"Zero norm embedding for {community_id}, using fallback")
                        # Fallback: use mean of member embeddings
                        if member_embeddings:
                            community_embedding = np.mean(member_embeddings, axis=0)
                            community_embedding = community_embedding / np.linalg.norm(community_embedding)
                            self.node_embeddings[community_id] = community_embedding
                            self.node_text_embeddings[community_id] = community_embedding
                                
                except Exception as e:
                    logger.warning(f"Failed to generate text-based embedding for {community_id}: {e}")
                    # Fallback: use mean of member embeddings
                    if member_embeddings:
                        community_embedding = np.mean(member_embeddings, axis=0)
                        community_embedding = community_embedding / np.linalg.norm(community_embedding)
                        self.node_embeddings[community_id] = community_embedding
                        self.node_text_embeddings[community_id] = community_embedding
            else:
                logger.warning(f"No valid summary for {community_id}, using member embedding average")
                # Fallback: use mean of member embeddings
                if member_embeddings:
                    community_embedding = np.mean(member_embeddings, axis=0)
                    community_embedding = community_embedding / np.linalg.norm(community_embedding)
                    self.node_embeddings[community_id] = community_embedding
                    self.node_text_embeddings[community_id] = community_embedding

        logger.info(f"Generated enhanced summary and text-based embedding for {community_id}")
    
    async def _get_chunk_original_content(self, chunk_key: str, node_id: str) -> str:
        """
        Get the original content of a chunk.
        """
        chunk_data = None
        
        # Try to get chunk data from doc_chunk
        if hasattr(self, 'doc_chunk') and self.doc_chunk is not None:
            try:
                chunk_data = await self.doc_chunk.get_data_by_key(chunk_key)
            except Exception as e:
                logger.warning(f"Failed to get chunk data for {chunk_key}: {e}")
        
        # Fallback: get data from graph node
        if not chunk_data:
            try:
                node_data = await self._graph.get_node(node_id)
                if node_data:
                    chunk_data = node_data.get('description', f'Chunk {chunk_key}')
            except Exception as e:
                logger.warning(f"Failed to get node data for {node_id}: {e}")
                chunk_data = f'Chunk {chunk_key}'
        
        return chunk_data or f'Chunk {chunk_key}'
    
    async def _get_entity_name_and_desc(self, entity_node_id: str) -> str:
        """
        Get entity name and description.
        """
        try:
            entity_data = await self._graph.get_node(entity_node_id)
            if entity_data:
                entity_name = entity_data.get('entity_name', entity_node_id)
                entity_desc = entity_data.get('description', '')
                entity_type = entity_data.get('entity_type', '')
                
                # Format entity information
                if entity_desc:
                    return f"Entity: {entity_name} ({entity_type}). Description: {entity_desc}"
                else:
                    return f"Entity: {entity_name} ({entity_type})"
        except Exception as e:
            logger.warning(f"Failed to get entity data for {entity_node_id}: {e}")
        
        return f"Entity: {entity_node_id}"
    
    async def _collect_entity_entity_relationships(self, member_nodes: List[str]) -> List[str]:
        """
        Collect entity-entity relationships within the community.
        Only considers edges between entity nodes (not chunk nodes).
        """
        relationships = []
        
        # Filter out entity nodes (exclude chunk nodes and community nodes)
        entity_nodes = [
            node for node in member_nodes 
            if not node.startswith('CHUNK_') and not node.startswith('COMMUNITY_')
        ]
        
        if len(entity_nodes) < 2:
            return relationships
        
        logger.info(f"Collecting relationships among {len(entity_nodes)} entities")
        
        # Check relationships between each pair of entities
        for i, src_entity in enumerate(entity_nodes):
            try:
                # Get all edges from this entity
                if hasattr(self._graph, 'get_node_edges'):
                    node_edges = await self._graph.get_node_edges(src_entity)
                    if node_edges:
                        for edge_tuple in node_edges:
                            try:
                                if len(edge_tuple) >= 2:
                                    # Determine target entity
                                    tgt_entity = edge_tuple[1] if edge_tuple[0] == src_entity else edge_tuple[0]
                                    
                                    # Check if target is also in our entity list
                                    if tgt_entity in entity_nodes and tgt_entity != src_entity:
                                        # Get edge data
                                        edge_data = await self._graph.get_edge(src_entity, tgt_entity)
                                        if edge_data:
                                            relation_name = edge_data.get('relation_name', 'CONNECTED')
                                            relation_desc = edge_data.get('description', '')
                                            
                                            # Get entity names for better readability
                                            src_name = await self._get_entity_display_name(src_entity)
                                            tgt_name = await self._get_entity_display_name(tgt_entity)
                                            
                                            # Format relationship
                                            if relation_desc:
                                                rel_text = f"{src_name} --[{relation_name}]--> {tgt_name}. {relation_desc}"
                                            else:
                                                rel_text = f"{src_name} --[{relation_name}]--> {tgt_name}"
                                            
                                            relationships.append(rel_text)
                            except Exception as edge_error:
                                logger.warning(f"Error processing edge {edge_tuple}: {edge_error}")
                                continue
            except Exception as node_error:
                logger.warning(f"Error getting edges for entity {src_entity}: {node_error}")
                continue
        
        # Remove duplicates (since we might add the same relationship twice in undirected graphs)
        unique_relationships = list(set(relationships))
        logger.info(f"Found {len(unique_relationships)} unique entity-entity relationships")
        
        return unique_relationships
    
    async def _get_entity_display_name(self, entity_node_id: str) -> str:
        """
        Get the display name of an entity for relationship descriptions.
        """
        try:
            entity_data = await self._graph.get_node(entity_node_id)
            if entity_data:
                return entity_data.get('entity_name', entity_node_id)
        except Exception:
            pass
        return entity_node_id
    
    async def _generate_enhanced_community_summary(self, community_id: str, chunk_contents: List[str], 
                                                  entity_names: List[str], entity_relationships: List[str], 
                                                  level: int) -> str:
        """
        Generate enhanced community summary considering chunks, entities, and their relationships.
        This summary will be used to generate the community embedding via embedding model.
        """
        summary_parts = []
        
        # Add chunk contents
        if chunk_contents:
            summary_parts.append("=== Document Contents ===")
            summary_parts.extend(chunk_contents)
        
        # Add entity information
        if entity_names:
            summary_parts.append("\n=== Entities ===")
            summary_parts.extend(entity_names)
        
        # Add entity relationships
        if entity_relationships:
            summary_parts.append("\n=== Entity Relationships ===")
            summary_parts.extend(entity_relationships)
        
        if not summary_parts:
            fallback_summary = f"Community at level {level} with {len(entity_names + chunk_contents)} members"
            self.community_summaries[community_id] = fallback_summary
            return fallback_summary
        
        combined_content = "\n".join(summary_parts)
        
        # Enhanced prompt that specifically considers chunks, entities, and relationships
        # The summary will be used for embedding generation, so it should be comprehensive
        summary_prompt = f"""
Please provide a comprehensive summary of this knowledge community that includes:
1. Key themes and topics from the document contents
2. Important entities and their roles
3. Relationships and connections between entities
4. Overall context and significance

Content to summarize:
{combined_content}

Provide a concise but comprehensive summary (max {self.community_summary_length} words) that captures the main themes, entities, and their interconnections. This summary will be used to generate semantic embeddings, so ensure it contains rich semantic information:
"""
        
        try:
            summary = await self.llm.aask(summary_prompt, max_tokens=self.community_summary_length)
            # Ensure the summary is not empty and has meaningful content
            if summary and summary.strip():
                self.community_summaries[community_id] = summary.strip()
                logger.info(f"Generated enhanced summary for {community_id}: {len(summary)} characters")
            else:
                fallback_summary = f"Community at level {level} with entities and relationships from {len(entity_names + chunk_contents)} members"
                self.community_summaries[community_id] = fallback_summary
                logger.warning(f"Empty summary generated for {community_id}, using fallback")
        except Exception as e:
            logger.warning(f"Failed to generate enhanced summary for {community_id}: {e}")
            fallback_summary = f"Community at level {level} with entities and relationships from {len(entity_names + chunk_contents)} members"
            self.community_summaries[community_id] = fallback_summary
        
        return self.community_summaries[community_id]

    async def _initialize_text_embeddings(self, nodes: List[str], graph: nx.Graph) -> np.ndarray:
        """
        Initialize node embeddings using text content instead of random initialization.
        """
        n_nodes = len(nodes)
        embeddings = np.zeros((n_nodes, self.cleora_dim))
        
        for i, node_id in enumerate(nodes):
            # Get node data from the homogeneous graph
            node_data = graph.nodes[node_id]
            
            # Extract text content based on node type
            if node_id.startswith('CHUNK_'):
                # For chunk nodes, use the description (which contains full content)
                text_content = node_data.get('description', '')
                if not text_content:
                    # Fallback: try to get from original graph
                    original_node_data = await self._graph.get_node(node_id)
                    text_content = original_node_data.get('description', '') if original_node_data else ''
            else:
                # For entity nodes, combine name and description
                entity_name = node_data.get('name', node_id)
                entity_desc = node_data.get('description', '')
                text_content = f"{entity_name}. {entity_desc}".strip()
                if not text_content:
                    # Fallback: try to get from original graph
                    original_node_data = await self._graph.get_node(node_id)
                    if original_node_data:
                        entity_name = original_node_data.get('entity_name', node_id)
                        entity_desc = original_node_data.get('description', '')
                        text_content = f"{entity_name}. {entity_desc}".strip()
            
            # Generate embedding using the encoder
            if text_content:
                try:
                    # Get text embedding
                    text_embedding = await self._embed_text(text_content)
                    
                    # Handle dimension mismatch
                    if len(text_embedding) == self.cleora_dim:
                        embeddings[i] = text_embedding
                    elif len(text_embedding) > self.cleora_dim:
                        # Truncate to fit cleora_dim
                        embeddings[i] = text_embedding[:self.cleora_dim]
                    else:
                        # Pad with zeros if embedding is smaller
                        embeddings[i, :len(text_embedding)] = text_embedding
                        
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for node {node_id}: {e}")
                    # Fallback to random initialization for this node
                    embeddings[i] = np.random.normal(0, 0.1, self.cleora_dim)
            else:
                logger.warning(f"No text content found for node {node_id}, using random initialization")
                # Fallback to random initialization
                embeddings[i] = np.random.normal(0, 0.1, self.cleora_dim)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        logger.info(f"Initialized {n_nodes} node embeddings using text content")
        return embeddings

    async def get_hierarchy_info(self) -> Dict[str, Any]: #TODO 打印层次信息debug
        """
        Get information about the built hierarchy.
        """
        logger.debug(f"🔍 Hierarchy Info Debug:")
        logger.debug(f"🔍- hierarchy_levels: {len(self.hierarchy_levels)} levels")
        logger.debug(f"🔍- node_embeddings: {len(self.node_embeddings)} embeddings")
        logger.debug(f"🔍- community_summaries: {len(self.community_summaries)} summaries")
        
        if self.hierarchy_levels:
            for level, communities in self.hierarchy_levels.items():
                logger.debug(f"🔍- Level {level}: {len(communities)} communities")
        else:
            logger.warning("⚠️ No hierarchy levels found!")
            
        return {
            'levels': len(self.hierarchy_levels),
            'hierarchy_levels': self.hierarchy_levels,
            'community_summaries': self.community_summaries,
            'community_children': self.community_children,
            'community_parents': self.community_parents,
            'total_embeddings': len(self.node_embeddings)
        }

    async def _embed_text(self, text: str):
        return self.embedding_model._get_text_embedding(text)
    
    async def get_community_children(self, community_id: str) -> set:
        """
        Get direct children of a community (similar to EraRAG's tree structure).
        """
        return self.community_children.get(community_id, set())
    
    async def get_community_parent(self, node_id: str) -> str:
        """
        Get the parent community of a node or community.
        """
        return self.community_parents.get(node_id, None)
    
    async def is_leaf_node(self, node_id: str) -> bool:
        """
        Check if a node is a leaf node (has no children).
        Similar to EraRAG's leaf node detection.
        """
        if node_id.startswith('COMMUNITY_'):
            # Community nodes are not leaf nodes
            return False
        # Base nodes (entities and chunks) are leaf nodes
        return True
    
    async def get_all_descendants(self, community_id: str) -> set:
        """
        Get all descendant nodes of a community (recursive).
        Enables top-down traversal like EraRAG.
        """
        descendants = set()
        direct_children = await self.get_community_children(community_id)
        
        for child in direct_children:
            descendants.add(child)
            if child.startswith('COMMUNITY_'):
                # If child is also a community, get its descendants
                child_descendants = await self.get_all_descendants(child)
                descendants.update(child_descendants)
        
        return descendants
    
    async def get_nodes_by_level(self, level: int) -> List[str]:
        """
        Get all community nodes at a specific level.
        """
        if level not in self.hierarchy_levels:
            return []
        
        return [community['id'] for community in self.hierarchy_levels[level]]
    
    async def _persist_graph(self, force: bool = False):

        await self._graph.persist(force)

        if (self.hierarchy_levels or self.node_embeddings or 
            self.community_summaries or self.community_children):
            await self._save_hierarchy_to_storage(force=force)

        if any(index is not None for index in self.faiss_indexes.values()):
            await self._save_faiss_indexes()

    # ===== FAISS Index Management Methods =====
    
    async def _build_faiss_index(self):
        """
        Three independent FAISS vector indexes are constructed, corresponding to entity, chunk, and community nodes. 
        Each node type uses its own FAISS index to improve search efficiency.
        """
        try:
            logger.info(f"🔍 Building separate FAISS indexes for {len(self.node_text_embeddings)} nodes")
            
            if not self.node_text_embeddings:
                logger.warning("⚠️ No node embeddings available for FAISS index")
                return

            node_groups = {
                'entity': [],
                'chunk': [],
                'community': []
            }
            
            for node_id in self.node_text_embeddings.keys():
                if node_id.startswith('CHUNK_'):
                    node_groups['chunk'].append(node_id)
                elif node_id.startswith('COMMUNITY_'):
                    node_groups['community'].append(node_id)
                else:
                    node_groups['entity'].append(node_id)

            for node_type in node_groups:
                node_groups[node_type].sort()

            for node_type, node_ids in node_groups.items():
                if not node_ids:
                    logger.info(f"🔸 No {node_type} nodes found, skipping index creation")
                    continue
                
                logger.info(f"🏗️ Building FAISS index for {len(node_ids)} {node_type} nodes")

                embeddings_list = []
                valid_node_ids = []
                
                for node_id in node_ids:
                    embedding = self.node_text_embeddings.get(node_id)
                    if embedding is not None and len(embedding) > 0:
                        embeddings_list.append(embedding)
                        valid_node_ids.append(node_id)
                
                if not embeddings_list:
                    logger.warning(f"⚠️ No valid embeddings found for {node_type} nodes")
                    continue

                embeddings_matrix = np.array(embeddings_list).astype('float32')
                n_vectors, embedding_dim = embeddings_matrix.shape
                
                logger.info(f"📊 {node_type} index: {n_vectors} vectors, {embedding_dim} dimensions")

                index = await self._create_single_faiss_index(embedding_dim, node_type)
                
                if index is None:
                    logger.warning(f"⚠️ Failed to create FAISS index for {node_type}")
                    continue

                index.add(embeddings_matrix)

                self.faiss_indexes[node_type] = index
                self.faiss_id_to_node[node_type] = {i: node_id for i, node_id in enumerate(valid_node_ids)}
                self.node_to_faiss_id[node_type] = {node_id: i for i, node_id in enumerate(valid_node_ids)}
                
                logger.info(f"✅ {node_type} FAISS index built successfully: {index.ntotal} vectors")

            total_vectors = sum(index.ntotal if index else 0 for index in self.faiss_indexes.values())
            active_indexes = sum(1 for index in self.faiss_indexes.values() if index is not None)
            
            logger.info(f"🎯 All FAISS indexes built successfully:")
            logger.info(f"   📊 Active indexes: {active_indexes}/3")
            logger.info(f"   📊 Total vectors: {total_vectors}")
            logger.info(f"   🏗️ Index type: {self.faiss_index_type}")
            logger.info(f"   📦 Chunk vectors: {self.faiss_indexes['chunk'].ntotal if self.faiss_indexes['chunk'] else 0}")
            logger.info(f"   🏷️ Entity vectors: {self.faiss_indexes['entity'].ntotal if self.faiss_indexes['entity'] else 0}")
            logger.info(f"   🏛️ Community vectors: {self.faiss_indexes['community'].ntotal if self.faiss_indexes['community'] else 0}")
            
        except Exception as e:
            logger.error(f"❌ Failed to build FAISS indexes: {e}")
            raise

    async def _create_single_faiss_index(self, embedding_dim: int, node_type: str):
        """
        Creates a single FAISS index for a specific node type.

        Args:
        embedding_dim: Embedding vector dimension
        node_type: Node type ('entity', 'chunk', 'community')

        Returns:
        FAISS index object or None
        """
        try:

            if self.faiss_index_type.upper() == 'HNSW':

                index = faiss.IndexHNSWFlat(embedding_dim, self.faiss_hnsw_m)
                index.hnsw.efConstruction = self.faiss_hnsw_ef_construction
                index.hnsw.efSearch = self.faiss_hnsw_ef_search
                logger.debug(f"🏗️ Created HNSW index for {node_type}: M={self.faiss_hnsw_m}, efConstruction={self.faiss_hnsw_ef_construction}")
                
            elif self.faiss_index_type.upper() == 'HNSW_IP':

                try:
                    #base_index = faiss.IndexFlatIP(embedding_dim)
                    index = faiss.index_factory(embedding_dim, f"HNSW{self.faiss_hnsw_m}_FLAT", faiss.METRIC_INNER_PRODUCT)
                    #index = faiss.IndexHNSW(base_index, self.faiss_hnsw_m)
                    index.hnsw.efConstruction = self.faiss_hnsw_ef_construction
                    index.hnsw.efSearch = self.faiss_hnsw_ef_search
                    logger.debug(f"🏗️ Created HNSW+IP index for {node_type}")
                except Exception as e:
                    logger.warning(f"Failed to create HNSW_IP index for {node_type}: {e}, falling back to FLAT IP")
                    index = faiss.IndexFlatIP(embedding_dim)
                
            elif self.faiss_index_type.upper() == 'FLAT':
                index = faiss.IndexFlatIP(embedding_dim)
                logger.debug(f"🏗️ Created Flat IP index for {node_type}")
                
            elif self.faiss_index_type.upper() == 'FLAT_L2':
                index = faiss.IndexFlatL2(embedding_dim)
                logger.debug(f"🏗️ Created Flat L2 index for {node_type}")
                
            else:
                index = faiss.IndexHNSWFlat(embedding_dim, self.faiss_hnsw_m)
                index.hnsw.efConstruction = self.faiss_hnsw_ef_construction
                index.hnsw.efSearch = self.faiss_hnsw_ef_search
                logger.warning(f"⚠️ Unknown index type {self.faiss_index_type}, using HNSW+L2 for {node_type}")
            
            return index
            
        except Exception as e:
            logger.error(f"❌ Failed to create FAISS index for {node_type}: {e}")
            return None

    async def _save_faiss_indexes(self):

        try:
            active_indexes = {node_type: index for node_type, index in self.faiss_indexes.items() if index is not None}
            
            if not active_indexes:
                logger.warning("⚠️ No FAISS indexes to save")
                return
            
            logger.info(f"💾 Saving {len(active_indexes)} FAISS indexes")

            index_dir = self.faiss_index_path
            os.makedirs(index_dir, exist_ok=True)

            saved_indexes = {}
            total_vectors = 0
            
            for node_type, index in active_indexes.items():

                if index.ntotal == 0:
                    logger.warning(f"⚠️ {node_type} FAISS index is empty (ntotal=0), skipping save")
                    continue

                if hasattr(index, 'is_trained') and not index.is_trained:
                    logger.warning(f"⚠️ {node_type} FAISS index is not trained, skipping save")
                    continue

                index_file = os.path.join(index_dir, f"faiss_index_{node_type}.index")
                faiss.write_index(index, index_file)
                
                saved_indexes[node_type] = {
                    'index_file': index_file,
                    'total_vectors': index.ntotal,
                    'embedding_dim': index.d
                }
                total_vectors += index.ntotal
                
                logger.debug(f"📄 Saved {node_type} index: {index.ntotal} vectors to {index_file}")
            
            if not saved_indexes:
                logger.warning("⚠️ No valid FAISS indexes were saved")
                return

            mapping_file = os.path.join(index_dir, "faiss_mappings_multi.pkl")
            mapping_data = {
                'faiss_id_to_node': self.faiss_id_to_node,
                'node_to_faiss_id': self.node_to_faiss_id,
                'index_type': self.faiss_index_type,
                'saved_indexes': saved_indexes,
                'total_vectors': total_vectors
            }
            
            with open(mapping_file, 'wb') as f:
                pickle.dump(mapping_data, f)

            config_file = os.path.join(index_dir, "faiss_config_multi.pkl")
            config_data = {
                'faiss_index_type': self.faiss_index_type,
                'faiss_hnsw_m': self.faiss_hnsw_m,
                'faiss_hnsw_ef_construction': self.faiss_hnsw_ef_construction,
                'faiss_hnsw_ef_search': self.faiss_hnsw_ef_search,
                'node_types': list(saved_indexes.keys())
            }
            
            with open(config_file, 'wb') as f:
                pickle.dump(config_data, f)
            
            logger.info(f"✅ FAISS indexes saved to {index_dir}")
            logger.info(f"   📊 Saved indexes: {list(saved_indexes.keys())}")
            logger.info(f"   📊 Total vectors: {total_vectors}")
            logger.info(f"   📄 Mappings file: {mapping_file}")
            logger.info(f"   📄 Config file: {config_file}")
            for node_type, info in saved_indexes.items():
                logger.info(f"   📄 {node_type} index: {info['total_vectors']} vectors")
            
        except Exception as e:
            logger.error(f"❌ Failed to save FAISS indexes: {e}")
            raise

    async def _load_faiss_indexes(self) -> bool:

        try:
            index_dir = self.faiss_index_path

            mapping_file = os.path.join(index_dir, "faiss_mappings_multi.pkl")
            config_file = os.path.join(index_dir, "faiss_config_multi.pkl")
            
            if not os.path.exists(mapping_file):
                logger.info("📂 Multi-index FAISS files not found, will build new indexes")
                return False

            with open(mapping_file, 'rb') as f:
                mapping_data = pickle.load(f)
                self.faiss_id_to_node = mapping_data['faiss_id_to_node']
                self.node_to_faiss_id = mapping_data['node_to_faiss_id']
                saved_indexes_info = mapping_data.get('saved_indexes', {})

            if os.path.exists(config_file):
                with open(config_file, 'rb') as f:
                    config_data = pickle.load(f)
                    expected_node_types = config_data.get('node_types', [])
                    logger.debug(f"📋 Expected node types: {expected_node_types}")

            loaded_indexes = {}
            total_vectors = 0
            
            for node_type in ['entity', 'chunk', 'community']:
                index_file = os.path.join(index_dir, f"faiss_index_{node_type}.index")
                
                if not os.path.exists(index_file):
                    logger.debug(f"📂 {node_type} index file not found, skipping")
                    continue
                
                try:
                    index = faiss.read_index(index_file)

                    expected_size = len(self.faiss_id_to_node.get(node_type, {}))
                    if index.ntotal != expected_size:
                        logger.warning(f"⚠️ {node_type} index size mismatch: {index.ntotal} vs {expected_size}")
                        continue

                    if hasattr(index, 'hnsw'):
                        index.hnsw.efSearch = self.faiss_hnsw_ef_search
                    
                    self.faiss_indexes[node_type] = index
                    loaded_indexes[node_type] = index.ntotal
                    total_vectors += index.ntotal
                    
                    logger.debug(f"✅ Loaded {node_type} index: {index.ntotal} vectors")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {node_type} index: {e}")
                    continue
            
            if not loaded_indexes:
                logger.warning("⚠️ No FAISS indexes were successfully loaded")
                return False
            
            logger.info(f"✅ FAISS indexes loaded successfully:")
            logger.info(f"   📊 Loaded indexes: {list(loaded_indexes.keys())}")
            logger.info(f"   📊 Total vectors: {total_vectors}")
            for node_type, count in loaded_indexes.items():
                logger.info(f"   🔹 {node_type}: {count} vectors")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load FAISS indexes: {e}")
            return False

    async def get_node_info_by_ids(self, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed node information based on a list of node IDs.
        Used to retrieve node information after a FAISS search.

        Args:
        node_ids: List of node IDs

        Returns:
        Dict[node_id, node_info]: Node information dictionary.
        """
        node_info_dict = {}
        
        for node_id in node_ids:
            try:
                node_info = {
                    'node_id': node_id,
                    'node_type': self._get_node_type(node_id),
                    'embedding': self.node_embeddings.get(node_id),
                    'text_embedding': self.node_text_embeddings.get(node_id)
                }
                
                if node_id.startswith('CHUNK_'):
                    # Chunk节点信息
                    chunk_key = node_id.replace('CHUNK_', '')
                    chunk_content = await self._get_chunk_original_content(chunk_key, node_id)
                    node_info.update({
                        'chunk_key': chunk_key,
                        'content': chunk_content,
                        'content_type': 'chunk'
                    })
                    
                elif node_id.startswith('COMMUNITY_'):
                    node_info.update({
                        'summary': self.community_summaries.get(node_id, ''),
                        'children': list(self.community_children.get(node_id, set())),
                        'parent': self.community_parents.get(node_id),
                        'content': self.community_summaries.get(node_id, ''),
                        'content_type': 'community'
                    })

                    level_match = re.search(r'COMMUNITY_L(\d+)_C\d+', node_id)
                    if level_match:
                        node_info['level'] = int(level_match.group(1))
                    
                else:
                    entity_data = await self._graph.get_node(node_id)
                    if entity_data:
                        node_info.update({
                            'entity_name': entity_data.get('entity_name', node_id),
                            'entity_type': entity_data.get('entity_type', ''),
                            'description': entity_data.get('description', ''),
                            'source_id': entity_data.get('source_id', ''),
                            'content': f"{entity_data.get('entity_name', node_id)}. {entity_data.get('description', '')}".strip(),
                            'content_type': 'entity'
                        })
                    else:
                        node_info.update({
                            'entity_name': node_id,
                            'content': node_id,
                            'content_type': 'entity'
                        })
                
                node_info_dict[node_id] = node_info
                
            except Exception as e:
                logger.warning(f"Failed to get info for node {node_id}: {e}")
                node_info_dict[node_id] = {
                    'node_id': node_id,
                    'node_type': self._get_node_type(node_id),
                    'content': node_id,
                    'content_type': 'unknown',
                    'error': str(e)
                }
        
        return node_info_dict

    def _get_node_type(self, node_id: str) -> str:
        """Get the type of node"""
        if node_id.startswith('CHUNK_'):
            return 'chunk'
        elif node_id.startswith('COMMUNITY_'):
            return 'community'
        else:
            return 'entity'

    async def search_similar_entities(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Searches for the most similar entity nodes using the entity-specific FAISS index.

        Args:
        query_embedding: Query vector
        top_k: Returns the top k results

        Returns:
        List[(node_id, similarity_score)]: List of similar entity nodes
        """
        return await self._search_single_index('entity', query_embedding, top_k)
    
    async def search_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Searches for the most similar chunk nodes using a chunk-specific FAISS index.

        Args:
        query_embedding: Query vector
        top_k: Returns the top k results

        Returns:
        List[(node_id, similarity_score)]: List of similar chunk nodes
        """
        return await self._search_single_index('chunk', query_embedding, top_k)
    
    async def search_similar_communities(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Searches for the most similar community nodes using the community-specific FAISS index.

        Args:
        query_embedding: Query vector
        top_k: Returns the top k results

        Returns:
        List[(node_id, similarity_score)]: List of similar community nodes
        """
        return await self._search_single_index('community', query_embedding, top_k)
    
    async def _search_single_index(self, node_type: str, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """
        Internal method for searching for similar nodes in a single FAISS index.

        Args:
        node_type: Node type ('entity', 'chunk', 'community')
        query_embedding: Query embedding
        top_k: Returns the top k results.

        Returns:
        List[(node_id, similarity_score)]: List of similar nodes
        """
        if node_type not in self.faiss_indexes:
            logger.error(f"❌ Invalid node type: {node_type}")
            return []
        
        index = self.faiss_indexes[node_type]
        if index is None:
            logger.warning(f"⚠️ {node_type} FAISS index not available")
            return []
        
        try:
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
            
            query_embedding = query_embedding.astype('float32').reshape(1, -1)

            search_k = min(top_k , index.ntotal)
            distances, indices = index.search(query_embedding, search_k)

            results = []
            id_mapping = self.faiss_id_to_node[node_type]
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  
                    continue
                    
                node_id = id_mapping.get(idx)
                if node_id is None:
                    continue

                similarity_score = self._convert_distance_to_similarity(float(dist))
                results.append((node_id, similarity_score))

            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]
            
            logger.debug(f"🔍 Found {len(results)} similar {node_type} nodes (requested: {top_k})")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search {node_type} index: {e}")
            return []
    
    def _convert_distance_to_similarity(self, distance_or_score: float) -> float:
        """
        Intelligently converts distances/scores to a unified similarity score.

        Args:
        distance_or_score: The original value returned by FAISS.

        Returns:
        similarity: The normalized similarity score (larger means more similarity), in the range [0, 1].
        """
        if self.faiss_index_type.upper() in ['FLAT', 'HNSW_IP']:
            if distance_or_score < 0:
                similarity = 1.0 / (1.0 + np.exp(-distance_or_score))
            else:
                similarity = np.tanh(distance_or_score)
            
        elif self.faiss_index_type.upper() in ['HNSW', 'FLAT_L2']:
            similarity = np.exp(-distance_or_score)
            
        else:
            logger.warning(f"Unknown index type: {self.faiss_index_type}, attempting auto-detection")

            if distance_or_score < 0 or distance_or_score > 2.0:
                similarity = 1.0 / (1.0 + np.exp(-distance_or_score))
            else:
                similarity = np.exp(-distance_or_score)

        similarity = max(0.0, min(1.0, float(similarity)))
        return similarity
    
    def _l2_distance_to_similarity(self, l2_distance: float) -> float:
        """
        Converts L2 distance to a similarity score (retained for backward compatibility)

        Args:
        l2_distance: L2 distance (smaller, more similar)

        Returns:
        similarity: Similarity score (larger, more similar), range [0, 1]
        """
        return self._convert_distance_to_similarity(l2_distance)

    async def get_faiss_stats(self) -> Dict[str, Any]:

        available_indexes = {nt: idx for nt, idx in self.faiss_indexes.items() if idx is not None}
        
        if not available_indexes:
            return {'status': 'not_built'}

        total_vectors = 0
        vector_dimensions = []
        
        for index in available_indexes.values():
            total_vectors += index.ntotal
            vector_dimensions.append(index.d)
        
        stats = {
            'status': 'ready',
            'total_vectors': total_vectors,
            'vector_dimension': vector_dimensions[0] if vector_dimensions else 0,
            'index_type': self.faiss_index_type,
            'active_indexes': list(available_indexes.keys()),
            'total_node_mappings': sum(len(mappings) for mappings in self.faiss_id_to_node.values()),
        }

        index_details = {}
        type_counts = {'chunk': 0, 'entity': 0, 'community': 0}
        
        for node_type, index in available_indexes.items():
            node_mappings = len(self.faiss_id_to_node.get(node_type, {}))
            type_counts[node_type] = node_mappings
            
            index_details[node_type] = {
                'total_vectors': index.ntotal,
                'vector_dimension': index.d,
                'node_mappings': node_mappings
            }

            if hasattr(index, 'hnsw'):
                index_details[node_type]['hnsw_stats'] = {
                    'M': self.faiss_hnsw_m,
                    'efConstruction': self.faiss_hnsw_ef_construction,
                    'efSearch': self.faiss_hnsw_ef_search,
                    'max_level': index.hnsw.max_level
                }
        
        stats['node_type_counts'] = type_counts
        stats['index_details'] = index_details
        
        return stats