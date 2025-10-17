import asyncio
import re
from collections import defaultdict
from typing import Any, List, Dict, Set, Tuple
from itertools import combinations

from Core.Graph.BaseGraph import BaseGraph
from Core.Common.Logger import logger
from Core.Common.Utils import clean_str, prase_json_from_response
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.Message import Message
from Core.Prompt import GraphPrompt
from Core.Prompt.Base import TextPrompt
from Core.Schema.EntityRelation import Entity, Relationship
from Core.Common.Constants import (
    NODE_PATTERN,
    REL_PATTERN,
    GRAPH_FIELD_SEP
)
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Utils.WAT import WATAnnotation
import requests
from Core.Common.Constants import GCUBE_TOKEN
from tqdm import tqdm


class HKGraph(BaseGraph):
    """
    HKGraph (Hybrid Knowledge Graph) combines the advantages of ER Graph and Passage Graph.
    
    Key Features:
    1. Entity-Entity connections: Like ER Graph, entities are connected through relationships
    2. Chunk-Chunk connections: Like Passage Graph, chunks are connected through shared entities
    3. Entity-Chunk connections: Entities are connected to their source chunks
    
    This creates a comprehensive hybrid knowledge graph that captures both:
    - Fine-grained entity relationships
    - Document-level semantic connections
    - Entity-document provenance relationships
    """

    def __init__(self, config, llm, encoder):
        super().__init__(config, llm, encoder)
        self._graph = NetworkXStorage()
        
        # Configuration for different extraction methods
        self.use_wat_linking = getattr(config, 'use_wat_linking', False)
        self.extract_two_step = getattr(config, 'extract_two_step', True)
        self.prior_prob = getattr(config, 'prior_prob', 0.8)

    async def _build_graph(self, chunk_list: List[Any]):
        """
        Build the hybrid knowledge graph combining ER and Passage graph approaches.
        """
        try:
            logger.info("Building Hybrid Knowledge Graph (HK Graph)")
            
            # Step 1: Extract entities and relationships from each chunk (ER Graph approach)
            logger.info("Step 1: Extracting entities and relationships from chunks")
            er_results = await asyncio.gather(
                *[self._extract_entity_relationship(chunk) for chunk in chunk_list]
            )
            
            # Step 2: Extract entity linking information for passage connections (Passage Graph approach)
            logger.info("Step 2: Extracting entity linking for passage connections")
            passage_results = await asyncio.gather(
                *[self._extract_passage_entities(chunk) for chunk in chunk_list]
            )
            
            # Step 3: Build the hybrid graph
            logger.info("Step 3: Building hybrid graph structure")
            await self._build_hybrid_graph(er_results, passage_results, chunk_list)
            
        except Exception as e:
            logger.exception(f"Error building HK graph: {e}")
        finally:
            logger.info("✅ HK Graph construction finished")

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
            logger.info("Using WAT entity linking system")
            wat_annotations = await self._wat_entity_linking(chunk_content)
            wiki_entities = await self._build_wiki_entities_from_wat(wat_annotations, chunk_key)
        else:
            # Use simple entity extraction
            logger.info("Using simple entity extraction")
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
        logger.info("Processing Entity-Entity connections")
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
        logger.info("Processing Passage entity linking")
        for passage_result in passage_results:
            chunk_key = passage_result['chunk_key']
            wiki_entities = passage_result['wiki_entities']
            
            for wiki_entity, _ in wiki_entities.items():
                wiki_entities_map[wiki_entity].append(chunk_key)
        
        # Step 3: Create Chunk nodes
        logger.info("Creating Chunk nodes")
        chunk_nodes = defaultdict(list)
        for chunk_key, chunk_info in chunk_list:
            chunk_entity = Entity(
                entity_name=f"CHUNK_{chunk_key}",
                entity_type="CHUNK",
                description=chunk_info.content,  # 保留完整的chunk内容
                source_id=chunk_key
            )
            chunk_nodes[f"CHUNK_{chunk_key}"].append(chunk_entity)
        
        # Step 4: Create Entity-Chunk connections
        logger.info("Creating Entity-Chunk connections")
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
        logger.info("Creating Chunk-Chunk connections")
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
        logger.info("Merging all nodes and edges into the graph")
        
        # Merge entity nodes
        all_nodes = {**all_entities, **chunk_nodes}
        await asyncio.gather(*[self._merge_nodes_then_upsert(k, v) for k, v in all_nodes.items()])
        
        # Merge all relationships
        all_edges = {**all_relationships, **entity_chunk_relationships, **chunk_chunk_relationships}
        await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in all_edges.items()])
        
        logger.info(f"✅ HK Graph built with {len(all_nodes)} nodes and {len(all_edges)} edges")

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
            entity = Entity(entity_name=entity_name, source_id=chunk_key)
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
                entity = Entity(entity_name=entity_name, entity_type=entity_type, source_id=chunk_key)
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
        return "entity_name"