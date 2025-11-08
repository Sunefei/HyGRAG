"""
Triple Extraction and Embedding-based Matching for HKGraphPPR
Enhanced triple extraction and semantic matching module
"""

import asyncio
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from Core.Common.Logger import logger
from Core.Common.Utils import prase_json_from_response, clean_str


@dataclass
class Triple:
    """Triple data structure"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    
    def __str__(self):
        return f"({self.subject}, {self.predicate}, {self.object})"
    
    def to_dict(self):
        return {
            "subject": self.subject,
            "predicate": self.predicate, 
            "object": self.object,
            "confidence": self.confidence
        }


@dataclass 
class MatchedSeed:
    """Information about matched seed entities/chunks"""
    original_text: str  # Original text in the query
    matched_entity: str  # Matched entity in the graph
    entity_type: str    # Entity type (entity/chunk)
    similarity_score: float  # Similarity score
    graph_node_id: str  # Node ID in the graph
    

class TripleExtractor:
    """Class for extracting triples from queries and performing semantic matching"""
    
    def __init__(self, llm, entities_vdb=None, graph=None, doc_chunk=None):
        self.llm = llm
        self.entities_vdb = entities_vdb
        self.graph = graph
        self.doc_chunk = doc_chunk
        
        # Triple extraction prompt template
        self.TRIPLE_EXTRACTION_PROMPT = """
**Task**: Extract structured triples (subject, predicate, object) from the given query

**Instructions**:
1. Identify entities and their relationships in the query
2. Represent relationships as (subject, predicate, object) triples
3. Ensure extracted triples fully express the semantic meaning of the query
4. Even for simple queries, try to construct meaningful triples
5. Use "?" as placeholder for unknown entities that the query is asking about

**Examples**:
Query: "What is the capital of France?"
Triples: [("France", "has_capital", "?"), ("?", "is_capital_of", "France")]

Query: "Who directed the movie Inception?"
Triples: [("?", "directed", "Inception"), ("Inception", "directed_by", "?")]

Query: "Show me documents about machine learning algorithms"
Triples: [("documents", "about", "machine learning algorithms"), ("machine learning algorithms", "mentioned_in", "documents")]

Query: "How are neural networks related to deep learning?"
Triples: [("neural networks", "related_to", "deep learning"), ("deep learning", "uses", "neural networks")]

**Output Format**: 
Return JSON format with triples field, each triple contains subject, predicate, object, confidence fields
```json
{
    "triples": [
        {
            "subject": "subject_entity",
            "predicate": "relationship", 
            "object": "object_entity",
            "confidence": 0.9
        }
    ]
}
```

**Query**: {query}

Please extract triples:
"""

        self.SIMPLE_ENTITY_PROMPT = """
**Task**: Extract key entities from the query

**Instructions**:
1. Identify important entities in the query (person names, places, organizations, concepts, etc.)
2. Extract keywords that can be used for retrieval
3. Focus on entities that are likely to appear in documents

**Examples**:
Query: "What is the capital of France?"
Entities: ["France", "capital"]

Query: "Who directed the movie Inception?"
Entities: ["Inception", "movie", "director"]

Query: "Show me documents about machine learning algorithms"
Entities: ["machine learning", "algorithms", "documents"]

**Output Format**:
```json
{
    "entities": ["entity1", "entity2", "entity3"]
}
```

**Query**: {query}

Please extract entities:
"""

    async def extract_triples_from_query(self, query: str) -> List[Triple]:
        """
        Extract triples from the given query
        
        Args:
            query: User query
            
        Returns:
            List[Triple]: List of extracted triples
        """
        logger.info(f"🔍 Starting triple extraction from query: {query[:100]}...")
        
        try:
            # Use LLM to extract triples
            prompt = self.TRIPLE_EXTRACTION_PROMPT.format(query=query)
            response = await self.llm.aask(prompt, format="json")
            
            triples = []
            
            # Handle different response types
            if isinstance(response, dict):
                if 'triples' in response:
                    for triple_data in response['triples']:
                        triple = Triple(
                            subject=clean_str(triple_data.get('subject', '')),
                            predicate=clean_str(triple_data.get('predicate', '')),
                            object=clean_str(triple_data.get('object', '')),
                            confidence=float(triple_data.get('confidence', 1.0))
                        )
                        triples.append(triple)
            elif isinstance(response, str):
                # Try to parse string format JSON
                try:
                    import json
                    parsed_response = json.loads(response)
                    if 'triples' in parsed_response:
                        for triple_data in parsed_response['triples']:
                            triple = Triple(
                                subject=clean_str(triple_data.get('subject', '')),
                                predicate=clean_str(triple_data.get('predicate', '')),
                                object=clean_str(triple_data.get('object', '')),
                                confidence=float(triple_data.get('confidence', 1.0))
                            )
                            triples.append(triple)
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ JSON parsing failed for response: {response[:100]}...")
            else:
                logger.warning(f"⚠️ Unexpected response type: {type(response)}")
            
            # If triple extraction fails, fallback to simple entity extraction
            if not triples:
                logger.warning("⚠️ Triple extraction failed, falling back to entity extraction")
                entities = await self._extract_simple_entities(query)
                triples = self._entities_to_triples(entities, query)
            
            logger.info(f"✅ Successfully extracted {len(triples)} triples")
            return triples
            
        except Exception as e:
            logger.error(f"❌ Triple extraction failed: {e}")
            # Emergency fallback: extract simple entities from query
            entities = await self._extract_simple_entities(query)
            return self._entities_to_triples(entities, query)
    
    async def _extract_simple_entities(self, query: str) -> List[str]:
        """Simple entity extraction as fallback"""
        try:
            prompt = self.SIMPLE_ENTITY_PROMPT.format(query=query)
            response = await self.llm.aask(prompt, format="json")
            
            entities = []
            
            # Handle different response types
            if isinstance(response, dict):
                if 'entities' in response:
                    entities = [clean_str(entity) for entity in response['entities']]
            elif isinstance(response, str):
                # Try to parse string format JSON
                try:
                    import json
                    parsed_response = json.loads(response)
                    if 'entities' in parsed_response:
                        entities = [clean_str(entity) for entity in parsed_response['entities']]
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Entity JSON parsing failed for response: {response[:100]}...")
            else:
                logger.warning(f"⚠️ Unexpected entity response type: {type(response)}")
            
            # If no entities are extracted, use keyword splitting as a final fallback
            if not entities:
                entities = [word.strip() for word in query.split() if len(word.strip()) > 2]
                logger.info(f"🔄 Using keyword fallback, extracted {len(entities)} entities")
            
            return entities
            
        except Exception as e:
            logger.error(f"Simple entity extraction also failed: {e}")
            # Last resort fallback: keyword-based simple splitting
            return [word.strip() for word in query.split() if len(word.strip()) > 2]
    
    def _entities_to_triples(self, entities: List[str], query: str) -> List[Triple]:
        """Convert entities to simple triples"""
        triples = []
        for entity in entities:
            # Create simple query triples
            triple = Triple(
                subject=entity,
                predicate="related_to",
                object="query_context",
                confidence=0.7
            )
            triples.append(triple)
        return triples
    
    async def match_triples_to_graph_seeds(self, 
                                         triples: List[Triple], 
                                         top_k_entities: int = 8,
                                         top_k_chunks: int = 5,
                                         similarity_threshold: float = 0.3) -> Tuple[List[MatchedSeed], List[MatchedSeed]]:
        """
        Match extracted triples to entities and chunks in the graph using semantic similarity
        
        Args:
            triples: List of extracted triples
            top_k_entities: Return top-k matched entities
            top_k_chunks: Return top-k matched chunks
            similarity_threshold: Similarity threshold for matching
            
        Returns:
            Tuple[List[MatchedSeed], List[MatchedSeed]]: (matched entity list, matched chunk list)
        """
        logger.info(f"🎯 Starting semantic matching of {len(triples)} triples to graph seeds...")
        
        # Collect all texts that need to be matched
        texts_to_match = []
        for triple in triples:
            # Add subject and object (exclude placeholders)
            if triple.subject and triple.subject != "?" and triple.subject != "query_context":
                texts_to_match.append(triple.subject)
            if triple.object and triple.object != "?" and triple.object != "query_context":
                texts_to_match.append(triple.object)
            # Also consider the string representation of the entire triple
            texts_to_match.append(str(triple))
        
        # Remove duplicates
        texts_to_match = list(set(texts_to_match))
        logger.info(f"📝 Number of texts to match: {len(texts_to_match)}")
        
        # Match entities and chunks in parallel
        entity_matches, chunk_matches = await asyncio.gather(
            self._match_to_entities(texts_to_match, top_k_entities, similarity_threshold),
            self._match_to_chunks(texts_to_match, top_k_chunks, similarity_threshold)
        )
        
        logger.info(f"✅ Matching completed: {len(entity_matches)} entities, {len(chunk_matches)} chunks")
        return entity_matches, chunk_matches
    
    async def _match_to_entities(self, texts: List[str], top_k: int, threshold: float) -> List[MatchedSeed]:
        """Match to entity nodes in the graph"""
        if not self.entities_vdb or not texts:
            return []
        
        try:
            entity_matches = []
            
            for text in texts:
                # Use vector database for semantic search
                matched_nodes = await self.entities_vdb.retrieval_nodes(
                    text, top_k=top_k, graph=self.graph
                )
                
                for i, node in enumerate(matched_nodes):
                    if node is None:
                        continue
                        
                    # Simplified handling, can calculate more precise similarity in practice
                    similarity_score = max(0.1, 1.0 - i * 0.1)  # Simple decreasing score
                    
                    if similarity_score >= threshold:
                        matched_seed = MatchedSeed(
                            original_text=text,
                            matched_entity=node.get("entity_name", "UNKNOWN"),
                            entity_type="entity",
                            similarity_score=similarity_score,
                            graph_node_id=node.get("entity_name", "")
                        )
                        entity_matches.append(matched_seed)
            
            # Sort by similarity and remove duplicates
            entity_matches = self._deduplicate_matches(entity_matches)
            entity_matches.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return entity_matches[:top_k]
            
        except Exception as e:
            logger.error(f"Entity matching failed: {e}")
            return []
    
    async def _match_to_chunks(self, texts: List[str], top_k: int, threshold: float) -> List[MatchedSeed]:
        """Match to document chunk nodes in the graph"""
        if not self.doc_chunk or not texts:
            return []
        
        try:
            chunk_matches = []
            
            # Simplified implementation, should use chunk embeddings for matching in practice
            # Can find relevant chunks through keyword matching or other methods
            
            for text in texts:
                # Simplified chunk matching logic
                # Should use chunk vector search in practice
                chunk_keys = await self._simple_chunk_search(text, top_k)
                
                for i, chunk_key in enumerate(chunk_keys):
                    similarity_score = max(0.1, 1.0 - i * 0.15)
                    
                    if similarity_score >= threshold:
                        matched_seed = MatchedSeed(
                            original_text=text,
                            matched_entity=f"CHUNK_{chunk_key}",
                            entity_type="chunk",
                            similarity_score=similarity_score,
                            graph_node_id=f"CHUNK_{chunk_key}"
                        )
                        chunk_matches.append(matched_seed)
            
            # Sort by similarity and remove duplicates
            chunk_matches = self._deduplicate_matches(chunk_matches)
            chunk_matches.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return chunk_matches[:top_k]
            
        except Exception as e:
            logger.error(f"Chunk matching failed: {e}")
            return []
    
    async def _simple_chunk_search(self, text: str, top_k: int) -> List[str]:
        """Simplified chunk search (should use vector search in practice)"""
        try:
            # This is a simplified implementation
            # Should use chunk embedding-based search in practice
            all_chunk_keys = await self.doc_chunk.get_all_keys() if self.doc_chunk else []
            
            # Simple keyword matching (should use better methods in practice)
            keywords = text.lower().split()
            matched_chunks = []
            
            for chunk_key in all_chunk_keys[:min(100, len(all_chunk_keys))]:  # Limit search scope
                chunk_content = await self.doc_chunk.get_data_by_key(chunk_key)
                if chunk_content and any(keyword in chunk_content.lower() for keyword in keywords):
                    matched_chunks.append(chunk_key)
                    if len(matched_chunks) >= top_k:
                        break
            
            return matched_chunks
            
        except Exception as e:
            logger.error(f"Chunk search failed: {e}")
            return []
    
    def _deduplicate_matches(self, matches: List[MatchedSeed]) -> List[MatchedSeed]:
        """Remove duplicate matches"""
        seen = set()
        deduplicated = []
        
        for match in matches:
            key = (match.matched_entity, match.entity_type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(match)
        
        return deduplicated
    
    def convert_matches_to_legacy_format(self, entity_matches: List[MatchedSeed], chunk_matches: List[MatchedSeed]) -> List[Dict]:
        """
        Convert matching results to the format expected by HKPPRRetriever
        
        Returns:
            List[Dict]: Entity list compatible with legacy format
        """
        legacy_entities = []
        
        # Convert entity matches
        for match in entity_matches:
            entity_dict = {
                "entity_name": match.matched_entity,
                "entity_type": "INFERRED",  # Can infer type through other methods
                "description": f"Matched from query: {match.original_text}",
                "similarity_score": match.similarity_score,
                "matching_method": "triple_extraction_embedding"
            }
            legacy_entities.append(entity_dict)
        
        # Can also convert chunk matches to special entity format
        for match in chunk_matches:
            entity_dict = {
                "entity_name": match.matched_entity,
                "entity_type": "CHUNK",
                "description": f"Chunk matched from query: {match.original_text}",
                "similarity_score": match.similarity_score,
                "matching_method": "triple_extraction_chunk_matching"
            }
            legacy_entities.append(entity_dict)
        
        return legacy_entities
