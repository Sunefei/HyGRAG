import asyncio
from typing import List, Dict, Any, Optional

from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Prompt import QueryPrompt, GraphPrompt
from Core.Retriever.HKGraphTreeRetriever import HKGraphTreeRetriever
from Core.Query.TripleExtractor import TripleExtractor
from Core.Common.Utils import truncate_list_by_token_size, list_to_quoted_csv_string, prase_json_from_response, clean_str
from Core.Common.Constants import Retriever


class HKGraphTreeQuery(BaseQuery):
    """
    Dedicated querier for HKGraphTree.
    
    Supports top-down retrieval and query processing based on hierarchical community structure.
    """
    
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)
        
        # Directly get context information from retriever_context
        # RetrieverContext.context is a dictionary containing all registered contexts
        contexts = retriever_context.context
        
        # Initialize TripleExtractor, passing the required llm parameter
        self.triple_extractor = TripleExtractor(
            llm=self.llm,
            entities_vdb=contexts.get('entities_vdb'),
            graph=contexts.get('graph'),
            doc_chunk=contexts.get('doc_chunk')
        )
        
        # Create a dedicated retriever for HKGraphTree
        # Remove config from contexts to avoid duplicate passing
        retriever_contexts = {k: v for k, v in contexts.items() if k != 'config'}
        self.tree_retriever = HKGraphTreeRetriever(
            config=config,
            **retriever_contexts
        )

    async def _retrieve_relevant_contexts(self, query: str, **kwargs) -> str:
        """
        Retrieve relevant context - implementation of the abstract method from BaseQuery.
        
        Args:
            query: User query question.
            
        Returns:
            Constructed context string.
        """
        
        try:
            # Step 1: Extract query entities (if needed)
            # query_entities = await self.extract_query_entities(query)
            query_entities = [] 

            # Step 2: Execute hierarchical graph retrieval
            retrieval_results = await self._execute_hierarchical_retrieval(
                query, query_entities
            )
            
            # Step 3: Build context
            context = await self._build_context_from_results(retrieval_results)
            
            #logger.info("✅ HKGraphTree context retrieval completed")
            return context
            
        except Exception as e:
            logger.error(f"❌ HKGraphTree context retrieval failed: {e}")
            return f"Context retrieval failed: {str(e)}"

    async def query(self, question: str) -> str:
        """
        Main query method for HKGraphTree (overrides BaseQuery's query method).
        
        Args:
            question: User query question.
            
        Returns:
            Generated answer.
        """
        logger.info(f"🌲 HKGraphTree query started: {question[:100]}...")
        
        try:
            # Get context
            context = await self._retrieve_relevant_contexts(question)
            
            # Generate answer based on query type
            if self.config.query_type == "summary":
                response = await self.generation_summary(question, context)
            elif self.config.query_type == "qa":
                response = await self.generation_qa(question, context)
            else:
                logger.error("Invalid query type")
                response = "Unsupported query type"
            
            #logger.debug("✅ HKGraphTree query completed")
            return response
            
        except Exception as e:
            logger.error(f"❌ HKGraphTree query failed: {e}")
            return f"Query processing failed: {str(e)}"

    async def _execute_hierarchical_retrieval(self, question: str, query_entities: List[Dict]) -> Dict[str, Any]:
        """
        Execute retrieval
        """
        
        logger.info(f"🔧 Using retrieval method: hk_tree_flat_search")
        
        try:
            # Check if tree_retriever is available
            if not hasattr(self, 'tree_retriever') or self.tree_retriever is None:
                logger.error("❌ tree_retriever not initialized correctly")
                return self._get_empty_results()
                
            results = None
            
            try:
                if hasattr(self.tree_retriever, '_hk_tree_flat_search_retrieval'):
                    results = await self.tree_retriever._hk_tree_flat_search_retrieval(question, query_entities)
                else:
                    logger.warning("⚠️ flat_search retrieval method does not exist, using fallback method")
                    results = await self._fallback_retrieval(question, query_entities)
                        
            except Exception as retrieval_error:
                logger.error(f"❌ Hierarchical retrieval method execution failed: {retrieval_error}")
                logger.info("🔄 Switching to fallback retrieval method...")
                results = await self._fallback_retrieval(question, query_entities)
            
            # Ensure results is not None
            if results is None:
                logger.warning("⚠️ Retrieval method returned None, using fallback results")
                results = await self._fallback_retrieval(question, query_entities)
            
            return results
            
        except Exception as e:
            logger.error(f"Hierarchical retrieval failed: {e}")
            return self._get_empty_results()

    def _validate_retrieval_results(self, results: Dict[str, Any]) -> bool:
        """
        Validate whether the retrieval results are valid.
        
        Args:
            results: Retrieval results.
            
        Returns:
            Whether they are valid.
        """
        if not results or not isinstance(results, dict):
            return False
        
        # Check if necessary fields exist
        required_fields = ['communities', 'entities', 'chunks', 'relationships', 'community_summaries']
        for field in required_fields:
            if field not in results:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Check for valid content
        total_content = (len(results.get('communities', [])) + 
                        len(results.get('entities', [])) + 
                        len(results.get('chunks', [])))
        
        if total_content == 0:
            logger.warning("No content found in retrieval results")
            return False
        
        return True

    def _get_empty_results(self) -> Dict[str, Any]:
        """
        Get empty retrieval results.
        """
        return {
            'communities': [],
            'entities': [],
            'chunks': [],
            'relationships': [],
            'community_summaries': []
        }
    
    async def _fallback_retrieval(self, question: str, query_entities: List[Dict]) -> Dict[str, Any]:
        """
        Fallback retrieval method, using a basic retrieval strategy.
        """
        logger.info("🔄 Using fallback retrieval method...")
        
        try:
            results = self._get_empty_results()
            
            # 1. Process query entities
            if query_entities:
                # Filter out entities that are too short
                valid_entities = []
                for entity in query_entities:
                    if isinstance(entity, dict):
                        entity_name = entity.get('entity_name', '')
                        if len(entity_name) > 2 and entity_name not in ['the', 'and', 'for', 'are', 'was', 'were']:
                            valid_entities.append(entity)
                    elif isinstance(entity, str) and len(entity) > 2:
                        valid_entities.append({
                            'entity_name': entity,
                            'entity_type': 'EXTRACTED',
                            'description': f'Entity extracted from query: {entity}'
                        })
                
                results['entities'] = valid_entities[:5]
                logger.info(f"📝 Fallback retrieval returned {len(results['entities'])} valid entities")
            
            # 2. Try to get some sample data from the constructed graph
            try:
                if hasattr(self, 'tree_retriever') and hasattr(self.tree_retriever, 'graph'):
                    graph = self.tree_retriever.graph
                    if graph:
                        # Try to get basic graph information
                        try:
                            all_nodes = await graph.get_nodes()
                            if all_nodes:
                                # Get the first few entity nodes as examples
                                sample_entity_nodes = []
                                for node_id in list(all_nodes)[:10]:
                                    if not node_id.startswith('CHUNK_') and not node_id.startswith('COMMUNITY_'):
                                        node_data = await graph.get_node(node_id)
                                        if node_data:
                                            sample_entity_nodes.append({
                                                'entity_name': node_data.get('entity_name', node_id),
                                                'entity_type': node_data.get('entity_type', 'GRAPH_ENTITY'),
                                                'description': node_data.get('description', 'Entity in the graph'),
                                                'source': 'graph_sample'
                                            })
                                
                                if sample_entity_nodes:
                                    results['entities'].extend(sample_entity_nodes[:3])
                                    logger.info(f"📊 Added {len(sample_entity_nodes[:3])} sample entities from the graph")
                        except Exception as e:
                            logger.warning(f"Failed to get graph nodes: {e}")
            except Exception as e:
                logger.warning(f"Failed to access graph: {e}")
            
            # 3. Try to get document chunks
            try:
                if hasattr(self, 'tree_retriever') and hasattr(self.tree_retriever, 'doc_chunk'):
                    doc_chunk = self.tree_retriever.doc_chunk
                    if doc_chunk:
                        sample_chunks = []
                        
                        # Try different methods to get document chunks
                        chunk_keys = []
                        if hasattr(doc_chunk, 'get_all_keys'):
                            try:
                                chunk_keys = await doc_chunk.get_all_keys()
                            except:
                                pass
                        
                        # If there are no keys, try using index
                        if not chunk_keys:
                            for i in range(5):  # Try the first 5 indices
                                try:
                                    chunk_content = await doc_chunk.get_data_by_index(i)
                                    if chunk_content:
                                        sample_chunks.append({
                                            'id': str(i),
                                            'content': chunk_content[:800],  # Increase content length
                                            'type': 'chunk'
                                        })
                                except:
                                    continue
                        else:
                            # Get content using keys
                            for i, key in enumerate(chunk_keys[:3]):
                                try:
                                    chunk_content = await doc_chunk.get_data_by_key(key)
                                    if chunk_content:
                                        sample_chunks.append({
                                            'id': key,
                                            'content': chunk_content[:800],  # Increase content length
                                            'type': 'chunk'
                                        })
                                except:
                                    continue
                        
                        results['chunks'] = sample_chunks
                        logger.info(f"📄 Fallback retrieval returned {len(sample_chunks)} document chunks")
            except Exception as e:
                logger.warning(f"Failed to get document chunks: {e}")
            
            # 4. Ensure there is at least some basic information
            if not results['entities'] and not results['chunks']:
                # If there is nothing, at least provide the query keywords as entities
                query_words = [word.strip() for word in question.split() 
                              if len(word.strip()) > 3 and word.strip().lower() not in 
                              ['what', 'where', 'when', 'who', 'why', 'how', 'does', 'was', 'were', 'are', 'the', 'and', 'for']]
                
                fallback_entities = []
                for word in query_words[:5]:
                    fallback_entities.append({
                        'entity_name': word,
                        'entity_type': 'KEYWORD',
                        'description': f'Keyword extracted from query: {word}'
                    })
                
                results['entities'] = fallback_entities
                logger.info(f"🎯 Used keyword fallback, extracted {len(fallback_entities)} keyword entities")
            
            total_content = len(results['entities']) + len(results['chunks'])
            logger.info(f"✅ Fallback retrieval completed, returned a total of {total_content} items")
            
            return results
            
        except Exception as e:
            logger.error(f"Fallback retrieval also failed: {e}")
            # As a last resort, at least return some query keywords
            try:
                query_words = [word.strip() for word in question.split() if len(word.strip()) > 3][:3]
                fallback_entities = [{'entity_name': word, 'entity_type': 'KEYWORD', 'description': f'Keyword: {word}'} for word in query_words]
                return {
                    'communities': [],
                    'entities': fallback_entities,
                    'chunks': [],
                    'relationships': [],
                    'community_summaries': []
                }
            except:
                return self._get_empty_results()

    async def _build_context_from_results(self, retrieval_results: Dict[str, Any]) -> str:
        """
        Build query context from RAPTOR-style retrieval results.
        """
        #logger.info("📝 Building RAPTOR-style query context...")
        
        context_parts = []
        max_context_length = getattr(self.config, 'max_token_for_text_unit', 4000) * 10 #TODO context truncation
        
        # 1. Hierarchical community information (sorted by level and similarity)
        communities = retrieval_results.get('communities', [])
        if communities:
            context_parts.append("=== Hierarchical Community Analysis ===")
            
            # Group by level
            level_groups = {}
            for community in communities:
                level = community.get('level', 0)
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append(community)
            
            # Display from high level to low level
            for level in sorted(level_groups.keys(), reverse=True):
                level_communities = level_groups[level]
                # Sort by similarity score
                level_communities.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                
                context_parts.append(f"Level {level} Communities:")
                for i, community in enumerate(level_communities[:1]):  # Max 1 per level
                    score = community.get('similarity_score', 0)
                    context_parts.append(f"  Community {i+1} (Score: {score:.3f}, Members: {community.get('member_count', 0)}):")
                    if community.get('summary'):
                        summary = community['summary']
                        if len(summary) > 3000:
                            summary = summary[:3000] + "..."
                        context_parts.append(f"    {summary}")
                context_parts.append("")
        
        # 2. High-similarity entity information
        entities = retrieval_results.get('entities', [])
        if entities:
            context_parts.append("=== Most Relevant Entities ===")
            # Entities are already sorted by similarity, use directly
            for i, entity in enumerate(entities):
                entity_name = entity.get('entity_name', 'N/A')
                entity_type = entity.get('entity_type', '')
                score = entity.get('similarity_score', 0)
                description = entity.get('description', '')
                
                entity_info = f"{i+1}. {entity_name}"
                if entity_type:
                    entity_info += f" ({entity_type})"
                if score > 0:
                    entity_info += f" [Score: {score:.3f}]"
                context_parts.append(entity_info)
                
                if description:
                    # Truncate overly long descriptions
                    if len(description) > 150:
                        description = description[:150] + "..."
                    context_parts.append(f"   Description: {description}")
            context_parts.append("")
        
        # 3. Relationship network information
        relationships = retrieval_results.get('relationships', [])
        if relationships:
            context_parts.append("=== Key Relationships ===")
            for i, rel in enumerate(relationships):
                src_id = rel.get('src_id', 'N/A')
                tgt_id = rel.get('tgt_id', 'N/A')
                relation_name = rel.get('relation_name', 'N/A')
                description = rel.get('description', '')
                
                rel_info = f"{i+1}. {src_id} --[{relation_name}]--> {tgt_id}"
                context_parts.append(rel_info)
                
                if description:
                    if len(description) > 100:
                        description = description[:100] + "..."
                    context_parts.append(f"   Context: {description}")
            context_parts.append("")
        
        # 4. Most relevant document content
        chunks = retrieval_results.get('chunks', [])
        if chunks:
            context_parts.append("=== Most Relevant Documents ===")
            # Document chunks are already sorted by similarity
            for i, chunk in enumerate(chunks):#TODO
                score = chunk.get('similarity_score', 0)
                content = chunk.get('content', '')

                context = "\n".join(context_parts)  # context full content truncation
                if len(content) + len(context) > max_context_length:
                    break
                
                context_parts.append(f"Document {i+1} [Score: {score:.3f}]:")
                
                # Smart truncation: preserve important parts - configurable version
                max_full_length = getattr(self.config, 'max_document_display_length', 8000)  # Max length for full display
                max_smart_truncate_length = getattr(self.config, 'max_smart_truncate_length', 8000)  # Threshold for smart truncation
                head_chars = getattr(self.config, 'truncate_head_chars', 4000)  # Number of characters to keep at the beginning
                tail_chars = getattr(self.config, 'truncate_tail_chars', 3000)  # Number of characters to keep at the end
                
                if len(content) > max_full_length:
                    # Smart truncation: keep beginning and end
                    content = content[:head_chars] + "\n...[content truncated]...\n" + content[-tail_chars:]
                elif len(content) > max_smart_truncate_length:
                    # Simple truncation: keep only the beginning
                    content = content[:head_chars] + "..."

                context_parts.append(content)
                context_parts.append("")
        
        # 5. Add retrieval metadata
        total_items = len(communities) + len(entities) + len(chunks) + len(relationships)
        #total_items = 1 + len(entities) + 4 + len(relationships)
        if total_items > 0:
            context_parts.append("=== Retrieval Summary ===")
            context_parts.append(f"Retrieved {len(communities)} communities across {len(set(c.get('level', 0) for c in communities))} levels, "
                                f"{len(entities)} entities, {len(chunks)} documents, and {len(relationships)} relationships "
                                f"using  hierarchical retrieval.")
            # context_parts.append(f"Retrieved 1 communities across {len(set(c.get('level', 0) for c in communities))} levels, "
            #                     f"{len(entities)} entities, 4 documents, and {len(relationships)} relationships "
            #                     f"using hierarchical retrieval.")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Limit total length
        max_context_length = getattr(self.config, 'max_token_for_text_unit', 4000) * 10 #TODO context truncation
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n...(content truncated for length)"
        
        logger.info(f"📋 HKGraphTreeLSH context built, length: {len(context)} chars, containing {total_items} items")
        return context

    async def generation_qa(self, query: str, context: str) -> str:
        """
        Generate Q&A response - implementation of the abstract method from BaseQuery
        
        Args:
            query: User query question
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        logger.debug("🤖 Starting Q&A response generation...")
        
        if not context or context.strip() == "":
            return "Sorry, no relevant information was found to answer the question."
        
        try:
            # Build prompt
            system_prompt = self._build_system_prompt_for_qa_prompt_options_analyze_nm() #TODO prompt
            
            # Build user message
            user_message = f"""
Based on the following context information, please answer the user's question.

Context information:
{context}

User question: {query}

Give the best full answer amongst the option to question.(if the question is a option chosing question)
According to the retrieved context, please provide detailed and accurate answers.If the context does not contain sufficient information to answer the question, please state "Insufficient information". When possible, reference specific information from the context.
"""
            
            # Call LLM to generate answer # system_msgs=[system_prompt] if system_prompt else None 
            if hasattr(self, 'llm') and self.llm:
                #print(f"system_prompt: {system_prompt}\n")
                # print(f"user_message: {user_message}\n") #TODO debuging
                #print("user_message: {}\n".format(user_message.replace('\n', '\\n')))
                response = await self.llm.aask(
                    user_message, system_msgs=[system_prompt] if system_prompt else None
                    
                )
            else:
                # Simple template response
                response = f"Based on the retrieved information, here is the answer to the question '{query}':\n\n{context[:200]}..."
            
            logger.debug("✅ Q&A response generated successfully")
            #print(f"response: {response}\n") #TODO debuging
            return response
            
        except Exception as e:
            logger.error(f"Q&A response generation failed: {e}")
            return f"Sorry, an error occurred while generating the answer: {str(e)}"

    async def generation_summary(self, query: str, context: str) -> str:
        """
        Generate summary - implementation of the abstract method from BaseQuery
        
        Args:
            query: User query question
            context: Retrieved context
            
        Returns:
            Generated summary
        """
        logger.info("📋 Starting summary generation...")
        
        if not context or context.strip() == "":
            return "Sorry, no relevant information was found to generate a summary."
        
        try:
            # Build summary prompt
            system_prompt = self._build_system_prompt_for_summary() #TODO
            
            # Build user message
            user_message = f"""
Based on the following context information, generate a concise summary to answer the query.

Context information:
{context}

Query topic: {query}

Please generate a concise and comprehensive summary that highlights the most relevant points to the query.
"""
            
            # Call LLM to generate summary
            if hasattr(self, 'llm') and self.llm:
                response = await self.llm.aask(
                    user_message,
                    system_msgs=[system_prompt] if system_prompt else None
                )
            else:
                # Simple template summary
                response = f"Summary based on hierarchical retrieval:\n\n{context[:300]}..."
            
            logger.info("✅ Summary generation completed")
            return response
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Sorry, an error occurred while generating the summary: {str(e)}"



    def _build_system_prompt_for_qa_new_prompt_options(self) -> str:
        """
        Build system prompt for Q&A (after modified)
        """
        system_prompt = """
        You are an intelligent RAG Q&A assistant using hierarchical knowledge graphs.

Rules:
1. Consider **Entities**, **Key Relationships**, **Documents**, and **Community Summaries** together.  
2. If a fact appears in **Key Relationships**, treat it as reliable even if not repeated elsewhere.  
3. Use **Documents** for context or confirmation, but do not require them to validate relationship facts.  
4. Report consistency across sources; if sources conflict, describe the discrepancy.  
5. If none of the sections provide sufficient relevant evidence, explicitly state “Insufficient information”. After that, you may provide a plausible guess or hypothesis, clearly labeling it as a guess and separating it from the evidence-based answer. 
6. Always state which section(s) support your answer.
        """

        return system_prompt
    
    def _build_system_prompt_for_qa_prompt_options_analyze(self) -> str: 
        """
        Build system prompt for Q&A (after modified)
        """
        system_prompt = """
You are an intelligent RAG Q&A assistant using hierarchical knowledge graphs.

Rules:
1. Consider **Entities**, **Key Relationships**, **Documents**, and **Community Summaries** together.  
2. If a fact appears in **Key Relationships**, treat it as the most reliable source of truth, even if it seems unusual or is not repeated elsewhere. Do not override it with everyday common-sense assumptions.  
3. Use **Documents** for context or confirmation, but do not require them to validate relationship facts.  
4. Report consistency across sources; if sources conflict, describe the discrepancy.  
5. If none of the sections provide sufficient relevant evidence, explicitly state “Insufficient information”. After that, you may provide a plausible guess or hypothesis, clearly labeling it as a guess and separating it from the evidence-based answer.  
6. You need to analyze based on the original text, not over-interpret it.

Response format: First analyze the evidence and reasoning process, then provide your answer with source attribution.

"""

        return system_prompt

    def _build_system_prompt_for_qa_prompt_options_analyze_nm(self) -> str:  
        """
        Build system prompt for Q&A (after modified)
        """
        system_prompt = """
You are an intelligent RAG Q&A assistant using hierarchical knowledge graphs.

Rules:
1. Consider **Entities**, **Key Relationships**, **Documents**, and **Community Summaries** together.  
2. If a fact appears in **Key Relationships**, treat it as the most reliable source of truth, even if it seems unusual or is not repeated elsewhere. Do not override it with everyday common-sense assumptions.  
3. Use **Documents** for context or confirmation, but do not require them to validate relationship facts.  
4. Report consistency across sources; if sources conflict, describe the discrepancy.  
5. Do not make up information.
6. You need to analyze based on the original text, not over-interpret it.

Response format: First analyze the evidence and reasoning process, then provide your answer with source attribution.

"""

        return system_prompt

    def _build_system_prompt_for_qa_prompt_options_analyze_noII(self) -> str:  
        """
        Build system prompt for Q&A (after modified)
        """
        system_prompt = """
You are an intelligent RAG Q&A assistant using hierarchical knowledge graphs.

Rules:
1. Consider **Entities**, **Key Relationships**, **Documents**, and **Community Summaries** together.  
2. If a fact appears in **Key Relationships**, treat it as the most reliable source of truth, even if it seems unusual or is not repeated elsewhere. Do not override it with everyday common-sense assumptions.  
3. Use **Documents** for context or confirmation, but do not require them to validate relationship facts.  
4. Report consistency across sources; if sources conflict, describe the discrepancy.  
5. Do not make up. 
6. You need to analyze based on the original text, not over-interpret it.

Response format: First analyze the evidence and reasoning process, then provide your answer with source attribution.

"""

        return system_prompt

    def _build_system_prompt_for_summary(self) -> str:
        """
        Build system prompt for summary
        """
        system_prompt = """You are a professional information summarization expert specializing in organizing information based on hierarchical knowledge graphs.

Summary requirements:
1. Extract the most core topics and concepts
2. Maintain hierarchical structure and logical relationships
3. Highlight key entities and their important relationships
4. Be concise and clear, avoiding redundant information
5. Maintain an objective and neutral tone

Please generate well-structured and well-highlighted English summaries."""
        
        return system_prompt