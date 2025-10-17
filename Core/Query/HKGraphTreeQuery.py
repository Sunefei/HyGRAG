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
    HKGraphTree专用查询器
    
    支持基于层次化社区结构的从顶向下检索和查询处理
    """
    
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)
        
        # 直接从retriever_context获取上下文信息
        # RetrieverContext.context是一个字典，包含所有注册的上下文
        contexts = retriever_context.context
        
        # 初始化TripleExtractor，传入必需的llm参数
        self.triple_extractor = TripleExtractor(
            llm=self.llm,
            entities_vdb=contexts.get('entities_vdb'),
            graph=contexts.get('graph'),
            doc_chunk=contexts.get('doc_chunk')
        )
        
        # 创建HKGraphTree专用检索器
        # 从contexts中移除config，避免重复传递
        retriever_contexts = {k: v for k, v in contexts.items() if k != 'config'}
        self.tree_retriever = HKGraphTreeRetriever(
            config=config,
            **retriever_contexts
        )

    async def _retrieve_relevant_contexts(self, query: str, **kwargs) -> str:
        """
        检索相关上下文 - BaseQuery的抽象方法实现
        
        Args:
            query: 用户查询问题
            
        Returns:
            构建的上下文字符串
        """
        
        try:
            # Step 1: 提取查询实体（如果需要）
            # query_entities = await self.extract_query_entities(query)
            query_entities = [] 

            # Step 2: 执行层次化图检索
            retrieval_results = await self._execute_hierarchical_retrieval(
                query, query_entities
            )
            
            # Step 3: 构建上下文
            context = await self._build_context_from_results(retrieval_results)
            
            #logger.info("✅ HKGraphTree上下文检索完成")
            return context
            
        except Exception as e:
            logger.error(f"❌ HKGraphTree context retrieval failed: {e}")
            return f"Context retrieval failed: {str(e)}"

    async def query(self, question: str) -> str:
        """
        HKGraphTree的主查询方法（重写BaseQuery的query方法）
        
        Args:
            question: 用户查询问题
            
        Returns:
            生成的回答
        """
        logger.info(f"🌲 HKGraphTree查询开始: {question[:100]}...")
        
        try:
            # 获取上下文
            context = await self._retrieve_relevant_contexts(question)
            
            # 根据查询类型生成回答
            if self.config.query_type == "summary":
                response = await self.generation_summary(question, context)
            elif self.config.query_type == "qa":
                response = await self.generation_qa(question, context)
            else:
                logger.error("Invalid query type")
                response = "Unsupported query type"
            
            #logger.debug("✅ HKGraphTree查询完成")
            return response
            
        except Exception as e:
            logger.error(f"❌ HKGraphTree query failed: {e}")
            return f"Query processing failed: {str(e)}"

    async def _execute_hierarchical_retrieval(self, question: str, query_entities: List[Dict]) -> Dict[str, Any]:
        """
        Execute retrieval
        """
        
        logger.info(f"🔧 使用检索方法: hk_tree_flat_search")
        
        try:
            # 检查tree_retriever是否可用
            if not hasattr(self, 'tree_retriever') or self.tree_retriever is None:
                logger.error("❌ tree_retriever未正确初始化")
                return self._get_empty_results()
                
            results = None
            
            try:
                if hasattr(self.tree_retriever, '_hk_tree_flat_search_retrieval'):
                    results = await self.tree_retriever._hk_tree_flat_search_retrieval(question, query_entities)
                else:
                    logger.warning("⚠️ flat_search检索方法不存在，使用回退方法")
                    results = await self._fallback_retrieval(question, query_entities)
                        
            except Exception as retrieval_error:
                logger.error(f"❌ 层次化检索方法执行失败: {retrieval_error}")
                logger.info("🔄 转为使用回退检索方法...")
                results = await self._fallback_retrieval(question, query_entities)
            
            # 确保results不为None
            if results is None:
                logger.warning("⚠️ 检索方法返回None，使用回退结果")
                results = await self._fallback_retrieval(question, query_entities)
            
            return results
            
        except Exception as e:
            logger.error(f"Hierarchical retrieval failed: {e}")
            return self._get_empty_results()

    def _validate_retrieval_results(self, results: Dict[str, Any]) -> bool:
        """
        验证检索结果是否有效
        
        Args:
            results: 检索结果
            
        Returns:
            是否有效
        """
        if not results or not isinstance(results, dict):
            return False
        
        # 检查必要的字段是否存在
        required_fields = ['communities', 'entities', 'chunks', 'relationships', 'community_summaries']
        for field in required_fields:
            if field not in results:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # 检查是否有有效内容
        total_content = (len(results.get('communities', [])) + 
                        len(results.get('entities', [])) + 
                        len(results.get('chunks', [])))
        
        if total_content == 0:
            logger.warning("No content found in retrieval results")
            return False
        
        return True

    def _get_empty_results(self) -> Dict[str, Any]:
        """
        获取空的检索结果
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
        回退检索方法，使用基础检索策略
        """
        logger.info("🔄 使用回退检索方法...")
        
        try:
            results = self._get_empty_results()
            
            # 1. 处理查询实体
            if query_entities:
                # 过滤掉太短的实体
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
                            'description': f'从查询中提取的实体: {entity}'
                        })
                
                results['entities'] = valid_entities[:5]
                logger.info(f"📝 回退检索返回了 {len(results['entities'])} 个有效实体")
            
            # 2. 尝试从已构建的图中获取一些示例数据
            try:
                if hasattr(self, 'tree_retriever') and hasattr(self.tree_retriever, 'graph'):
                    graph = self.tree_retriever.graph
                    if graph:
                        # 尝试获取图的基本信息
                        try:
                            all_nodes = await graph.get_nodes()
                            if all_nodes:
                                # 获取前几个实体节点作为示例
                                sample_entity_nodes = []
                                for node_id in list(all_nodes)[:10]:
                                    if not node_id.startswith('CHUNK_') and not node_id.startswith('COMMUNITY_'):
                                        node_data = await graph.get_node(node_id)
                                        if node_data:
                                            sample_entity_nodes.append({
                                                'entity_name': node_data.get('entity_name', node_id),
                                                'entity_type': node_data.get('entity_type', 'GRAPH_ENTITY'),
                                                'description': node_data.get('description', '图中的实体'),
                                                'source': 'graph_sample'
                                            })
                                
                                if sample_entity_nodes:
                                    results['entities'].extend(sample_entity_nodes[:3])
                                    logger.info(f"📊 从图中添加了 {len(sample_entity_nodes[:3])} 个示例实体")
                        except Exception as e:
                            logger.warning(f"获取图节点失败: {e}")
            except Exception as e:
                logger.warning(f"访问图失败: {e}")
            
            # 3. 尝试获取文档块
            try:
                if hasattr(self, 'tree_retriever') and hasattr(self.tree_retriever, 'doc_chunk'):
                    doc_chunk = self.tree_retriever.doc_chunk
                    if doc_chunk:
                        sample_chunks = []
                        
                        # 尝试不同的方法获取文档块
                        chunk_keys = []
                        if hasattr(doc_chunk, 'get_all_keys'):
                            try:
                                chunk_keys = await doc_chunk.get_all_keys()
                            except:
                                pass
                        
                        # 如果没有keys，尝试使用索引
                        if not chunk_keys:
                            for i in range(5):  # 尝试前5个索引
                                try:
                                    chunk_content = await doc_chunk.get_data_by_index(i)
                                    if chunk_content:
                                        sample_chunks.append({
                                            'id': str(i),
                                            'content': chunk_content[:800],  # 增加内容长度
                                            'type': 'chunk'
                                        })
                                except:
                                    continue
                        else:
                            # 使用keys获取内容
                            for i, key in enumerate(chunk_keys[:3]):
                                try:
                                    chunk_content = await doc_chunk.get_data_by_key(key)
                                    if chunk_content:
                                        sample_chunks.append({
                                            'id': key,
                                            'content': chunk_content[:800],  # 增加内容长度
                                            'type': 'chunk'
                                        })
                                except:
                                    continue
                        
                        results['chunks'] = sample_chunks
                        logger.info(f"📄 回退检索返回了 {len(sample_chunks)} 个文档块")
            except Exception as e:
                logger.warning(f"获取文档块失败: {e}")
            
            # 4. 确保至少有一些基础信息
            if not results['entities'] and not results['chunks']:
                # 如果什么都没有，至少提供查询关键词作为实体
                query_words = [word.strip() for word in question.split() 
                              if len(word.strip()) > 3 and word.strip().lower() not in 
                              ['what', 'where', 'when', 'who', 'why', 'how', 'does', 'was', 'were', 'are', 'the', 'and', 'for']]
                
                fallback_entities = []
                for word in query_words[:5]:
                    fallback_entities.append({
                        'entity_name': word,
                        'entity_type': 'KEYWORD',
                        'description': f'从查询中提取的关键词: {word}'
                    })
                
                results['entities'] = fallback_entities
                logger.info(f"🎯 使用关键词回退，提取了 {len(fallback_entities)} 个关键词实体")
            
            total_content = len(results['entities']) + len(results['chunks'])
            logger.info(f"✅ 回退检索完成，总共返回 {total_content} 项内容")
            
            return results
            
        except Exception as e:
            logger.error(f"回退检索也失败了: {e}")
            # 最后的最后，至少返回一些查询关键词
            try:
                query_words = [word.strip() for word in question.split() if len(word.strip()) > 3][:3]
                fallback_entities = [{'entity_name': word, 'entity_type': 'KEYWORD', 'description': f'关键词: {word}'} for word in query_words]
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
        从RAPTOR式检索结果构建查询上下文
        """
        #logger.info("📝 构建RAPTOR式查询上下文...")
        
        context_parts = []
        max_context_length = getattr(self.config, 'max_token_for_text_unit', 4000) * 10 #TODO context 截断
        
        # 1. 层次化社区信息（按层次和相似性排序）
        communities = retrieval_results.get('communities', [])
        if communities:
            context_parts.append("=== Hierarchical Community Analysis ===")
            
            # 按层次分组
            level_groups = {}
            for community in communities:
                level = community.get('level', 0)
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append(community)
            
            # 从高层到低层展示
            for level in sorted(level_groups.keys(), reverse=True):
                level_communities = level_groups[level]
                # 按相似性分数排序
                level_communities.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                
                context_parts.append(f"Level {level} Communities:")
                for i, community in enumerate(level_communities[:1]):  # 每层最多1个
                    score = community.get('similarity_score', 0)
                    context_parts.append(f"  Community {i+1} (Score: {score:.3f}, Members: {community.get('member_count', 0)}):")
                    if community.get('summary'):
                        summary = community['summary']
                        if len(summary) > 3000:
                            summary = summary[:3000] + "..."
                        context_parts.append(f"    {summary}")
                context_parts.append("")
        
        # 2. 高相似性实体信息
        entities = retrieval_results.get('entities', [])
        if entities:
            context_parts.append("=== Most Relevant Entities ===")
            # 实体已经按相似性排序，直接使用
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
                    # 截断过长的描述
                    if len(description) > 150:
                        description = description[:150] + "..."
                    context_parts.append(f"   Description: {description}")
            context_parts.append("")
        
        # 3. 关系网络信息
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
        
        # 4. 最相关的文档内容
        chunks = retrieval_results.get('chunks', [])
        if chunks:
            context_parts.append("=== Most Relevant Documents ===")
            # 文档块已经按相似性排序
            for i, chunk in enumerate(chunks):#TODO
                score = chunk.get('similarity_score', 0)
                content = chunk.get('content', '')

                context = "\n".join(context_parts)  # context完整内容截断
                if len(content) + len(context) > max_context_length:
                    break
                
                context_parts.append(f"Document {i+1} [Score: {score:.3f}]:")
                
                # 智能截断：保留重要部分 - 可配置版本
                max_full_length = getattr(self.config, 'max_document_display_length', 8000)  # 允许完整显示的最大长度
                max_smart_truncate_length = getattr(self.config, 'max_smart_truncate_length', 8000)  # 智能截断的阈值
                head_chars = getattr(self.config, 'truncate_head_chars', 4000)  # 保留开头字符数
                tail_chars = getattr(self.config, 'truncate_tail_chars', 3000)  # 保留结尾字符数
                
                if len(content) > max_full_length:
                    # 智能截断：保留开头和结尾
                    content = content[:head_chars] + "\n...[content truncated]...\n" + content[-tail_chars:]
                elif len(content) > max_smart_truncate_length:
                    # 简单截断：只保留开头
                    content = content[:head_chars] + "..."

                context_parts.append(content)
                context_parts.append("")
        
        # 5. 添加检索元信息
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
        
        # 限制总长度
        max_context_length = getattr(self.config, 'max_token_for_text_unit', 4000) * 10 #TODO context 截断
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n...(content truncated for length)"
        
        logger.info(f"📋 HKGraphTreeLSH上下文构建完成，长度: {len(context)} 字符，包含 {total_items} 项内容")
        return context

    async def generation_qa(self, query: str, context: str) -> str:
        """
        生成问答回复 - BaseQuery的抽象方法实现
        
        Args:
            query: 用户查询问题
            context: 检索到的上下文
            
        Returns:
            生成的回答
        """
        logger.debug("🤖 开始生成问答回复...")
        
        if not context or context.strip() == "":
            return "Sorry, no relevant information was found to answer your question."
        
        try:
            # 构建提示词
            system_prompt = self._build_system_prompt_for_qa_prompt_options_analyze_nm() #TODO 提示词
            #system_prompt = ""
            # Build user message
            user_message = f"""
Based on the following context information, please answer the user's question.

Context information:
{context}

User question: {query}

Give the best full answer amongst the option to question.(if the question is a option chosing question)
According to the retrieved context, please provide detailed and accurate answers.If the context does not contain sufficient information to answer the question, please state "Insufficient information". When possible, reference specific information from the context.
"""
            
            # 调用LLM生成回答 # system_msgs=[system_prompt] if system_prompt else None 
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
            
            logger.debug("✅ 问答回复生成完成")
            #print(f"response: {response}\n") #TODO debuging
            return response
            
        except Exception as e:
            logger.error(f"Q&A response generation failed: {e}")
            return f"Sorry, an error occurred while generating the answer: {str(e)}"

    async def generation_summary(self, query: str, context: str) -> str:
        """
        生成摘要 - BaseQuery的抽象方法实现
        
        Args:
            query: 用户查询问题
            context: 检索到的上下文
            
        Returns:
            生成的摘要
        """
        logger.info("📋 开始生成摘要...")
        
        if not context or context.strip() == "":
            return "Sorry, no relevant information was found to generate a summary."
        
        try:
            # 构建摘要提示词
            system_prompt = self._build_system_prompt_for_summary() #TODO
            
            # Build user message
            user_message = f"""
Based on the following context information, generate a concise summary to answer the query.

Context information:
{context}

Query topic: {query}

Please generate a concise and comprehensive summary that highlights the most relevant points to the query.
"""
            
            # 调用LLM生成摘要
            if hasattr(self, 'llm') and self.llm:
                response = await self.llm.aask(
                    user_message,
                    system_msgs=[system_prompt] if system_prompt else None
                )
            else:
                # Simple template summary
                response = f"Summary based on hierarchical retrieval:\n\n{context[:300]}..."
            
            logger.info("✅ 摘要生成完成")
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
#7. Give the best full answer amongst the option to question.

        return system_prompt
    
    def _build_system_prompt_for_qa_prompt_options_analyze(self) -> str:  #TODO claude prompt analyze
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

    def _build_system_prompt_for_qa_prompt_options_analyze_nm(self) -> str:  #TODO claude prompt analyze
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

    def _build_system_prompt_for_qa_prompt_options_analyze_noII(self) -> str:  #TODO claude prompt analyze
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

    def _build_system_prompt_for_qa_prompt_options_analyze_debug(self) -> str:  #TODO claude prompt analyze
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