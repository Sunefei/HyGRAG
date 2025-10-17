import asyncio
from Core.Chunk.ChunkFactory import create_chunk_method
from Core.Common.Utils import mdhash_id
from Core.Common.Logger import logger
from Core.Schema.ChunkSchema import TextChunk
from Core.Storage.ChunkKVStorage import ChunkKVStorage
from typing import List, Union


class DocChunk:
    def __init__(self, config, token_model, namesapce):
        self.config = config
        self.chunk_method = create_chunk_method(self.config.chunk_method)
        self._chunk = ChunkKVStorage(namespace=namesapce)
        self.token_model = token_model

    @property
    def namespace(self):
        return None

    # TODO: Try to rewrite here, not now
    @namespace.setter
    def namespace(self, namespace):
        self.namespace = namespace

    async def build_chunks(self, docs: Union[str, List[str]], force=True):
        logger.info("Starting chunk the given documents")
  
        is_exist = await self._load_chunk(force)
        if not is_exist or force:

            # TODO: Now we only support the str, list[str], Maybe for more types.
            if isinstance(docs, str):
                docs = [docs]

            # 
            original_docs_count = len(docs)
            logger.info(f"Original docs count: {original_docs_count}")

            if isinstance(docs, list):
                if all(isinstance(doc, dict) for doc in docs):
                    docs = {
                        mdhash_id(doc["content"].strip(), prefix="doc-"): {
                            "content": doc["content"].strip(),
                            "title": doc.get("title", ""),
                        }
                        for doc in docs
                    }
                else:
                    docs = {
                        mdhash_id(doc.strip(), prefix="doc-"): {
                            "content": doc.strip(),
                            "title": "",
                        }
                        for doc in docs
                    }

            # 
            filtered_docs_count = len(docs)
            logger.info(f"After filtering docs count: {filtered_docs_count}")
            logger.info(f"Filtered out {original_docs_count - filtered_docs_count} documents")

            flatten_list = list(docs.items())
            docs = [doc[1]["content"] for doc in flatten_list]
            doc_keys = [doc[0] for doc in flatten_list]
            title_list = [doc[1]["title"] for doc in flatten_list]
            tokens = self.token_model.encode_batch(docs, num_threads=16)

            # 
            logger.info(f"Tokenized {len(tokens)} documents")
            total_tokens = sum(len(token_list) for token_list in tokens)
            avg_tokens = total_tokens / len(tokens) if tokens else 0
            logger.info(f"Total tokens: {total_tokens}, Average tokens per doc: {avg_tokens:.2f}")

            chunks = await self.chunk_method(
                tokens,
                doc_keys=doc_keys,
                tiktoken_model=self.token_model,
                title_list=title_list,
                overlap_token_size=self.config.chunk_overlap_token_size,
                max_token_size=self.config.chunk_token_size,
            )

            #  
            final_chunks_count = len(chunks)
            logger.info(f"Final chunks count: {final_chunks_count}")
            logger.info(f"Chunking method: {self.config.chunk_method}")
            logger.info(f"Chunk config - max_size: {self.config.chunk_token_size}, overlap: {self.config.chunk_overlap_token_size}")

            # 
            if chunks:
                chunk_sizes = [chunk["tokens"] for chunk in chunks]
                avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
                max_chunk_size = max(chunk_sizes)
                min_chunk_size = min(chunk_sizes)
                logger.info(f"Chunk size stats - avg: {avg_chunk_size:.2f}, min: {min_chunk_size}, max: {max_chunk_size}")

            for chunk in chunks:
                chunk["chunk_id"] = mdhash_id(chunk["content"], prefix="chunk-")
                await self._chunk.upsert(chunk["chunk_id"], TextChunk(**chunk))

            await self._chunk.persist()
        logger.info("✅ Finished the chunking stage")

    async def update_chunks(self, incremental_docs: Union[str, List[str]]):
        """
        增量更新chunks：只处理新文档，不影响现有chunk数据
        
        Args:
            incremental_docs: 新增的文档数据
        """
        logger.info(f"🔄 开始增量更新chunks，新增文档数量: {len(incremental_docs) if isinstance(incremental_docs, list) else 1}")
        
        # 确保现有chunk数据已加载
        is_loaded = await self._load_chunk(force=False)
        if not is_loaded:
            logger.warning("⚠️ 未找到现有chunk数据，将作为初始构建处理")
            await self.build_chunks(incremental_docs, force=True)
            return
        
        # 获取当前chunk数量（用于统计）
        existing_chunks = await self.get_chunks()
        existing_count = len(existing_chunks) if existing_chunks else 0
        logger.info(f"📚 现有chunk数量: {existing_count}")
        
        # 处理增量文档数据格式
        if isinstance(incremental_docs, str):
            incremental_docs = [incremental_docs]
        
        # 记录原始文档数
        original_docs_count = len(incremental_docs)
        logger.info(f"增量文档数: {original_docs_count}")
        
        # 转换为标准格式并去重
        if isinstance(incremental_docs, list):
            if all(isinstance(doc, dict) for doc in incremental_docs):
                new_docs = {
                    mdhash_id(doc["content"].strip(), prefix="doc-"): {
                        "content": doc["content"].strip(),
                        "title": doc.get("title", ""),
                    }
                    for doc in incremental_docs
                }
            else:
                new_docs = {
                    mdhash_id(doc.strip(), prefix="doc-"): {
                        "content": doc.strip(),
                        "title": "",
                    }
                    for doc in incremental_docs
                }
        
        # 过滤已存在的文档（基于hash ID去重）
        existing_doc_ids = set()
        if existing_chunks:
            for chunk_item in existing_chunks:
                if isinstance(chunk_item, tuple) and len(chunk_item) == 2:
                    chunk_key, chunk_obj = chunk_item
                    if hasattr(chunk_obj, 'doc_id'):
                        existing_doc_ids.add(chunk_obj.doc_id)
        
        # 只保留真正新增的文档
        truly_new_docs = {}
        for doc_id, doc_data in new_docs.items():
            if doc_id not in existing_doc_ids:
                truly_new_docs[doc_id] = doc_data
            else:
                logger.debug(f"跳过重复文档: {doc_id}")
        
        if not truly_new_docs:
            logger.info("📝 没有真正新增的文档，跳过chunk更新")
            return []
        
        logger.info(f"📝 过滤后真正新增文档数: {len(truly_new_docs)}")
        
        # 对新文档进行chunk分割
        flatten_list = list(truly_new_docs.items())
        docs_content = [doc[1]["content"] for doc in flatten_list]
        doc_keys = [doc[0] for doc in flatten_list]
        title_list = [doc[1]["title"] for doc in flatten_list]
        
        # Token化新文档
        tokens = self.token_model.encode_batch(docs_content, num_threads=16)
        
        # 记录token化信息
        logger.info(f"新文档token化完成: {len(tokens)} 个文档")
        total_tokens = sum(len(token_list) for token_list in tokens)
        avg_tokens = total_tokens / len(tokens) if tokens else 0
        logger.info(f"新文档总tokens: {total_tokens}, 平均tokens: {avg_tokens:.2f}")
        
        # 获取现有chunk的最大索引，确保新chunk索引不冲突
        max_existing_index = -1
        if existing_chunks:
            for chunk_item in existing_chunks:
                if isinstance(chunk_item, tuple) and len(chunk_item) == 2:
                    _, chunk_obj = chunk_item
                    if hasattr(chunk_obj, 'index') and chunk_obj.index is not None:
                        max_existing_index = max(max_existing_index, chunk_obj.index)
        
        logger.info(f"现有chunk最大索引: {max_existing_index}")
        
        # 使用相同的chunk配置进行分割
        new_chunks = await self.chunk_method(
            tokens,
            doc_keys=doc_keys,
            tiktoken_model=self.token_model,
            title_list=title_list,
            overlap_token_size=self.config.chunk_overlap_token_size,
            max_token_size=self.config.chunk_token_size,
        )
        
        # 重新分配索引，避免与现有chunk冲突
        next_index = max_existing_index + 1
        for chunk in new_chunks:
            chunk["index"] = next_index
            next_index += 1
        
        # 记录新chunk信息
        new_chunks_count = len(new_chunks)
        logger.info(f"新增chunk数量: {new_chunks_count}")
        logger.info(f"使用chunk配置 - max_size: {self.config.chunk_token_size}, overlap: {self.config.chunk_overlap_token_size}")
        
        # 分析新chunk分布
        if new_chunks:
            chunk_sizes = [chunk["tokens"] for chunk in new_chunks]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
            max_chunk_size = max(chunk_sizes)
            min_chunk_size = min(chunk_sizes)
            logger.info(f"新chunk大小统计 - 平均: {avg_chunk_size:.2f}, 最小: {min_chunk_size}, 最大: {max_chunk_size}")
        
        # 将新chunks添加到存储中（不影响现有数据）
        for chunk in new_chunks:
            chunk["chunk_id"] = mdhash_id(chunk["content"], prefix="chunk-")
            await self._chunk.upsert(chunk["chunk_id"], TextChunk(**chunk))
        
        # 持久化更新后的数据
        await self._chunk.persist()
        
        # 最终统计
        updated_chunks = await self.get_chunks()
        final_count = len(updated_chunks) if updated_chunks else 0
        logger.info(f"✅ 增量chunk更新完成")
        logger.info(f"📊 chunk数量变化: {existing_count} -> {final_count} (+{final_count - existing_count})")

        return new_chunks

    async def _load_chunk(self, force=False):
        if force:
            return False
        return await self._chunk.load_chunk()

    async def get_chunks(self):
        return await self._chunk.get_chunks()

    async def get_index_by_merge_key(self, chunk_id):
        return await self._chunk.get_index_by_merge_key(chunk_id)

    @property
    async def size(self):
        return await self._chunk.size()

    async def get_index_by_key(self, key):
        return await self._chunk.get_index_by_key(key)

    async def get_data_by_key(self, chunk_id):

        chunk = await self._chunk.get_by_key(chunk_id)
        return chunk.content

    async def get_data_by_index(self, index):
        chunk = await self._chunk.get_data_by_index(index)
        if chunk is None:
            # 添加日志记录，方便调试
            from Core.Common.Logger import logger
            logger.warning(f"⚠️ 索引 {index} 对应的chunk为空")
            return None
        return chunk.content

    async def get_key_by_index(self, index):
        return await self._chunk.get_key_by_index(index)

    async def get_data_by_indices(self, indices):
        return await asyncio.gather(
            *[self.get_data_by_index(index) for index in indices]
        )
