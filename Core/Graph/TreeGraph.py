from Core.Graph.BaseGraph import BaseGraph
from Core.Schema.ChunkSchema import TextChunk
from Core.Common.Logger import logger
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Prompt.RaptorPrompt import SUMMARIZE
from Core.Storage.TreeGraphStorage import TreeGraphStorage
from Core.Schema.TreeSchema import TreeNode

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from typing import List, Set, Any

Embedding = List[float]

import numpy as np
import umap
import random
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import multiprocessing as mp

class TreeGraph(BaseGraph):
    max_workers: int = 48  # 增加到48个worker
    leaf_workers: int = 56  # 增加到56个worker
    def __init__(self, config, llm, encoder):
        super().__init__(config, llm, encoder)
        self._graph: TreeGraphStorage = TreeGraphStorage()  # Tree index
        self.embedding_model = get_rag_embedding(config.embedding.api_type, config)  # Embedding model
        self.config = config.graph # Only keep the graph config
        random.seed(self.config.random_seed)
        # 设置joblib的并行度
        self.n_jobs = min(48, mp.cpu_count())

    def _parallel_gmm_fit(self, embeddings, n_components, random_state):
        """拟合GMM模型"""
        gm = GaussianMixture(
            n_components=n_components, 
            random_state=random_state,
            n_init=3,  # 减少初始化次数以提高速度
            max_iter=100,  # 限制最大迭代次数
            tol=1e-3,  # 稍微放宽收敛条件
            init_params='kmeans'  # 使用kmeans初始化，通常更快
        )
        gm.fit(embeddings)
        return gm

    def _compute_bic_for_n_components(self, embeddings, n_components, random_state):
        """计算单个聚类数的BIC值"""
        gm = self._parallel_gmm_fit(embeddings, n_components, random_state)
        return n_components, gm.bic(embeddings)

    def _parallel_bic_computation(self, embeddings, n_clusters_range, random_state):
        """使用多线程计算BIC值"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        with ThreadPoolExecutor(max_workers=min(self.n_jobs, len(n_clusters_range))) as executor:
            # 提交所有任务
            future_to_n = {
                executor.submit(self._compute_bic_for_n_components, embeddings, n, random_state): n 
                for n in n_clusters_range
            }
            
            # 收集结果
            for future in as_completed(future_to_n):
                n_components, bic_value = future.result()
                results.append((n_components, bic_value))
        
        # 按n_components排序
        results.sort(key=lambda x: x[0])
        n_clusters_list, bics = zip(*results)
        return list(n_clusters_list), list(bics)

    async def _GMM_cluster(self, embeddings: np.ndarray, threshold: float, random_state: int = 0):
        """优化的GMM聚类方法"""
        if len(embeddings) > self.config.threshold_cluster_num:
            max_clusters = len(embeddings) // 100
            n_clusters = np.arange(max_clusters - 1, max_clusters)
        else:
            max_clusters = min(50, len(embeddings))
            n_clusters = np.arange(1, max_clusters)
        
        # 并行计算BIC值
        n_clusters_list, bics = self._parallel_bic_computation(embeddings, n_clusters, random_state)
        optimal_clusters = n_clusters_list[np.argmin(bics)]

        # 使用最优聚类数拟合最终模型
        gm = self._parallel_gmm_fit(embeddings, optimal_clusters, random_state)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, optimal_clusters

    def _create_task_for(self, func):
        def _pool_func(**params):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(func(**params))
            loop.close()
        return _pool_func

    def _create_task_with_return(self, func):
        def _pool_func(**params):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(func(**params))
            loop.close()
            return result
        return _pool_func

    async def _process_cluster(self, i, global_clusters, embeddings, dim, threshold):
        logger.info("Processing cluster i={i}", i=i)
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            return
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # 优化UMAP参数以提高速度
            reduced_embeddings_local = umap.UMAP(
                n_neighbors=min(10, len(global_cluster_embeddings_) - 1), 
                n_components=dim, 
                metric=self.config.cluster_metric,
                low_memory=False,  # 启用低内存模式
                random_state=42
            ).fit_transform(global_cluster_embeddings_)
            
            local_clusters, n_local_clusters = await self._GMM_cluster(
                reduced_embeddings_local, threshold
            )

        return i, local_clusters, n_local_clusters

    async def _perform_clustering(
        self, embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
    ) -> List[np.ndarray]:
        logger.info("Length of embeddings: {length}".format(length=len(embeddings)))
        logger.info("Starting UMAP")
        # 优化UMAP参数
        n_neighbors = min(int((len(embeddings) - 1) ** 0.5), len(embeddings) - 1)
        n_components = min(dim, len(embeddings) - 2)
        
        reduced_embeddings_global = umap.UMAP(
            n_neighbors=n_neighbors, 
            n_components=n_components, 
            metric=self.config.cluster_metric,
            low_memory=False,  # 启用低内存模式
            random_state=42
        ).fit_transform(embeddings)

        logger.info("Finished UMAP")
        logger.info("Starting GMM clustering")
        global_clusters, n_global_clusters = await self._GMM_cluster(
            reduced_embeddings_global, threshold
        )
        
        logger.info("Finished GMM clustering, {n} clusters".format(n=n_global_clusters))

        if verbose:
            logger.info(f"Global Clusters: {n_global_clusters}")

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        completed_list = []

        # 优化线程池使用
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # 创建所有任务
            cluster_tasks = [
                pool.submit(
                    self._create_task_with_return(self._process_cluster),
                    i=i,
                    global_clusters=global_clusters,
                    embeddings=embeddings,
                    dim=dim,
                    threshold=threshold
                ) for i in range(n_global_clusters)
            ]
            
            # 等待所有任务完成
            completed_list = list(as_completed(cluster_tasks))

        for task in completed_list:
            result = task.result()
            if result is None:
                continue
            i, local_clusters, n_local_clusters = result
            global_indices = np.where(np.array([i in gc for gc in global_clusters]))[0]
            
            for j in range(n_local_clusters):
                indices = global_indices[np.array([j in lc for lc in local_clusters])]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local_clusters

        logger.info(f"Total Clusters: {total_clusters}")
        return all_local_clusters


    async def _clustering(self, nodes: List[TreeNode], max_length_in_cluster, tokenizer, reduction_dimension, threshold, verbose, depth: int = 0) -> List[List[TreeNode]]:
        logger.info("Clustering: dep = {depth}", depth=depth)
        if depth >= 20: return [nodes]
        
        # Get the embeddings from the nodes
        embeddings = np.array([node.embedding for node in nodes])


        # Perform the clustering
        clusters = await self._perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )

        # Initialize an empty list to store the clusters of nodes
        node_clusters = []

        if len(np.unique(np.concatenate(clusters))) == 1:
            logger.info("Only one cluster length = {len}, return".format(len = len(nodes)))
            return [nodes]

        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # Calculate the total length of the text in the nodes
            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )

            # If the total length exceeds the maximum allowed length, recluster this cluster
            if total_length > max_length_in_cluster and len(cluster_nodes) > self.config.reduction_dimension + 1:
                if verbose:
                    logger.info(
                        f"reclustering cluster with {len(cluster_nodes)} nodes"
                    )
    
                node_clusters.extend(
                    await self._clustering(
                        cluster_nodes, max_length_in_cluster, tokenizer, reduction_dimension, threshold, verbose, depth + 1
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters

    def _embed_text(self, text: str):
        return self.embedding_model._get_text_embedding(text)

    async def _create_node(self, layer: int, text: str, children_indices: Set[int] = None):
        embedding = self._embed_text(text)
        node_id = self._graph.num_nodes  # Give it an index
        logger.info(
            "Create node_id = {node_id}, children = {children}".format(node_id=node_id, children=children_indices))
        return self._graph.upsert_node(node_id=node_id,
                                       node_data={"layer": layer, "text": text, "children": children_indices,
                                                  "embedding": embedding})

    async def _extract_entity_relationship(self, chunk_key_pair: tuple[str, TextChunk]) -> TreeNode:
        # Build a leaf node from a text chunk
        chunk_key, chunk_info = chunk_key_pair
        leaf_node = await self._create_node(0, chunk_info.content)
        return leaf_node

    async def _extract_cluster_relationship(self, layer: int, cluster: List[TreeNode]) -> TreeNode:
        # Build a non-leaf node from a cluster of nodes
        summarized_text = await self._summarize_from_cluster(cluster, self.config.summarization_length)
        parent_node = await self._create_node(layer, summarized_text, {node.index for node in cluster})
        return parent_node

    async def _create_node_without_embedding(self, layer: int, text: str, children_indices: Set[int] = None):
        # embedding = self._embed_text(text)
        logger.info(
            "Create node_id = unassigned, children = {children}".format(node_id=0, children=children_indices))
        return self._graph.upsert_node(node_id=0,
                                       node_data={"layer": layer, "text": text, "children": children_indices,
                                                  "embedding": []})

    async def _extract_entity_relationship_without_embedding(self, chunk_key_pair: tuple[str, TextChunk]) -> TreeNode:
        # Build a leaf node from a text chunk
        chunk_key, chunk_info = chunk_key_pair
        leaf_node = await self._create_node_without_embedding(0, chunk_info.content)
        return leaf_node

    async def _extract_cluster_relationship_without_embedding(self, layer: int, cluster: List[TreeNode]) -> TreeNode:
        # Build a non-leaf node from a cluster of nodes
        summarized_text = await self._summarize_from_cluster(cluster, self.config.summarization_length)
        parent_node = await self._create_node_without_embedding(layer, summarized_text, {node.index for node in cluster})
        return parent_node

    async def _summarize_from_cluster(self, node_list: List[TreeNode], summarization_length=150) -> str:
        # Give a summarization from a cluster of nodes
        node_texts = f"\n\n".join([' '.join(node.text.splitlines()) for node in node_list])
        content = SUMMARIZE.format(context=node_texts)
        return await self.llm.aask(content, max_tokens=summarization_length)

    async def _batch_embed_and_assign(self, layer):
        current_layer = self._graph.get_layer(layer)
        texts = [node.text for node in current_layer]
        # For openai embedding model 
        embeddings = []
        batch_size = self.embedding_model.embed_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model._get_text_embeddings(batch)
            embeddings.extend(batch_embeddings)
        # embeddings = self.embedding_model._get_text_embeddings(texts)
        start_id = self._graph.get_node_num() - len(self._graph.get_layer(layer))
        for i in range(start_id, len(self._graph.nodes)):
            self._graph.nodes[i].id = i
            self._graph.nodes[i].embedding = embeddings[i - start_id]
        for node, embedding in zip(self._graph.get_layer(layer), embeddings):
            node.embeddings = embedding
            node.index = start_id
            start_id += 1

    async def _build_tree_from_leaves(self):
        for layer in range(self.config.num_layers):  # build a new layer
            logger.info("length of layer: {length}".format(length=len(self._graph.get_layer(layer))))
            if len(self._graph.get_layer(layer)) <= self.config.reduction_dimension + 1:
                break

            self._graph.add_layer()

            clusters = await self._clustering(
                nodes = self._graph.get_layer(layer),
                max_length_in_cluster =  self.config.max_length_in_cluster,
                tokenizer = self.ENCODER,
                reduction_dimension = self.config.reduction_dimension,
                threshold = self.config.threshold,
                verbose = self.config.verbose,
            )

            # 使用asyncio并发而不是ThreadPoolExecutor来避免event loop冲突
            logger.info(f"Processing {len(clusters)} clusters using asyncio concurrency...")
            
            # 创建异步任务列表
            async_tasks = []
            for cluster in clusters:
                task = self._extract_cluster_relationship_without_embedding(layer + 1, cluster)
                async_tasks.append(task)
            
            # 使用asyncio.gather并发执行所有任务
            logger.info(f"Waiting for {len(async_tasks)} cluster tasks to complete...")
            await asyncio.gather(*async_tasks)
            logger.info(f"All {len(async_tasks)} cluster tasks completed successfully")

            logger.info("To batch embed current layer")
            await self._batch_embed_and_assign(self._graph.num_layers - 1)
            # for cluster in clusters:  # for each cluster, create a new node
            #     await self._extract_cluster_relationship(layer + 1, cluster)

            logger.info("Layer: {layer}".format(layer=layer))
            # logger.info(self._graph.get_layer(layer + 1))

        logger.info(self._graph.num_layers)
        

    async def _build_graph(self, chunks: List[Any]):
        if self.config.build_tree_from_leaves:
            await self._graph.load_tree_graph_from_leaves()
            logger.info(f"Loaded {len(self._graph.leaf_nodes)} Leaf Embeddings")
        else:
            self._graph.clear()  # clear the storage before rebuilding
            self._graph.add_layer()
            # 使用asyncio并发而不是ThreadPoolExecutor来避免event loop冲突
            logger.info(f"Processing {len(chunks)} chunks using asyncio concurrency...")
            
            # 创建异步任务列表
            async_tasks = []
            for chunk in chunks:
                task = self._extract_entity_relationship_without_embedding(chunk_key_pair=chunk)
                async_tasks.append(task)
            
            # 使用asyncio.gather并发执行所有任务
            logger.info(f"Waiting for {len(async_tasks)} leaf tasks to complete...")
            await asyncio.gather(*async_tasks)
            logger.info(f"All {len(async_tasks)} leaf tasks completed successfully")
            logger.info(len(chunks))
            logger.info(f"To batch embed leaves")
            await self._batch_embed_and_assign(self._graph.num_layers - 1)
            logger.info(f"Created {len(self._graph.leaf_nodes)} Leaf Embeddings")
            await self._graph.write_tree_leaves()
        await self._build_tree_from_leaves()
        
    @property
    def entity_metakey(self):
        return "index"
