import html
import json
import os
import pickle  # 添加pickle导入
import time
from collections import defaultdict
from typing import Any, Union, cast, Dict, Optional
import networkx as nx
import numpy as np
from pydantic import model_validator
import asyncio
from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Schema.CommunitySchema import LeidenInfo
from Core.Storage.BaseGraphStorage import BaseGraphStorage


class HKGraphTreeStorage(BaseGraphStorage):
    def __init__(self):
        super().__init__()
        self.edge_list = None
        self.node_list = None
        # 层次结构相关的存储
        self._hierarchy_data = None
        # 增量更新相关的存储
        self._incremental_data = None
        self._last_update_timestamp = None

    name: str = "nx_data.graphml"  # NetworkX基础图文件
    hierarchy_name: str = "hk_hierarchy.pkl"  # 层次结构数据文件
    incremental_name: str = "hk_incremental.pkl"  # 增量更新数据文件
    metadata_name: str = "hk_metadata.json"  # 元数据文件
    _graph: nx.Graph = nx.Graph()

    def load_nx_graph(self) -> bool:
        # Attempting to load the graph from the specified GraphML file
        logger.info(f"Attempting to load the graph from: {self.graphml_xml_file}")
        if os.path.exists(self.graphml_xml_file):
            try:
                self._graph = nx.read_graphml(self.graphml_xml_file)
                logger.info(
                    f"Successfully loaded graph from: {self.graphml_xml_file} with {self._graph.number_of_nodes()} nodes and {self._graph.number_of_edges()} edges")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to load graph from: {self.graphml_xml_file} with {e}! Need to re-build the graph.")
                return False
        else:
            # GraphML file doesn't exist; need to construct the graph from scratch
            logger.info("GraphML file does not exist! Need to build the graph from scratch.")
            return False

    def load_hierarchy_data(self) -> bool:
        """加载层次结构数据"""
        logger.info(f"Attempting to load hierarchy data from: {self.hierarchy_pkl_file}")
        if os.path.exists(self.hierarchy_pkl_file):
            try:
                with open(self.hierarchy_pkl_file, "rb") as file:
                    self._hierarchy_data = pickle.load(file)
                logger.info(
                    f"Successfully loaded hierarchy data from: {self.hierarchy_pkl_file} with {len(self._hierarchy_data.get('hierarchy_levels', {}))} levels")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to load hierarchy data from: {self.hierarchy_pkl_file} with {e}! Need to re-build the hierarchy.")
                return False
        else:
            logger.info("Hierarchy data file does not exist! Need to build the hierarchy from scratch.")
            return False

    def write_hierarchy_data(self, hierarchy_data: Dict[str, Any]):
        """保存层次结构数据"""
        logger.info(f"Writing hierarchy data to: {self.hierarchy_pkl_file}")
        with open(self.hierarchy_pkl_file, "wb") as file:
            pickle.dump(hierarchy_data, file)
        logger.info(f"Successfully wrote hierarchy data with {len(hierarchy_data.get('hierarchy_levels', {}))} levels")

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @property
    def hierarchy_data(self):
        """获取层次结构数据"""
        return self._hierarchy_data

    @property
    def hierarchy_pkl_file(self):
        """层次结构数据文件路径"""
        assert self.namespace is not None
        return self.namespace.get_save_path(self.hierarchy_name)
    
    @property
    def incremental_pkl_file(self):
        """增量更新数据文件路径"""
        assert self.namespace is not None
        return self.namespace.get_save_path(self.incremental_name)
    
    @property
    def metadata_json_file(self):
        """元数据文件路径"""
        assert self.namespace is not None
        return self.namespace.get_save_path(self.metadata_name)

    @model_validator(mode="after")
    def _register_node2emb(cls, data):
        cls._node_embed_algorithms = {
            "node2vec": data._node2vec_embed,
        }
        return data

    @property
    def graphml_xml_file(self):
        assert self.namespace is not None
        return self.namespace.get_save_path(self.name)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    async def load_graph(self, force: bool = False) -> bool:
        """加载图和层次结构数据"""
        if force:
            logger.info("Force rebuilding the graph and hierarchy")
            return False
        else:
            # 同时加载基础图和层次结构
            graph_loaded = self.load_nx_graph()
            hierarchy_loaded = self.load_hierarchy_data()
            
            # 只有当两者都成功加载时才返回True
            success = graph_loaded and hierarchy_loaded
            if success:
                logger.info("✅ Successfully loaded both graph and hierarchy data")
            elif graph_loaded and not hierarchy_loaded:
                logger.warning("⚠️ Graph loaded but hierarchy data missing - will rebuild hierarchy")
                return False
            elif not graph_loaded and hierarchy_loaded:
                logger.warning("⚠️ Hierarchy data loaded but graph missing - will rebuild both")
                return False
            else:
                logger.info("Neither graph nor hierarchy data found - will build from scratch")
                
            return success

    @property
    def graph(self):
        return self._graph

    async def _persist(self, force):
        """持久化基础图"""
        if os.path.exists(self.graphml_xml_file) and not force:
            return
        logger.info(f"Writing graph into {self.graphml_xml_file}")
        HKGraphTreeStorage.write_nx_graph(self.graph, self.graphml_xml_file)

    async def persist_hierarchy(self, hierarchy_data: Dict[str, Any], force: bool = False):
        """持久化层次结构数据"""
        if os.path.exists(self.hierarchy_pkl_file) and not force:
            logger.info("Hierarchy data already exists and force=False, skipping")
            return
        self.write_hierarchy_data(hierarchy_data)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def get_edge_weight(
            self, source_node_id: str, target_node_id: str
    ) -> Union[float, None]:
        edge_data = self._graph.edges.get((source_node_id, target_node_id))
        return edge_data.get("weight") if edge_data is not None else None

    async def get_edge(
            self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict):
        self._graph.add_node(node_id, **node_data)

    # TODO: not use dict for edge_data
    async def upsert_edge(
            self, source_node_id: str, target_node_id: str, edge_data: dict
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):

        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)
        logger.info(f"Rewrite the graph with cluster data")
        await self._persist(force=True)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        pass

    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return HKGraphTreeStorage._stabilize_graph(graph)

    async def persist(self, force):
        return await self._persist(force)

    async def get_nodes(self):
        return self._graph.nodes()


    # TODO: remove to the basegraph class
    async def get_nodes_data(self):
        """
        获取所有层次化节点数据，包括：
        1. 底层基础图节点（entities和chunks）
        2. 上层摘要节点（community summaries）
        类似TreeGraphStorage的实现，返回用于entities_vdb的数据
        """
        all_nodes_data = []
        
        # 1. 获取基础图的entity节点（底层节点）
        base_nodes = await self._get_base_graph_nodes_data()
        all_nodes_data.extend(base_nodes)
        
        # 2. 获取层次结构的摘要节点（上层节点）
        if self._hierarchy_data:
            hierarchy_nodes = await self._get_hierarchy_nodes_data()
            all_nodes_data.extend(hierarchy_nodes)
        
        logger.info(f"📊 总共准备 {len(all_nodes_data)} 个节点用于entities_vdb: "
                   f"基础节点 {len(base_nodes)}, 层次节点 {len(hierarchy_nodes) if self._hierarchy_data else 0}")
        
        return all_nodes_data

    async def _get_base_graph_nodes_data(self):
        """获取基础图节点数据"""
        node_list = list(self._graph.nodes())

        async def get_base_node_data(node_id):
            node_data = await self.get_node(node_id)
            node_data.setdefault("description", "")
            node_data.setdefault("entity_type", "")
            
            # 判断是否为chunk节点（以CHUNK开头）
            if node_id.startswith("CHUNK"):
                # 处理chunk节点
                chunk_text = node_data.get("content", node_data.get("description", ""))
                
                # 获取chunk节点的文本嵌入
                node_embedding = self._get_node_text_embedding(node_id)
                
                return {
                    "content": chunk_text,  # chunk的content就是chunk本身的文本
                    "node_id": node_id,
                    "node_type": "chunk",  # 标识为文档块节点
                    "level": 0,  # 基础层级
                    "chunk_text": chunk_text,
                    "source_id": node_data.get("source_id", ""),
                    "chunk_index": node_data.get("chunk_index", ""),
                    "index": node_id,           # ✅ 改为存储节点ID
                    "embedding": node_embedding  # ✅ embedding存储在单独字段
            
                }
            else:
                # 处理entity节点
                content_parts = []
                content_parts.append(node_data.get("entity_name", node_id))

                if node_data.get("entity_type"):
                    content_parts.append(f"entity_type: {node_data['entity_type']}")

                if node_data.get("description"):
                    content_parts.append(f"description: {node_data['description']}")

                content = ": ".join(content_parts) if content_parts else node_id
                
                # 获取entity节点的文本嵌入
                node_embedding = self._get_node_text_embedding(node_id)
                
                return {
                    "content": content,
                    "node_id": node_id,
                    "node_type": "entity",  # 标识为基础实体节点
                    "level": 0,  # 基础层级
                    "entity_name": node_data.get("entity_name", node_id),
                    "entity_type": node_data.get("entity_type", ""),
                    "description": node_data.get("description", ""),
                    "index": node_id,           # ✅ 改为存储节点ID
                    "embedding": node_embedding  # ✅ embedding存储在单独字段
                }

        nodes = await asyncio.gather(*[get_base_node_data(node) for node in node_list])
        return nodes

    async def _get_hierarchy_nodes_data(self):
        """获取层次结构节点数据"""
        if not self._hierarchy_data:
            return []
            
        hierarchy_nodes = []
        
        # 从hierarchy_data中提取信息
        community_summaries = self._hierarchy_data.get('community_summaries', {})
        community_children = self._hierarchy_data.get('community_children', {})
        hierarchy_levels = self._hierarchy_data.get('hierarchy_levels', {})
        
        for community_id, summary_text in community_summaries.items():
            # 获取层级信息
            level = None
            for lvl, communities in hierarchy_levels.items():
                if community_id in communities:
                    level = int(lvl) + 1  # +1因为基础节点是level 0
                    break
            
            if level is None:
                level = 1  # 默认层级
            
            # 获取子节点信息
            children = community_children.get(community_id, [])
            children_count = len(children)
            
            # 构建摘要节点内容
            #content_parts = [f"社区摘要 {community_id}"]
            content_parts = []
            if summary_text and summary_text.strip():
                content_parts.append(summary_text.strip())
            # if children_count > 0:
            #     content_parts.append(f"包含 {children_count} 个子节点")
                
            content = ": ".join(content_parts)
            
            # 获取community节点的文本嵌入 (community_id已经是正确的格式，如 COMMUNITY_L0_C0)
            node_embedding = self._get_node_text_embedding(community_id)
            
            hierarchy_nodes.append({
                "content": content,
                "node_id": community_id,
                "node_type": "community_summary",  # 标识为社区摘要节点
                "level": level,
                "community_id": community_id,
                "summary_text": summary_text,
                "children_count": children_count,
                "children": children,
                "index": community_id,           # ✅ 改为存储节点ID
                "embedding": node_embedding  # ✅ embedding存储在单独字段
            })
        
        return hierarchy_nodes

    async def get_edges_data(self, need_content=True):
        edge_list = list(self._graph.edges())
        edges = []

        async def get_edge_data(edge_id):
            edge_data = await self.get_edge(edge_id[0], edge_id[1])
            if need_content:
                description = edge_data.get("description", "")
                relation_name = edge_data.get("relation_name", "")
                keywords = edge_data.get("keywords", "")
                if relation_name != "":
                    edge_data["content"] = relation_name
                else:
                    edge_data["content"] = "{keywords} {src_id} {tgt_id} {description}".format(
                        keywords=keywords, src_id=edge_data["src_id"], tgt_id=edge_data["tgt_id"],
                        description=description)
            edges.append(edge_data)

        await asyncio.gather(*[get_edge_data(edge) for edge in edge_list])
        return edges

    async def get_subgraph_from_same_chunk(self):
        # origin_nodes = await self.get_nodes_data() # list[dict]
        from Core.Common.Constants import GRAPH_FIELD_SEP
        origin_edges = await self.get_edges_data() # list[dict]

        from collections import defaultdict
        # chunk_to_metagraph_nodes = defaultdict(list)  # {"chunk_id": [node1, node2,...]}
        chunk_to_metagraph_edges = defaultdict(list)  # {"chunk_id": [edge1, edge2,...]}
        # for node in origin_nodes:
        #     chunk_to_metagraph_nodes[node['source_id']].append(node)
        for edge in origin_edges:
            chunk_to_metagraph_edges[edge["source_id"]].append(edge)

        subgraphs = []
        async def get_subgraph_data(key, value):
            subgraph_context = ""
            for ed in value:
                seperated_edge = ed["relation_name"].split(GRAPH_FIELD_SEP)
                tmp = tuple(map(lambda x: ed['src_id'] + " " + x + " " + ed["tgt_id"], seperated_edge))
                tmp = "; ".join(tmp)
                subgraph_context += tmp
                subgraph_context += "; "
            subgraphs.append({"source_id":key, "content": subgraph_context})

        await asyncio.gather(*[get_subgraph_data(key, value) for key, value in chunk_to_metagraph_edges.items()])

        return subgraphs







    async def get_stable_largest_cc(self):
        return HKGraphTreeStorage.stable_largest_connected_component(self._graph)

    async def cluster_data_to_subgraphs(self, cluster_data):
        await self._cluster_data_to_subgraphs(cluster_data)

    async def get_community_schema(self):
        max_num_ids = 0
        levels = defaultdict(set)
        _schemas: dict[str, LeidenInfo] = defaultdict(LeidenInfo)
        for node_id, node_data in self._graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                _schemas[cluster_key].level = level
                _schemas[cluster_key].title = f"Cluster {cluster_key}"
                _schemas[cluster_key].nodes.add(node_id)
                _schemas[cluster_key].edges.update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                _schemas[cluster_key].chunk_ids.update(
                    node_data["source_id"].split(GRAPH_FIELD_SEP)
                )
                max_num_ids = max(max_num_ids, len(_schemas[cluster_key].chunk_ids))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                _schemas[comm].sub_communities = [
                    c
                    for c in next_level_comms
                    if _schemas[c].nodes.issubset(_schemas[comm].nodes)
                ]

        for _, v in _schemas.items():
            v.edges = list(v.edges)
            v.edges = [list(e) for e in v.edges]
            v.nodes = list(v.nodes)
            v.chunk_ids = list(v.chunk_ids)
            v.occurrence = len(v.chunk_ids) / max_num_ids
        return _schemas

    async def get_node_metadata(self) -> list[str]:
        """
        返回层次化节点的元数据字段列表
        类似TreeGraphStorage的实现，但包含更丰富的层次化信息
        """
        return [
            "node_id",        # 节点标识符（类似TreeGraphStorage的index）
            "node_type",      # 节点类型（entity/chunk/community_summary）
            "level",          # 层次级别
            "content",        # 节点内容
            "index",           # 节点的文本嵌入向量
            "embedding"
        ]

    # async def get_node_metadata(self) -> list[str]:
    #     """
    #     返回层次化节点的元数据字段列表
    #     类似TreeGraphStorage的实现，但包含更丰富的层次化信息
    #     """
    #     return [
    #         "node_id",        # 节点标识符（类似TreeGraphStorage的index）
    #         "node_type",      # 节点类型（entity/chunk/community_summary）
    #         "level",          # 层次级别
    #         "content",        # 节点内容
    #         "index"           # 节点的文本嵌入向量
    #     ]
    

    # async def get_node_metadata(self) -> list[str]:
    #     """
    #     返回层次化节点的元数据字段列表
    #     类似TreeGraphStorage的实现，但包含更丰富的层次化信息
    #     """
    #     return [
    #         "node_id",        # 节点标识符（类似TreeGraphStorage的index）
    #         "node_type",      # 节点类型（entity/chunk/community_summary）
    #         "level",          # 层次级别
    #         "entity_name",    # 实体名称（仅entity节点有）
    #         "entity_type",    # 实体类型（仅entity节点有）
    #         "source_id",      # 来源文档ID（仅chunk节点有）
    #         "chunk_index",    # 文档块索引（仅chunk节点有）
    #         "community_id",   # 社区ID（仅community_summary节点有）
    #         "children_count", # 子节点数量（仅community_summary节点有）
    #         "index"           # 节点的文本嵌入向量
    #     ]

    async def get_edge_metadata(self) -> list[str]:
        relation_metadata = ["src_id", "tgt_id"]
        return relation_metadata

    async def get_subgraph_metadata(self) -> list[str]:
        return ["source_id"]

    def get_node_num(self):
        return self._graph.number_of_nodes()

    def get_edge_num(self):
        return self._graph.number_of_edges()

    async def nodes(self):
        return self._graph.nodes()

    async def edges(self):
        return self._graph.edges()

    async def neighbors(self, node_id):
        return self._graph.neighbors(node_id)

    def get_edge_index(self, src_id, tgt_id):
        if self.edge_list is None:
            self.edge_list = list(self._graph.edges())
        try:
            return self.edge_list.index((src_id, tgt_id))
        except ValueError:
            return -1

    async def get_induced_subgraph(self, nodes: list[str]):
        return self._graph.subgraph(nodes)

    async def get_node_index(self, node_id):
        if self.node_list is None:
            self.node_list = list(self._graph.nodes())
        try:
            return self.node_list.index(node_id)
        except ValueError:
            logger.error(f"Node {node_id} not in graph")
            return None #TODO HKGraphPPR comprehensive

    async def get_node_by_index(self, index):
        if self.node_list is None:
            self.node_list = list(self._graph.nodes())
        return await self.get_node(self.node_list[index])

    async def get_edge_by_index(self, index):
        if self.edge_list is None:
            self.edge_list = list(self._graph.edges())

        return await self.get_edge(self.edge_list[index][0], self.edge_list[index][1])

    async def find_k_hop_neighbors(self, start_node: str, k: int) -> set:
        """
        find the k hop neighbors about the given input node
        :param start_node: str, entity_name
        :param k: int, k hop value
        :return: K hop sets(including start_node)
        """
        if k < 1:
            raise ValueError("K-hop neighbours value must greater than 1.")

        visited = set()
        current_level = {start_node}

        for _ in range(k):
            next_level = set()  # 下一层的节点
            for node in current_level:
                neighbors = set(await self.neighbors(node))
                next_level.update(neighbors - visited)
            visited.update(next_level)
            current_level = next_level

        return current_level

    async def find_k_hop_neighbors_batch(self, start_nodes: list[str], k: int) -> set:
        nodes_set_list = await asyncio.gather(
            *[self.find_k_hop_neighbors(node, k) for node in start_nodes]
        )
        nodes_list = []
        for node_set in nodes_set_list:
            nodes_list.extend(list(node_set))
        return set(nodes_list)

    async def get_edge_relation_name(self, source_node_id: str, target_node_id: str):
        edge_data = self._graph.edges.get((source_node_id, target_node_id))
        return edge_data.get("relation_name") if edge_data is not None else None

    async def get_edge_relation_name_batch(self, edges: list[tuple[str, str]]):
        relations = await asyncio.gather(
            *[self.get_edge_relation_name(edge[0], edge[1]) for edge in edges]
        )
        return relations

    async def get_one_path(self, start: str, cand: list[str], cutoff: int = 5):
        pred, dist = nx.dijkstra_predecessor_and_distance(self._graph, source = start, cutoff = cutoff, weight = None)
        end = None
        for node, dis in dist.items():
            if (node in cand) and (end is None or dis < dist[end]): end = node
        if end is None: return None

        # import pdb
        # pdb.set_trace()

        path = []
        cur = end
        while cur != start:
            path.append(await self.get_edge(pred[cur][0], cur))
            cur = pred[cur][0]
        # import pdb
        # pdb.set_trace()
        return end, path[::-1]

    async def get_paths_from_sources(self, start_nodes: list[str], cutoff: int = 5) -> list[tuple[str, str, str]]:
        # import pdb
        # pdb.set_trace()
        cand = set(start_nodes)
        paths = []
        while len(cand) != 0:
            start = next(iter(cand))
            cand.remove(start)
            
            path_concat = []
            while True:
                result = await self.get_one_path(start, cand, cutoff)
                if result is None: break
                end, path = result
                # import pdb
                # pdb.set_trace()
                path_concat.extend(path)
                cand.remove(end)
            
            if (len(path_concat)): paths.append(path_concat)

        return paths

    async def get_neighbors_from_sources(self, start_nodes: list[str]):
        # import pdb
        # pdb.set_trace()
        neighbor_list = []
        neighbor_list_cand = []
        for u in start_nodes:
            neis = [(await self.get_edge(e[0], e[1]))["tgt_id"] for e in await self.get_node_edges(u)]
            neighbor_list.extend([(await self.get_edge(e[0], e[1])) for e in await self.get_node_edges(u)])

            while neis != []:
                inter = list(set(neis) & set(start_nodes))
                new_neis = []

                if len(inter) != 0:
                    for v in inter:
                        new_neis.extend([(await self.get_edge(e[0], e[1]))["tgt_id"] for e in await self.get_node_edges(v)])
                        neighbor_list_cand.extend([(await self.get_edge(e[0], e[1])) for e in await self.get_node_edges(v)])
                else:
                    for v in neis:
                        new_neis.extend([(await self.get_edge(e[0], e[1]))["tgt_id"] for e in await self.get_node_edges(v)])
                        neighbor_list_cand.extend([(await self.get_edge(e[0], e[1])) for e in await self.get_node_edges(v)])
                if len(neighbor_list_cand) > 10:
                    break
                neis = new_neis
        if len(neighbor_list)<=5:
            neighbor_list.extend(neighbor_list_cand)
        # import pdb
        # pdb.set_trace()
        return neighbor_list

    # ==================== 层次化数据支持方法 ====================
    
    async def get_nodes_by_level(self, level: int):
        """根据层次级别获取节点"""
        all_nodes = await self.get_nodes_data()
        return [node for node in all_nodes if node.get("level") == level]
    
    async def get_nodes_by_type(self, node_type: str):
        """根据节点类型获取节点"""
        all_nodes = await self.get_nodes_data()
        return [node for node in all_nodes if node.get("node_type") == node_type]
        
    async def get_entity_nodes(self):
        """获取所有实体节点（底层基础节点）"""
        return await self.get_nodes_by_type("entity")
        
    async def get_community_summary_nodes(self):
        """获取所有社区摘要节点（上层摘要节点）"""
        return await self.get_nodes_by_type("community_summary")
        
    async def get_hierarchy_levels(self):
        """获取所有层次级别"""
        if not self._hierarchy_data:
            return [0]  # 只有基础层级
        
        levels = set([0])  # 基础层级
        hierarchy_levels = self._hierarchy_data.get('hierarchy_levels', {})
        for level in hierarchy_levels.keys():
            levels.add(int(level) + 1)
        
        return sorted(list(levels))
        
    async def get_community_info(self, community_id: str):
        """获取特定社区的详细信息"""
        if not self._hierarchy_data:
            return None
            
        community_summaries = self._hierarchy_data.get('community_summaries', {})
        community_children = self._hierarchy_data.get('community_children', {})
        community_parents = self._hierarchy_data.get('community_parents', {})
        
        if community_id not in community_summaries:
            return None
            
        return {
            "community_id": community_id,
            "summary": community_summaries.get(community_id, ""),
            "children": community_children.get(community_id, []),
            "parents": community_parents.get(community_id, []),
            "children_count": len(community_children.get(community_id, []))
        }
    
    async def get_hierarchy_statistics(self):
        """获取层次结构统计信息"""
        stats = {
            "total_nodes": 0,
            "base_nodes": 0,
            "summary_nodes": 0,
            "levels": 0,
            "levels_distribution": {}
        }
        
        all_nodes = await self.get_nodes_data()
        stats["total_nodes"] = len(all_nodes)
        
        for node in all_nodes:
            level = node.get("level", 0)
            node_type = node.get("node_type", "unknown")
            
            if node_type == "entity":
                stats["base_nodes"] += 1
            elif node_type == "community_summary":
                stats["summary_nodes"] += 1
                
            if level not in stats["levels_distribution"]:
                stats["levels_distribution"][level] = 0
            stats["levels_distribution"][level] += 1
        
        stats["levels"] = len(stats["levels_distribution"])
        
        return stats

    def _get_node_text_embedding(self, node_id: str):
        """
        获取节点的文本嵌入，如果没有则返回None
        """
        if not self._hierarchy_data:
            return None
            
        node_text_embeddings = self._hierarchy_data.get('node_text_embeddings', {})
        embedding = node_text_embeddings.get(node_id)
        
        if embedding is not None:
            # 如果是numpy数组，转换为列表以便序列化
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            return embedding
        
        return None

    def clear(self):
        self._graph = nx.Graph()
        self._hierarchy_data = None
        self._incremental_data = None
        self._last_update_timestamp = None

    # ==================== 增量更新数据管理方法 ====================
    
    def load_incremental_data(self) -> bool:
        """加载增量更新数据"""
        logger.info(f"Attempting to load incremental data from: {self.incremental_pkl_file}")
        if os.path.exists(self.incremental_pkl_file):
            try:
                with open(self.incremental_pkl_file, "rb") as file:
                    self._incremental_data = pickle.load(file)
                logger.info(f"Successfully loaded incremental data")
                return True
            except Exception as e:
                logger.error(f"Failed to load incremental data from: {self.incremental_pkl_file} with {e}")
                return False
        else:
            logger.info("Incremental data file does not exist")
            return False

    def write_incremental_data(self, incremental_data: Dict[str, Any]):
        """保存增量更新数据"""
        logger.info(f"Writing incremental data to: {self.incremental_pkl_file}")
        # 添加时间戳
        incremental_data['timestamp'] = time.time()
        incremental_data['last_modified'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.incremental_pkl_file, "wb") as file:
            pickle.dump(incremental_data, file)
        self._incremental_data = incremental_data
        self._last_update_timestamp = incremental_data['timestamp']
        logger.info(f"Successfully wrote incremental data")

    async def persist_incremental(self, incremental_data: Dict[str, Any], force: bool = False):
        """持久化增量数据"""
        if os.path.exists(self.incremental_pkl_file) and not force:
            logger.info("Incremental data already exists and force=False, skipping")
            return
        self.write_incremental_data(incremental_data)

    def load_metadata(self) -> Dict[str, Any]:
        """加载元数据"""
        if os.path.exists(self.metadata_json_file):
            try:
                with open(self.metadata_json_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from {self.metadata_json_file}")
                return metadata
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                return {}
        return {}

    def save_metadata(self, metadata: Dict[str, Any]):
        """保存元数据"""
        metadata['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        metadata['timestamp'] = time.time()
        
        try:
            with open(self.metadata_json_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {self.metadata_json_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def get_incremental_data(self) -> Optional[Dict[str, Any]]:
        """获取增量数据"""
        return self._incremental_data

    def get_last_update_timestamp(self) -> Optional[float]:
        """获取最后更新时间戳"""
        return self._last_update_timestamp

    async def load_full_graph_with_incremental(self, force: bool = False) -> bool:
        """
        加载完整图结构包括增量数据
        """
        if force:
            logger.info("Force rebuilding the graph with incremental data")
            return False
        
        # 加载基础图
        graph_loaded = self.load_nx_graph()
        # 加载层次结构
        hierarchy_loaded = self.load_hierarchy_data()
        # 加载增量数据
        incremental_loaded = self.load_incremental_data()
        
        success = graph_loaded and hierarchy_loaded
        if success:
            logger.info("✅ Successfully loaded graph and hierarchy data")
            if incremental_loaded:
                logger.info("✅ Also loaded incremental data")
            else:
                logger.info("⚠️ No incremental data found")
        
        return success

    def has_incremental_data(self) -> bool:
        """检查是否存在增量数据"""
        return os.path.exists(self.incremental_pkl_file)

    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        stats = {
            'graph_exists': os.path.exists(self.graphml_xml_file),
            'hierarchy_exists': os.path.exists(self.hierarchy_pkl_file),
            'incremental_exists': os.path.exists(self.incremental_pkl_file),
            'metadata_exists': os.path.exists(self.metadata_json_file),
            'graph_nodes': self._graph.number_of_nodes() if self._graph else 0,
            'graph_edges': self._graph.number_of_edges() if self._graph else 0,
            'hierarchy_levels': len(self._hierarchy_data.get('hierarchy_levels', {})) if self._hierarchy_data else 0,
            'last_update': None
        }
        
        # 获取最后更新时间
        if self._incremental_data and 'last_modified' in self._incremental_data:
            stats['last_update'] = self._incremental_data['last_modified']
        
        return stats
