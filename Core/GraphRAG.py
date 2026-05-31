from typing import Union, Any
from Core.Chunk.DocChunk import DocChunk
from Core.Common.Logger import logger
import tiktoken
from pydantic import BaseModel, model_validator
from Core.Common.ContextMixin import ContextMixin
from Core.Schema.RetrieverContext import RetrieverContext
from Core.Common.TimeStatistic import TimeStatistic
from Core.Graph import get_graph
from Core.Index import get_index, get_index_config
from Core.Query import get_query
from Core.Storage.NameSpace import Workspace
from Core.Community.ClusterFactory import get_community
from Core.Storage.PickleBlobStorage import PickleBlobStorage


class GraphRAG(ContextMixin, BaseModel):
    """Graph-based Retrieval-Augmented Generation system."""

    def __init__(self, config):
        super().__init__(config=config)

    @model_validator(mode="after")
    def _update_context(cls, data):
        cls.ENCODER = tiktoken.encoding_for_model(data.config.token_model)
        cls.workspace = Workspace(data.config.working_dir, data.config.index_name)
        cls.graph = get_graph(data.config, llm=data.llm, encoder=cls.ENCODER)
        cls.doc_chunk = DocChunk(data.config.chunk, cls.ENCODER, data.workspace.make_for("chunk_storage"))
        cls.time_manager = TimeStatistic()
        cls.retriever_context = RetrieverContext()
        data = cls._init_storage_namespace(data)
        data = cls._register_vdbs(data)
        data = cls._register_community(data)
        data = cls._register_e2r_r2c_matrix(data)
        data = cls._register_retriever_context(data)
        return data

    @classmethod
    def _init_storage_namespace(cls, data):
        data.graph.namespace = data.workspace.make_for("graph_storage")
        if data.config.use_entities_vdb:
            data.entities_vdb_namespace = data.workspace.make_for("entities_vdb")
        if data.config.use_relations_vdb:
            data.relations_vdb_namespace = data.workspace.make_for("relations_vdb")
        if data.config.use_subgraphs_vdb:
            data.subgraphs_vdb_namespace = data.workspace.make_for("subgraphs_vdb")
        if data.config.graph.use_community:
            data.community_namespace = data.workspace.make_for("community_storage")
        if data.config.use_entity_link_chunk:
            data.e2r_namespace = data.workspace.make_for("map_e2r")
            data.r2c_namespace = data.workspace.make_for("map_r2c")
        return data

    @classmethod
    def _register_vdbs(cls, data):
        if data.config.use_entities_vdb:
            cls.entities_vdb = get_index(
                get_index_config(data.config, persist_path=data.entities_vdb_namespace.get_save_path()))
        if data.config.use_relations_vdb:
            cls.relations_vdb = get_index(
                get_index_config(data.config, persist_path=data.relations_vdb_namespace.get_save_path()))
        if data.config.use_subgraphs_vdb:
            cls.subgraphs_vdb = get_index(
                get_index_config(data.config, persist_path=data.subgraphs_vdb_namespace.get_save_path()))
        return data

    @classmethod
    def _register_community(cls, data):
        if data.config.graph.use_community:
            cls.community = get_community(data.config.graph.graph_cluster_algorithm,
                                          enforce_sub_communities=data.config.graph.enforce_sub_communities,
                                          llm=data.llm, namespace=data.community_namespace)
        return data

    @classmethod
    def _register_e2r_r2c_matrix(cls, data):
        if data.config.graph.graph_type == "tree_graph":
            logger.warning("Tree graph does not support entity-link-chunk mapping. Skipping.")
            data.config.use_entity_link_chunk = False
            return data
        if data.config.use_entity_link_chunk:
            cls.entities_to_relationships = PickleBlobStorage(
                namespace=data.e2r_namespace, config=None
            )
            cls.relationships_to_chunks = PickleBlobStorage(
                namespace=data.r2c_namespace, config=None
            )
        return data

    @classmethod
    def _register_retriever_context(cls, data):
        cls._retriever_context = {
            "config": True,
            "graph": True,
            "doc_chunk": True,
            "llm": True,
            "entities_vdb": data.config.use_entities_vdb,
            "relations_vdb": data.config.use_relations_vdb,
            "subgraphs_vdb": data.config.use_subgraphs_vdb,
            "community": data.config.graph.use_community,
            "relationships_to_chunks": data.config.use_entity_link_chunk,
            "entities_to_relationships": data.config.use_entity_link_chunk,
        }
        return data

    async def _build_retriever_context(self):
        logger.info("Building retriever context...")
        try:
            for context_name, use_context in self._retriever_context.items():
                if use_context:
                    config_value = getattr(self, context_name)
                    if context_name == "config":
                        config_value = self.config.retriever
                    self.retriever_context.register_context(context_name, config_value)
            self._querier = get_query(self.config.retriever.query_type, self.config.query, self.retriever_context)
        except Exception as e:
            logger.error(f"Failed to build retriever context: {e}")
            raise

    async def build_e2r_r2c_maps(self, force=False):
        logger.info("Building entity<->relationship and relationship<->chunk maps...")
        if not await self.entities_to_relationships.load(force):
            await self.entities_to_relationships.set(await self.graph.get_entities_to_relationships_map(False))
            await self.entities_to_relationships.persist()
        if not await self.relationships_to_chunks.load(force):
            await self.relationships_to_chunks.set(await self.graph.get_relationships_to_chunks_map(self.doc_chunk))
            await self.relationships_to_chunks.persist()
        logger.info("Finished building entity-relationship-chunk maps")

    def _update_costs_info(self, stage_str: str):
        last_cost = self.llm.get_last_stage_cost()
        logger.info(f"{stage_str} stage cost - prompt tokens: {last_cost.total_prompt_tokens}, "
                    f"completion tokens: {last_cost.total_completion_tokens}, total cost: {last_cost.total_cost}")
        last_stage_time = self.time_manager.stop_last_stage()
        logger.info(f"{stage_str} time: {last_stage_time:.2f}s")

    async def insert(self, docs: Union[str, list[Any]]):
        self.time_manager.start_stage()
        await self.doc_chunk.build_chunks(docs)
        self._update_costs_info("Chunking")

        await self.graph.build_graph(await self.doc_chunk.get_chunks(), self.config.graph.force)
        self._update_costs_info("Build Graph")

        if self.config.use_entities_vdb:
            node_metadata = await self.graph.node_metadata()
            if not node_metadata:
                logger.warning("No node metadata found. Skipping entity indexing.")
            await self.entities_vdb.build_index(await self.graph.nodes_data(), node_metadata, False)

        if self.config.enable_graph_augmentation:
            await self.graph.augment_graph_by_similarity_search(self.entities_vdb)

        if self.config.use_entity_link_chunk:
            await self.build_e2r_r2c_maps(True)

        if self.config.use_relations_vdb:
            edge_metadata = await self.graph.edge_metadata()
            if not edge_metadata:
                logger.warning("No edge metadata found. Skipping relation indexing.")
                return
            await self.relations_vdb.build_index(await self.graph.edges_data(), edge_metadata, force=False)

        if self.config.use_subgraphs_vdb:
            subgraph_metadata = await self.graph.subgraph_metadata()
            if not subgraph_metadata:
                logger.warning("No subgraph metadata found. Skipping subgraph indexing.")
            await self.subgraphs_vdb.build_index(await self.graph.subgraphs_data(), subgraph_metadata, force=False)

        if self.config.graph.use_community:
            await self.community.cluster(largest_cc=await self.graph.stable_largest_cc(),
                                         max_cluster_size=self.config.graph.max_graph_cluster_size,
                                         random_seed=self.config.graph.graph_cluster_seed, force=False)
            await self.community.generate_community_report(self.graph, False)
        self._update_costs_info("Index Building")

        await self._build_retriever_context()

    async def query(self, query):
        response = await self._querier.query(query)
        logger.info("Query processing completed")
        return response
