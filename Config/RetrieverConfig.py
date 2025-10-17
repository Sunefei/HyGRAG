from Core.Utils.YamlModel import YamlModel


class RetrieverConfig(YamlModel):
    # Retrieval Config
    query_type: str = "ppr"
    enable_local: bool = False
    use_entity_similarity_for_ppr: bool = True
    top_k_entity_for_ppr: int = 8
    node_specificity: bool = True
    damping: float = 0.1
    top_k: int = 5
    k_nei: int = 3
    max_token_for_local_context: int = 4800  # maximum token  * 0.4
    max_token_for_global_context: int = 4000 # maximum token  * 0.3
    local_max_token_for_text_unit: int = 4000  # 12000 * 0.33
    use_relations_vdb: bool = False
    use_subgraphs_vdb: bool = False
    global_max_consider_community: int = 512
    global_min_community_rating: float = 0.0
    
    # HK Graph PPR 特定参数
    enable_hybrid_ppr: bool = True
    ppr_iterations: int = 3
    enable_direct_chunk_ppr: bool = True
    enable_hybrid_ppr_method: bool = True
    enable_comprehensive_ppr: bool = True
    
    # PPR 权重配置
    entity_weight: float = 0.4
    chunk_weight: float = 0.3
    relationship_weight: float = 0.3
    direct_chunk_weight: float = 0.4
    entity_derived_weight: float = 0.3
    relationship_derived_weight: float = 0.3
    
    # # 实体和关系 token 限制
    # entities_max_tokens: int = 2000
    # relationships_max_tokens: int = 2000
    
    # 其他检索器配置
    use_community: bool = False
    use_keywords: bool = False
    tree_search: bool = False
    
    # HyGRAG Configuration)
    enable_hierarchical_retrieval: bool = True
    # hierarchy_search_levels: int = 3
    # community_boost_factor: float = 1.2
    hk_tree_retrieval_method: str = "hk_tree_flat_search"
    
    # 树搜索特定设置
    max_communities_per_level: int = 5
    max_expansion_depth: int = 3
    community_relevance_threshold: float = 0.3