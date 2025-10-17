from typing import Optional

from Core.Utils.YamlModel import YamlModel


class GraphConfig(YamlModel):
    graph_type: str = "er_graph"
    # Building graph
    extract_two_step: bool = True
    max_gleaning: int = 1
    force: bool = False

    # For ER graph & KG graph and & RKG graph
    enable_entity_description: bool = False
    enable_entity_type: bool = False
    enable_edge_description: bool = False
    enable_edge_name: bool = False
    prior_prob: float = 0.8
    use_wat_linking: bool = False  # 是否使用WAT实体链接系统
    enable_edge_keywords: bool = False
    # Graph clustering
    use_community: bool = False  # Default to False
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    summary_max_tokens: int = 500
    llm_model_max_token_size: int = 32768

    # For Tree graph config 
    build_tree_from_leaves: bool = False
    reduction_dimension: int = 5
    summarization_length: int = 100
    num_layers: int = 10
    top_k: int = 5
    threshold_cluster_num: int = 5000
    start_layer: int = 5
    graph_cluster_params: Optional[dict] = None
    selection_mode: str = "top_k"
    max_length_in_cluster: int = 3500
    threshold: float = 0.1
    cluster_metric: str = "cosine"
    verbose: bool = False
    random_seed: int = 224
    enforce_sub_communities: bool = False
    max_size_percentage: float = 0.2
    tol: float = 1e-4
    max_iter: int = 300
    size_of_clusters: int = 10

    # For graph augmentation
    similarity_threshold: float = 0.8
    similarity_top_k: int = 10
    similarity_max: float = 1.0
    
    # HyGRAG config
    # Cleora config
    cleora_dim: int = 1024
    cleora_iterations: int = 2
    
    # LSH config
    lsh_num_hyperplanes: int = 16
    lsh_min_cluster_size: int = 5
    lsh_max_cluster_size: int = 50
    
    # 层次化配置
    max_hierarchy_levels: int = 4
    community_summary_length: int = 300
    max_concurrent_summaries: int = 35
    
    # Chunk-Chunk连接设置
    shared_entity_threshold: int = 1

    # FAISS config
    faiss_index_type: str = "HNSW"                   
    faiss_hnsw_m: int = 64                           
    faiss_hnsw_ef_construction: int = 500            
    faiss_hnsw_ef_search: int = 200                   
    faiss_index_path: str = "./faiss_index_th3/Popqa"     

    # Incremental Update Configuration
    enable_incremental_update: bool = False          
    incremental_batch_size: int = 10                
    max_affected_ratio: float = 0.5                
    incremental_force_rebuild_threshold: float = 0.5 