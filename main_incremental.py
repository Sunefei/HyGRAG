"""
HKGraphTree增量更新测试程序

测试EraRAG TreeGraphDynamic增量更新算法在HKGraphTree中的集成效果

使用方法:
1. 初始构建: python main_incremental.py -opt Option/Ours/HKGraphTreeDynamic.yaml -dataset_name multihop-rag -mode build
2. 增量更新: python main_incremental.py -opt Option/Ours/HKGraphTreeDynamic.yaml -dataset_name multihop-rag -mode incremental
3. 性能测试: python main_incremental.py -opt Option/Ours/HKGraphTreeDynamic.yaml -dataset_name multihop-rag -mode benchmark
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
warnings.filterwarnings("ignore")
import nltk
nltk.data.path.append('/data/zhy/nltk_data')

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
import argparse
import asyncio
from pathlib import Path
from shutil import copyfile
from Data.QueryDataset import RAGQueryDataset
import pandas as pd
from Core.Utils.Evaluation import Evaluator
import time
import json
from Core.Common.Logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="HKGraphTree增量更新测试程序")
    parser.add_argument("-opt", type=str, required=True, help="配置文件路径")
    parser.add_argument("-dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("-mode", type=str, choices=["build", "incremental", "benchmark", "query"], 
                       default="build", help="运行模式")
    parser.add_argument("-incremental_ratio", type=float, default=0.2, 
                       help="增量更新的数据比例 (0.1 = 10%)")
    parser.add_argument("-root", type=str, default="", help="结果目录前缀")
    parser.add_argument("-enable_query", type=str, default="1", help="是否运行查询评估")
    return parser.parse_args()


def check_dirs(opt, root, mode, opt_path):
    """创建结果目录"""
    base_dir = os.path.join(opt.working_dir, opt.exp_name, root) if root else os.path.join(opt.working_dir, opt.exp_name)
    
    # 根据模式创建不同的子目录
    mode_suffix = f"_{mode}" if mode != "build" else ""
    result_dir = os.path.join(base_dir, f"Results{mode_suffix}")
    config_dir = os.path.join(base_dir, f"Configs{mode_suffix}")
    metric_dir = os.path.join(base_dir, f"Metrics{mode_suffix}")
    
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    
    # 复制配置文件
    opt_name = opt_path[opt_path.rindex("/") + 1:]
    basic_name = os.path.join(opt_path.split("/")[0], "Config2.yaml")
    
    copyfile(opt_path, os.path.join(config_dir, opt_name))
    copyfile(basic_name, os.path.join(config_dir, "Config2.yaml"))
    
    return result_dir


def split_dataset_for_incremental(corpus, incremental_ratio=0.2):
    """
    将数据集分为初始构建集和增量更新集
    
    Args:
        corpus: 完整语料库
        incremental_ratio: 增量数据的比例
        
    Returns:
        (initial_corpus, incremental_corpus): 初始语料库和增量语料库
    """
    total_size = len(corpus)
    incremental_size = int(total_size * incremental_ratio)
    initial_size = total_size - incremental_size
    
    logger.info(f"数据集划分: 总计{total_size}, 初始{initial_size}, 增量{incremental_size}")
    
    # 简单按顺序分割，实际应用中可能需要更复杂的策略
    initial_corpus = corpus[:initial_size]
    incremental_corpus = corpus[initial_size:]
    
    return initial_corpus, incremental_corpus


async def build_initial_graph(digimon, initial_corpus):
    """构建初始图结构"""
    logger.info(f"🏗️ 开始构建初始图结构，包含{len(initial_corpus)}个文档")
    
    start_time = time.time()
    await digimon.insert(initial_corpus)
    build_time = time.time() - start_time
    
    # 获取图统计信息
    if hasattr(digimon.graph, 'get_incremental_statistics'):
        stats = digimon.graph.get_incremental_statistics()
        logger.info(f"📊 初始图构建统计: {stats}")
    
    logger.info(f"✅ 初始图构建完成，耗时: {build_time:.2f}秒")
    return build_time, stats if 'stats' in locals() else {}



async def insert_incremental_update(digimon, incremental_corpus):
    """执行语料插入更新"""
    logger.info(f"🔄 开始增量更新，添加{len(incremental_corpus)}个新文档")
    
    if not hasattr(digimon.graph, 'insert_incremental'):
        logger.error("❌ 当前图类型不支持增量更新")
        return None, {}
    
    start_time = time.time()
    
    try:
        # Step 1: 使用专门的增量更新方法处理chunk存储
        logger.info("📝 增量更新chunk存储（保护现有数据）...")
        
        # 使用新的update_chunks方法，只处理新文档，不影响现有chunk
        new_chunks = await digimon.doc_chunk.update_chunks(incremental_corpus)
        
        # Step 2: 获取新增的chunk数据并执行图增量更新
        if new_chunks:
            # 获取所有chunk数据，找出新增的chunk
            all_chunks = await digimon.doc_chunk.get_chunks()
            new_chunk_items = []
            
            # 根据new_chunks中的chunk_id找到对应的(key, TextChunk)对
            new_chunk_ids = {chunk["chunk_id"] for chunk in new_chunks}
            
            if all_chunks:
                for chunk_item in all_chunks:
                    if isinstance(chunk_item, tuple) and len(chunk_item) == 2:
                        chunk_key, chunk_obj = chunk_item
                        if chunk_key in new_chunk_ids:
                            new_chunk_items.append((chunk_key, chunk_obj))
            
            logger.info(f"🔧 执行图增量更新，处理{len(new_chunk_items)}个新chunk...")
            success = await digimon.graph.insert_incremental(new_chunk_items)
        else:
            logger.info("📝 没有新增chunk，跳过图更新")
            success = True
        #success = True
        update_time = time.time() - start_time
        
        if success:
            # 获取更新后的统计信息
            stats = digimon.graph.get_incremental_statistics()
            logger.info(f"📊 语料插入更新后统计: {stats}")
            logger.info(f"✅ 语料插入更新成功，耗时: {update_time:.2f}秒")
            return update_time, stats
        else:
            logger.error("❌ 语料插入更新失败")
            return None, {}

    except Exception as e:
        logger.error(f"❌ 语料插入更新过程中出错: {e}")
        return None, {}

async def benchmark_incremental_vs_full(digimon, initial_corpus, incremental_corpus):
    """
    对比增量更新和全量重构的性能
    """
    logger.info("🏁 开始性能基准测试")
    
    results = {
        'initial_build': {},
        'incremental_update': {},
        'full_rebuild': {},
        'comparison': {}
    }
    
    # 1. 构建初始图
    logger.info("Step 1: 构建初始图")
    initial_time, initial_stats = await build_initial_graph(digimon, initial_corpus)
    results['initial_build'] = {
        'time': initial_time,
        'stats': initial_stats
    }
    
    # 2. 保存初始状态（用于后续对比）
    if hasattr(digimon.graph._graph, 'save_metadata'):
        digimon.graph._graph.save_metadata({
            'stage': 'initial_build',
            'corpus_size': len(initial_corpus),
            'build_time': initial_time
        })
    
    # 3. 执行增量更新
    logger.info("Step 2: 执行增量更新")
    incremental_time, incremental_stats = await insert_incremental_update(digimon, incremental_corpus)
    if incremental_time is not None:
        results['incremental_update'] = {
            'time': incremental_time,
            'stats': incremental_stats
        }
    
    # 4. 重新构建图（全量）进行对比
    logger.info("Step 3: 全量重构用于对比")
    full_corpus = initial_corpus + incremental_corpus
    
    # 清理现有图
    if hasattr(digimon.graph, 'clear'):
        digimon.graph.clear()
    
    # 强制重构
    original_force = digimon.config.graph.force
    digimon.config.graph.force = True
    
    start_time = time.time()
    await digimon.insert(full_corpus)
    full_rebuild_time = time.time() - start_time
    
    # 恢复原始设置
    digimon.config.graph.force = original_force
    
    if hasattr(digimon.graph, 'get_incremental_statistics'):
        full_rebuild_stats = digimon.graph.get_incremental_statistics()
    else:
        full_rebuild_stats = {}
    
    results['full_rebuild'] = {
        'time': full_rebuild_time,
        'stats': full_rebuild_stats
    }
    
    # 5. 计算对比结果
    if incremental_time is not None:
        total_incremental_time = initial_time + incremental_time
        speedup = full_rebuild_time / total_incremental_time
        efficiency = (full_rebuild_time - total_incremental_time) / full_rebuild_time * 100
        
        results['comparison'] = {
            'total_incremental_time': total_incremental_time,
            'full_rebuild_time': full_rebuild_time,
            'speedup': speedup,
            'efficiency_improvement': efficiency,
            'time_saved': full_rebuild_time - total_incremental_time
        }
        
        logger.info(f"📈 性能对比结果:")
        logger.info(f"   增量更新总时间: {total_incremental_time:.2f}秒")
        logger.info(f"   全量重构时间: {full_rebuild_time:.2f}秒")
        logger.info(f"   性能提升: {speedup:.2f}x")
        logger.info(f"   效率提升: {efficiency:.1f}%")
        logger.info(f"   节省时间: {full_rebuild_time - total_incremental_time:.2f}秒")
    
    return results


async def wrapper_query(query_dataset, digimon, result_dir, mode=""):
    """执行查询测试"""
    all_res = []
    
    dataset_len = min(len(query_dataset), 3702)  # 限制测试数量
    
    logger.info(f"🔍 开始查询测试，模式: {mode}, 测试{dataset_len}个问题")
    
    for i in range(dataset_len):
        query = query_dataset[i]
        logger.info(f"正在处理问题 {i+1}/{dataset_len}...")
        
        try:
            res = await digimon.query(query["question"])
            query["output"] = res
            query["mode"] = mode  # 标记查询模式
            all_res.append(query)
        except Exception as e:
            logger.error(f"查询 {i+1} 失败: {e}")
            query["output"] = f"Error: {str(e)}"
            query["mode"] = mode
            all_res.append(query)
    
    # 保存结果
    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, f"results_{mode}.json" if mode else "results.json")
    all_res_df.to_json(save_path, orient="records", lines=True)
    
    logger.info(f"✅ 查询测试完成，结果保存到: {save_path}")
    return save_path


async def wrapper_evaluation(path, opt, result_dir, mode=""):
    """执行评估"""
    try:
        eval = Evaluator(path, opt.dataset_name)
        res_dict = await eval.evaluate()
        
        save_path = os.path.join(result_dir, f"metrics_{mode}.json" if mode else "metrics.json")
        with open(save_path, "w") as f:
            json.dump(res_dict, f, indent=2)
        
        logger.info(f"✅ 评估完成，结果保存到: {save_path}")
        return res_dict
    except Exception as e:
        logger.error(f"评估失败: {e}")
        return {}


async def main():
    """主函数"""
    args = parse_args()
    
    # 解析配置
    opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name)
    
    # 检查是否为增量更新配置
    if opt.graph.graph_type != "hk_graph_tree_dynamic":
        logger.error(f"错误: 配置文件的graph_type应为'hk_graph_tree_dynamic'，当前为'{opt.graph.graph_type}'")
        return
    
    # 创建目录
    result_dir = check_dirs(opt, args.root, args.mode, args.opt)
    
    # 创建GraphRAG实例
    digimon = GraphRAG(config=opt)
    
    # 加载数据集
    query_dataset = RAGQueryDataset(
        data_dir=os.path.join(opt.data_root, opt.dataset_name)
    )
    corpus = query_dataset.get_corpus()
    logger.info(f"加载数据集: {len(corpus)} 个文档")
    
    # 根据模式执行不同操作
    if args.mode == "build":
        # 模式1: 仅构建初始图
        logger.info("🏗️ 模式: 构建初始图")
        await build_initial_graph(digimon, corpus)
        
        if args.enable_query == "1":
            save_path = await wrapper_query(query_dataset, digimon, result_dir, "initial")
            await wrapper_evaluation(save_path, opt, result_dir, "initial")
    
    elif args.mode == "incremental":
        # 模式2: 增量更新测试
        logger.info("🔄 模式: 增量更新测试")
        
        # 分割数据集
        initial_corpus, incremental_corpus = split_dataset_for_incremental(
            corpus, args.incremental_ratio
        )
        
        # 构建初始图
        await build_initial_graph(digimon, initial_corpus)
        
        # 执行增量更新
        await insert_incremental_update(digimon, incremental_corpus)
        
        if args.enable_query == "1":
            save_path = await wrapper_query(query_dataset, digimon, result_dir, "incremental")
            await wrapper_evaluation(save_path, opt, result_dir, "incremental")
    
    elif args.mode == "benchmark":
        # 模式3: 性能基准测试
        logger.info("🏁 模式: 性能基准测试")
        
        # 分割数据集
        initial_corpus, incremental_corpus = split_dataset_for_incremental(
            corpus, args.incremental_ratio
        )
        
        # 执行基准测试
        benchmark_results = await benchmark_incremental_vs_full(
            digimon, initial_corpus, incremental_corpus
        )
        
        # 保存基准测试结果
        benchmark_path = os.path.join(result_dir, "benchmark_results.json")
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"📊 基准测试结果保存到: {benchmark_path}")
        
        if args.enable_query == "1":
            save_path = await wrapper_query(query_dataset, digimon, result_dir, "benchmark")
            await wrapper_evaluation(save_path, opt, result_dir, "benchmark")
    
    elif args.mode == "query":
        # 模式4: 仅查询测试（需要已有图）
        logger.info("🔍 模式: 查询测试")
        
        # 尝试加载现有图
        if hasattr(digimon.graph, '_load_graph'):
            loaded = await digimon.graph._load_graph(force=False)
            if not loaded:
                logger.error("❌ 未找到已构建的图，请先运行build模式")
                return
        
        # 关键修复：加载现有的chunk数据和构建查询器上下文
        logger.info("🔧 加载现有chunk数据和构建查询器上下文...")
        try:
            # 加载现有的chunk数据（不重新构建）
            chunk_loaded = await digimon.doc_chunk._load_chunk(force=False)
            if not chunk_loaded:
                logger.error("❌ 未找到已有的chunk数据，请先运行完整的增量更新")
                return
            logger.info("✅ 成功加载现有chunk数据")
            
            # 如果需要实体链接映射，加载现有的映射数据
            if digimon.config.use_entity_link_chunk:
                await digimon.build_e2r_r2c_maps(force=False)
                logger.info("✅ 成功加载实体链接映射数据")
            
            # 构建查询器上下文（关键步骤）
            await digimon._build_retriever_context()
            logger.info("✅ 查询器上下文构建完成")
            
        except Exception as e:
            logger.error(f"❌ 构建查询器上下文失败: {e}")
            return
        
        if args.enable_query == "1":
            save_path = await wrapper_query(query_dataset, digimon, result_dir, "query_only")
            await wrapper_evaluation(save_path, opt, result_dir, "query_only")
    
    logger.info("✅ 程序执行完成")


if __name__ == "__main__":
    asyncio.run(main())
