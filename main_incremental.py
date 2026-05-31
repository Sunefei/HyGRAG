"""
HyGRAG incremental update evaluation script.

Usage:
  1. Build:       python main_incremental.py -opt Option/Ours/HKGraphTreeDynamic.yaml -dataset_name multihop-rag -mode build
  2. Incremental: python main_incremental.py -opt Option/Ours/HKGraphTreeDynamic.yaml -dataset_name multihop-rag -mode incremental
  3. Benchmark:   python main_incremental.py -opt Option/Ours/HKGraphTreeDynamic.yaml -dataset_name multihop-rag -mode benchmark
  4. Query only:  python main_incremental.py -opt Option/Ours/HKGraphTreeDynamic.yaml -dataset_name multihop-rag -mode query
"""

import os
import warnings
warnings.filterwarnings("ignore")

import nltk
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
    parser = argparse.ArgumentParser(description="HyGRAG incremental update evaluation")
    parser.add_argument("-opt", type=str, required=True, help="Configuration file path")
    parser.add_argument("-dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("-mode", type=str, choices=["build", "incremental", "benchmark", "query"],
                       default="build", help="Run mode")
    parser.add_argument("-incremental_ratio", type=float, default=0.2,
                       help="Incremental update data ratio (0.0-1.0)")
    parser.add_argument("-root", type=str, default="", help="Result directory prefix")
    parser.add_argument("-enable_query", type=str, default="1", help="Whether to run query evaluation")
    return parser.parse_args()


def check_dirs(opt, root, mode, opt_path):
    base_dir = os.path.join(opt.working_dir, opt.exp_name, root) if root else os.path.join(opt.working_dir, opt.exp_name)

    mode_suffix = f"_{mode}" if mode != "build" else ""
    result_dir = os.path.join(base_dir, f"Results{mode_suffix}")
    config_dir = os.path.join(base_dir, f"Configs{mode_suffix}")
    metric_dir = os.path.join(base_dir, f"Metrics{mode_suffix}")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    opt_name = opt_path[opt_path.rindex("/") + 1:]
    basic_name = os.path.join(opt_path.split("/")[0], "Config2.yaml")

    copyfile(opt_path, os.path.join(config_dir, opt_name))
    copyfile(basic_name, os.path.join(config_dir, "Config2.yaml"))

    return result_dir


def split_dataset_for_incremental(corpus, incremental_ratio=0.2):
    total_size = len(corpus)
    incremental_size = int(total_size * incremental_ratio)
    initial_size = total_size - incremental_size

    logger.info(f"Dataset split: Total {total_size}, Initial {initial_size}, Incremental {incremental_size}")

    initial_corpus = corpus[:initial_size]
    incremental_corpus = corpus[initial_size:]

    return initial_corpus, incremental_corpus


async def build_initial_graph(digimon, initial_corpus):
    logger.info(f"Building initial graph with {len(initial_corpus)} documents")

    start_time = time.time()
    await digimon.insert(initial_corpus)
    build_time = time.time() - start_time

    stats = {}
    if hasattr(digimon.graph, 'get_incremental_statistics'):
        stats = digimon.graph.get_incremental_statistics()
        logger.info(f"Initial graph statistics: {stats}")

    logger.info(f"Initial graph built, time: {build_time:.2f}s")
    return build_time, stats


async def insert_incremental_update(digimon, incremental_corpus):
    logger.info(f"Starting incremental update, adding {len(incremental_corpus)} new documents")

    if not hasattr(digimon.graph, 'insert_incremental'):
        logger.error("Current graph type does not support incremental update")
        return None, {}

    start_time = time.time()

    try:
        logger.info("Processing incremental chunk storage...")
        new_chunks = await digimon.doc_chunk.update_chunks(incremental_corpus)

        if new_chunks:
            all_chunks = await digimon.doc_chunk.get_chunks()
            new_chunk_items = []
            new_chunk_ids = {chunk["chunk_id"] for chunk in new_chunks}

            if all_chunks:
                for chunk_item in all_chunks:
                    if isinstance(chunk_item, tuple) and len(chunk_item) == 2:
                        chunk_key, chunk_obj = chunk_item
                        if chunk_key in new_chunk_ids:
                            new_chunk_items.append((chunk_key, chunk_obj))

            logger.info(f"Graph incremental update: processing {len(new_chunk_items)} new chunks...")
            success = await digimon.graph.insert_incremental(new_chunk_items)
        else:
            logger.info("No new chunks, skipping graph update")
            success = True

        update_time = time.time() - start_time

        if success:
            stats = digimon.graph.get_incremental_statistics()
            logger.info(f"Statistics after incremental update: {stats}")
            logger.info(f"Incremental update complete, time: {update_time:.2f}s")
            return update_time, stats
        else:
            logger.error("Incremental update failed")
            return None, {}

    except Exception as e:
        logger.error(f"Error during incremental update: {e}")
        return None, {}


async def benchmark_incremental_vs_full(digimon, initial_corpus, incremental_corpus):
    logger.info("Starting performance benchmark")

    results = {
        'initial_build': {},
        'incremental_update': {},
        'full_rebuild': {},
        'comparison': {}
    }

    logger.info("Step 1: Build initial graph")
    initial_time, initial_stats = await build_initial_graph(digimon, initial_corpus)
    results['initial_build'] = {
        'time': initial_time,
        'stats': initial_stats
    }

    logger.info("Step 2: Execute incremental update")
    incremental_time, incremental_stats = await insert_incremental_update(digimon, incremental_corpus)
    if incremental_time is not None:
        results['incremental_update'] = {
            'time': incremental_time,
            'stats': incremental_stats
        }

    logger.info("Step 3: Full rebuild for comparison")
    full_corpus = initial_corpus + incremental_corpus

    if hasattr(digimon.graph, 'clear'):
        digimon.graph.clear()

    original_force = digimon.config.graph.force
    digimon.config.graph.force = True

    start_time = time.time()
    await digimon.insert(full_corpus)
    full_rebuild_time = time.time() - start_time

    digimon.config.graph.force = original_force

    full_rebuild_stats = {}
    if hasattr(digimon.graph, 'get_incremental_statistics'):
        full_rebuild_stats = digimon.graph.get_incremental_statistics()

    results['full_rebuild'] = {
        'time': full_rebuild_time,
        'stats': full_rebuild_stats
    }

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

        logger.info(f"Performance comparison results:")
        logger.info(f"   Total incremental time: {total_incremental_time:.2f}s")
        logger.info(f"   Full rebuild time: {full_rebuild_time:.2f}s")
        logger.info(f"   Speedup: {speedup:.2f}x")
        logger.info(f"   Efficiency improvement: {efficiency:.1f}%")
        logger.info(f"   Time saved: {full_rebuild_time - total_incremental_time:.2f}s")

    return results


async def wrapper_query(query_dataset, digimon, result_dir, mode=""):
    all_res = []
    dataset_len = min(len(query_dataset), 3702)

    logger.info(f"Starting query test, mode: {mode}, {dataset_len} questions")

    for i in range(dataset_len):
        query = query_dataset[i]
        logger.info(f"Processing question {i+1}/{dataset_len}...")

        try:
            res = await digimon.query(query["question"])
            query["output"] = res
            query["mode"] = mode
            all_res.append(query)
        except Exception as e:
            logger.error(f"Query {i+1} failed: {e}")
            query["output"] = f"Error: {str(e)}"
            query["mode"] = mode
            all_res.append(query)

    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, f"results_{mode}.json" if mode else "results.json")
    all_res_df.to_json(save_path, orient="records", lines=True)

    logger.info(f"Query test complete, results saved to: {save_path}")
    return save_path


async def wrapper_evaluation(path, opt, result_dir, mode=""):
    try:
        eval = Evaluator(path, opt.dataset_name)
        res_dict = await eval.evaluate()

        save_path = os.path.join(result_dir, f"metrics_{mode}.json" if mode else "metrics.json")
        with open(save_path, "w") as f:
            json.dump(res_dict, f, indent=2)

        logger.info(f"Evaluation complete, results saved to: {save_path}")
        return res_dict
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {}


async def main():
    args = parse_args()

    opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name)

    if opt.graph.graph_type != "hk_graph_tree_dynamic":
        logger.error(f"Error: graph_type should be 'hk_graph_tree_dynamic', got '{opt.graph.graph_type}'")
        return

    result_dir = check_dirs(opt, args.root, args.mode, args.opt)

    digimon = GraphRAG(config=opt)

    query_dataset = RAGQueryDataset(
        data_dir=os.path.join(opt.data_root, opt.dataset_name)
    )
    corpus = query_dataset.get_corpus()
    logger.info(f"Loaded dataset: {len(corpus)} documents")

    if args.mode == "build":
        logger.info("Mode: Build initial graph")
        await build_initial_graph(digimon, corpus)

        if args.enable_query == "1":
            save_path = await wrapper_query(query_dataset, digimon, result_dir, "initial")
            await wrapper_evaluation(save_path, opt, result_dir, "initial")

    elif args.mode == "incremental":
        logger.info("Mode: Incremental update")

        initial_corpus, incremental_corpus = split_dataset_for_incremental(
            corpus, args.incremental_ratio
        )

        await build_initial_graph(digimon, initial_corpus)
        await insert_incremental_update(digimon, incremental_corpus)

        if args.enable_query == "1":
            save_path = await wrapper_query(query_dataset, digimon, result_dir, "incremental")
            await wrapper_evaluation(save_path, opt, result_dir, "incremental")

    elif args.mode == "benchmark":
        logger.info("Mode: Performance benchmark")

        initial_corpus, incremental_corpus = split_dataset_for_incremental(
            corpus, args.incremental_ratio
        )

        benchmark_results = await benchmark_incremental_vs_full(
            digimon, initial_corpus, incremental_corpus
        )

        benchmark_path = os.path.join(result_dir, "benchmark_results.json")
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)

        logger.info(f"Benchmark results saved to: {benchmark_path}")

        if args.enable_query == "1":
            save_path = await wrapper_query(query_dataset, digimon, result_dir, "benchmark")
            await wrapper_evaluation(save_path, opt, result_dir, "benchmark")

    elif args.mode == "query":
        logger.info("Mode: Query only")

        if hasattr(digimon.graph, '_load_graph'):
            loaded = await digimon.graph._load_graph(force=False)
            if not loaded:
                logger.error("No existing graph found, run build mode first")
                return

        logger.info("Loading existing chunk data and building retriever context...")
        try:
            chunk_loaded = await digimon.doc_chunk._load_chunk(force=False)
            if not chunk_loaded:
                logger.error("No existing chunk data found, run incremental update first")
                return
            logger.info("Successfully loaded existing chunk data")

            if digimon.config.use_entity_link_chunk:
                await digimon.build_e2r_r2c_maps(force=False)
                logger.info("Successfully loaded entity link mapping data")

            await digimon._build_retriever_context()
            logger.info("Retriever context built successfully")

        except Exception as e:
            logger.error(f"Failed to build retriever context: {e}")
            return

        if args.enable_query == "1":
            save_path = await wrapper_query(query_dataset, digimon, result_dir, "query_only")
            await wrapper_evaluation(save_path, opt, result_dir, "query_only")

    logger.info("Program execution completed")


if __name__ == "__main__":
    asyncio.run(main())
