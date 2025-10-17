import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
warnings.filterwarnings("ignore") # 忽略warning
import nltk
nltk.data.path.append('/data/zhy/nltk_data') #本地缓存nltk数据库

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
import argparse
import os
import asyncio
from pathlib import Path
from shutil import copyfile
from Data.QueryDataset import RAGQueryDataset
import pandas as pd
from Core.Utils.Evaluation import Evaluator
import json
import pickle
import time
from typing import List, Dict, Any


def check_dirs(opt):
    # For each query, save the results in a separate directory
    result_dir = os.path.join(opt.working_dir, opt.exp_name, "Results")
    # Save the current used config in a separate directory
    config_dir = os.path.join(opt.working_dir, opt.exp_name, "Configs")
    # Save the metrics of entire experiment in a separate directory
    metric_dir = os.path.join(opt.working_dir, opt.exp_name, "Metrics")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    opt_name = args.opt[args.opt.rindex("/") + 1 :]
    basic_name = os.path.join(args.opt.split("/")[0], "Config2.yaml")
    copyfile(args.opt, os.path.join(config_dir, opt_name))
    copyfile(basic_name, os.path.join(config_dir, "Config2.yaml"))
    return result_dir


def save_checkpoint(result_dir: str, current_index: int, all_results: List[Dict], dataset_len:int):
    """保存checkpoint到文件"""
    checkpoint_data = {
        'current_index': current_index,
        'completed_results': all_results,
        'total_length': dataset_len,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    checkpoint_path = os.path.join(result_dir, "checkpoint.pkl")
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # 同时保存一个可读的JSON版本用于调试
    checkpoint_json_path = os.path.join(result_dir, "checkpoint.json")
    with open(checkpoint_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'current_index': current_index,
            'total_length': dataset_len,
            'completed_count': len(all_results),
            'timestamp': checkpoint_data['timestamp']
        }, f, indent=2, ensure_ascii=False)
    
    from Core.Common.Logger import logger
    logger.info(f"Checkpoint已保存: 进度 {current_index + 1}/{dataset_len}, 已完成 {len(all_results)} 个查询")


def load_checkpoint(result_dir: str):
    """从文件加载checkpoint"""
    checkpoint_path = os.path.join(result_dir, "checkpoint.pkl")
    
    if not os.path.exists(checkpoint_path):
        return None, []
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        from Core.Common.Logger import logger
        logger.info(f"发现checkpoint: 上次进度 {checkpoint_data['current_index'] + 1}/{checkpoint_data['total_length']}, "
                   f"已完成 {len(checkpoint_data['completed_results'])} 个查询")
        logger.info(f"Checkpoint时间: {checkpoint_data['timestamp']}")
        
        return checkpoint_data['current_index'], checkpoint_data['completed_results']
    except Exception as e:
        from Core.Common.Logger import logger
        logger.warning(f"加载checkpoint失败: {e}, 将从头开始")
        return None, []


def clear_checkpoint(result_dir: str):
    """清除checkpoint文件"""
    checkpoint_path = os.path.join(result_dir, "checkpoint.pkl")
    checkpoint_json_path = os.path.join(result_dir, "checkpoint.json")
    
    for path in [checkpoint_path, checkpoint_json_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                from Core.Common.Logger import logger
                logger.warning(f"删除文件失败 {path}: {e}")


def wrapper_query_with_checkpoint(query_dataset, digimon, result_dir):
    """支持checkpoint的查询处理函数"""
    from Core.Common.Logger import logger
    
    dataset_len = len(query_dataset)
    #dataset_len = 3702 #TODO 减少测试集长度
    
    # 尝试加载checkpoint
    start_index, all_res = load_checkpoint(result_dir)
    
    if start_index is not None:
        logger.info(f"从checkpoint恢复，将从第 {start_index + 2} 个问题开始处理")
        start_index += 1  # 下一个要处理的索引
    else:
        logger.info("未找到checkpoint，从头开始处理")
        start_index = 0
        all_res = []
    
    # 记录查询阶段开始时间
    query_phase_start_time = time.time()
    logger.info(f"开始处理查询阶段，预计处理 {dataset_len - start_index} 个问题")
    
    # 处理剩余的查询
    for i in range(start_index, dataset_len):
        query = query_dataset[i]
        logger.info(f"正在处理问题 {i+1}/{dataset_len}...")
        
        try:
            res = asyncio.run(digimon.query(query["question"]))
            query["output"] = res
            all_res.append(query)
            
            logger.info(f"完成问题 {i+1}/{dataset_len}")
            
            # 每处理一个问题就保存一次checkpoint
            save_checkpoint(result_dir, i, all_res, dataset_len)
            
        except Exception as e:
            logger.error(f"处理问题 {i+1} 时出错: {e}")
            # 即使出错也保存checkpoint，但不包含当前失败的查询
            save_checkpoint(result_dir, i-1 if i > start_index else start_index-1, all_res, dataset_len)
            raise e
    
    # 计算并记录查询阶段时间统计
    query_phase_total_time = time.time() - query_phase_start_time
    processed_queries = dataset_len - start_index
    
    if processed_queries > 0:
        avg_query_time = query_phase_total_time / processed_queries
        
        logger.info("=" * 60)
        logger.info("查询阶段时间统计汇总:")
        logger.info(f"  总处理时间: {query_phase_total_time:.2f}秒 ({query_phase_total_time/60:.2f}分钟)")
        logger.info(f"  成功处理查询数: {processed_queries}")
        logger.info(f"  平均每个查询耗时: {avg_query_time:.2f}秒")
        logger.info("=" * 60)
    else:
        logger.info(f"查询阶段总耗时: {query_phase_total_time:.2f}秒，但没有成功处理任何查询")
    
    # 完成所有处理后，保存最终结果并清除checkpoint
    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, "results.json")
    all_res_df.to_json(save_path, orient="records", lines=True)
    
    # 清除checkpoint文件
    clear_checkpoint(result_dir)
    logger.info(f"所有查询处理完成，结果已保存到: {save_path}")
    
    return save_path


async def wrapper_evaluation(path, opt, result_dir):
    eval = Evaluator(path, opt.dataset_name)
    res_dict = await eval.evaluate()
    save_path = os.path.join(result_dir, "metrics.json")
    with open(save_path, "w") as f:
        f.write(str(res_dict))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
    parser.add_argument("--clear_checkpoint", action="store_true", help="Clear existing checkpoint and start from beginning")
    parser.add_argument("--show_checkpoint", action="store_true", help="Show checkpoint status and exit")
    args = parser.parse_args()

    opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name)
    result_dir = check_dirs(opt)
    
    # 显示checkpoint状态
    if args.show_checkpoint:
        checkpoint_path = os.path.join(result_dir, "checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            start_index, all_res = load_checkpoint(result_dir)
            print(f"发现checkpoint: 上次进度为第 {start_index + 1} 个问题，已完成 {len(all_res)} 个查询")
        else:
            print("未发现checkpoint文件")
        exit(0)
    
    # 清除checkpoint
    if args.clear_checkpoint:
        clear_checkpoint(result_dir)
        from Core.Common.Logger import logger
        logger.info("Checkpoint已清除，将从头开始处理")

    digimon = GraphRAG(config=opt)

    query_dataset = RAGQueryDataset(
        data_dir=os.path.join(opt.data_root, opt.dataset_name)
    )
    corpus = query_dataset.get_corpus()

    asyncio.run(digimon.insert(corpus))

    # 使用支持checkpoint的查询处理函数
    save_path = wrapper_query_with_checkpoint(query_dataset, digimon, result_dir)

    if save_path and os.path.exists(save_path):
        asyncio.run(wrapper_evaluation(save_path, opt, result_dir))