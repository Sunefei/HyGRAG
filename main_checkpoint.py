import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
warnings.filterwarnings("ignore") # Ignore warnings
import nltk
nltk.data.path.append('/data/zhy/nltk_data') # Local cache for nltk database

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
    """Save checkpoint to file"""
    checkpoint_data = {
        'current_index': current_index,
        'completed_results': all_results,
        'total_length': dataset_len,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    checkpoint_path = os.path.join(result_dir, "checkpoint.pkl")
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # Also save a readable JSON version for debugging
    checkpoint_json_path = os.path.join(result_dir, "checkpoint.json")
    with open(checkpoint_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'current_index': current_index,
            'total_length': dataset_len,
            'completed_count': len(all_results),
            'timestamp': checkpoint_data['timestamp']
        }, f, indent=2, ensure_ascii=False)
    
    from Core.Common.Logger import logger
    logger.info(f"Checkpoint saved: progress {current_index + 1}/{dataset_len}, completed {len(all_results)} queries")


def load_checkpoint(result_dir: str):
    """Load checkpoint from file"""
    checkpoint_path = os.path.join(result_dir, "checkpoint.pkl")
    
    if not os.path.exists(checkpoint_path):
        return None, []
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        from Core.Common.Logger import logger
        logger.info(f"Found checkpoint: last progress was question {checkpoint_data['current_index'] + 1}/{checkpoint_data['total_length']}, "
                   f"completed {len(checkpoint_data['completed_results'])} queries")
        logger.info(f"Checkpoint时间: {checkpoint_data['timestamp']}")
        
        return checkpoint_data['current_index'], checkpoint_data['completed_results']
    except Exception as e:
        from Core.Common.Logger import logger
        logger.warning(f"Failed to load checkpoint: {e}, will start from beginning")
        return None, []


def clear_checkpoint(result_dir: str):
    """Clear checkpoint files"""
    checkpoint_path = os.path.join(result_dir, "checkpoint.pkl")
    checkpoint_json_path = os.path.join(result_dir, "checkpoint.json")
    
    for path in [checkpoint_path, checkpoint_json_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                from Core.Common.Logger import logger
                logger.warning(f"Failed to delete file {path}: {e}")


def wrapper_query_with_checkpoint(query_dataset, digimon, result_dir):
    """Query processing function with checkpoint support"""
    from Core.Common.Logger import logger
    
    dataset_len = len(query_dataset)
    # dataset_len = 3702 #TODO Reduce test set length
    
    # Try to load checkpoint
    start_index, all_res = load_checkpoint(result_dir)
    
    if start_index is not None:
        logger.info(f"Resuming from checkpoint, will start processing from question {start_index + 2}")
        start_index += 1  # Index of the next item to process
    else:
        logger.info("No checkpoint found, starting from beginning")
        start_index = 0
        all_res = []
    
    # Record the start time of the query phase
    query_phase_start_time = time.time()
    logger.info(f"Starting query phase, expected to process {dataset_len - start_index} questions")
    
    # Process remaining queries
    for i in range(start_index, dataset_len):
        query = query_dataset[i]
        logger.info(f"Processing question {i+1}/{dataset_len}...")
        
        try:
            res = asyncio.run(digimon.query(query["question"]))
            query["output"] = res
            all_res.append(query)
            
            logger.info(f"Completed question {i+1}/{dataset_len}")
            
            # Save checkpoint after processing each question
            save_checkpoint(result_dir, i, all_res, dataset_len)
            
        except Exception as e:
            logger.error(f"Error processing question {i+1}: {e}")
            # Even if there's an error, save checkpoint, but don't include the current failed query
            save_checkpoint(result_dir, i-1 if i > start_index else start_index-1, all_res, dataset_len)
            raise e
    
    # Calculate and record query phase time statistics
    query_phase_total_time = time.time() - query_phase_start_time
    processed_queries = dataset_len - start_index
    
    if processed_queries > 0:
        avg_query_time = query_phase_total_time / processed_queries
        
        logger.info("=" * 60)
        logger.info("Query phase time statistics summary:")
        logger.info(f"  Total processing time: {query_phase_total_time:.2f} seconds ({query_phase_total_time/60:.2f} minutes)")
        logger.info(f"  Successfully processed queries: {processed_queries}")
        logger.info(f"  Average time per query: {avg_query_time:.2f} seconds")
        logger.info("=" * 60)
    else:
        logger.info(f"Query phase total time: {query_phase_total_time:.2f} seconds, but no queries were successfully processed")
    
    # After completing all processing, save the final results and clear checkpoint
    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, "results.json")
    all_res_df.to_json(save_path, orient="records", lines=True)
    
    # Clear checkpoint files
    clear_checkpoint(result_dir)
    logger.info(f"All queries processed, results saved to: {save_path}")
    
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
    
    # Show checkpoint status
    if args.show_checkpoint:
        checkpoint_path = os.path.join(result_dir, "checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            start_index, all_res = load_checkpoint(result_dir)
            print(f"Found checkpoint: last progress was question {start_index + 1}, completed {len(all_res)} queries")
        else:
            print("No checkpoint file found")
        exit(0)
    
    # Clear checkpoint
    if args.clear_checkpoint:
        clear_checkpoint(result_dir)
        from Core.Common.Logger import logger
        logger.info("Checkpoint cleared, will start processing from beginning")

    digimon = GraphRAG(config=opt)

    query_dataset = RAGQueryDataset(
        data_dir=os.path.join(opt.data_root, opt.dataset_name)
    )
    corpus = query_dataset.get_corpus()

    asyncio.run(digimon.insert(corpus))

    # Use query processing function with checkpoint support
    save_path = wrapper_query_with_checkpoint(query_dataset, digimon, result_dir)

    if save_path and os.path.exists(save_path):
        asyncio.run(wrapper_evaluation(save_path, opt, result_dir))