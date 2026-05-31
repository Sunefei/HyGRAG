import os
import warnings
warnings.filterwarnings("ignore")

import nltk
import asyncio
import argparse
import pandas as pd
from pathlib import Path
from shutil import copyfile
from Data.QueryDataset import RAGQueryDataset
from Core.Utils.Evaluation import Evaluator
from Core.Common.Logger import logger
from Core.Common.ContextMixin import ContextMixin
from Option.Config2 import Config


class ZeroShotQuery(ContextMixin):
    """Zero-shot query class: uses LLM directly without RAG retrieval."""

    def __init__(self, config):
        super().__init__(config=config)

    async def query(self, question: str) -> str:
        try:
            system_prompt = "You are a helpful QA assistant."
            cot_prompt = "Let's think step by step."
            response = await self.llm.aask(
                f"{question}\n{cot_prompt}",
                system_msgs=[system_prompt] if system_prompt else None
            )
            return response
        except Exception as e:
            logger.error(f"Zero-shot query failed: {e}")
            return f"Query failed: {str(e)}"


class ZeroShotGraphRAG(ContextMixin):
    """Zero-shot GraphRAG: skips all retrieval and index construction steps."""

    def __init__(self, config):
        super().__init__(config=config)
        self.zero_shot_query = ZeroShotQuery(config)

    async def insert(self, docs):
        logger.info("Zero-shot mode: skipping document insertion and index construction")
        return

    async def query(self, query):
        return await self.zero_shot_query.query(query)


def check_dirs(opt, opt_file_path, exp_name="ZeroShot_experiment"):
    result_dir = os.path.join(opt.working_dir, exp_name, "Results")
    config_dir = os.path.join(opt.working_dir, exp_name, "Configs")
    metric_dir = os.path.join(opt.working_dir, exp_name, "Metrics")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    try:
        opt_name = opt_file_path[opt_file_path.rindex("/") + 1:]
        basic_name = os.path.join(opt_file_path.split("/")[0], "Config2.yaml")
        copyfile(opt_file_path, os.path.join(config_dir, opt_name))
        copyfile(basic_name, os.path.join(config_dir, "Config2.yaml"))
    except Exception as e:
        logger.warning(f"Error saving config: {e}")

    return result_dir


async def wrapper_query(query_dataset, zero_shot_rag, result_dir):
    all_res = []
    dataset_len = len(query_dataset)

    logger.info(f"Starting zero-shot inference, {dataset_len} questions")

    for i in range(dataset_len):
        query = query_dataset[i]
        logger.info(f"Processing question {i+1}/{dataset_len}...")

        res = await zero_shot_rag.query(query["question"])
        query["output"] = res
        all_res.append(query)

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i+1}/{dataset_len} questions")

    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, "zero_shot_results.json")
    all_res_df.to_json(save_path, orient="records", lines=True)
    logger.info(f"Results saved to: {save_path}")
    return save_path


async def wrapper_evaluation(path, opt, result_dir):
    logger.info("Starting zero-shot evaluation...")
    eval = Evaluator(path, opt.dataset_name)
    res_dict = await eval.evaluate()
    save_path = os.path.join(result_dir, "zero_shot_metrics.json")
    with open(save_path, "w") as f:
        f.write(str(res_dict))
    logger.info(f"Evaluation results saved to: {save_path}")
    return res_dict


async def main():
    parser = argparse.ArgumentParser(description="Zero-shot inference script")
    parser.add_argument("-opt", type=str, help="Configuration file path")
    parser.add_argument("-dataset_name", type=str, help="Dataset name")
    parser.add_argument("--max_queries", type=int, default=None, help="Max number of queries (for testing)")
    args = parser.parse_args()

    opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name)
    zero_shot_rag = ZeroShotGraphRAG(config=opt)

    result_dir = check_dirs(opt, args.opt)

    query_dataset = RAGQueryDataset(
        data_dir=os.path.join(opt.data_root, opt.dataset_name)
    )

    if args.max_queries:
        query_dataset = query_dataset[:args.max_queries]
        logger.info(f"Limiting queries to: {args.max_queries}")

    logger.info("Zero-shot mode: skipping document insertion and index construction")
    save_path = await wrapper_query(query_dataset, zero_shot_rag, result_dir)

    metrics = await wrapper_evaluation(save_path, opt, result_dir)

    logger.info("=" * 50)
    logger.info("Zero-shot inference completed!")
    logger.info("=" * 50)
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
