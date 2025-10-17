import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
warnings.filterwarnings("ignore")
import nltk
nltk.data.path.append('/data/zhy/nltk_data')

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
    """Zero-shot查询类，直接使用LLM进行推理，不使用RAG数据库"""
    
    def __init__(self, config):
        super().__init__(config=config)
        
    async def query(self, question: str) -> str:
        """
        直接使用LLM进行Zero-shot推理
        
        Args:
            question: 问题文本
            
        Returns:
            str: LLM的回答
        """
        try:
            # 直接使用LLM进行推理，不提供任何上下文
            #response = await self.llm.aask(question)
            # user_message = question
            system_prompt = """
            You are a helpful QA assistant.
            """
            # response = await self.llm.aask(
            #         question, system_msgs=[system_prompt] if system_prompt else None
                    
            #     )
            # 修改后的 CoT 版本
            cot_prompt = """
            Let's think step by step.
            """

            # 整合到您的调用逻辑中
            # 假设 question 变量包含了您想要模型回答的问题
            # system_prompt 保持不变
            response = await self.llm.aask(
                f"{question}\n{cot_prompt}", # 将 CoT prompt 添加到原始问题后面
                system_msgs=[system_prompt] if system_prompt else None
            )
            return response
        except Exception as e:
            logger.error(f"Zero-shot查询失败: {e}")
            return f"查询失败: {str(e)}"


class ZeroShotGraphRAG(ContextMixin):
    """Zero-shot版本的GraphRAG，跳过所有检索和索引构建步骤"""
    
    def __init__(self, config):
        super().__init__(config=config)
        self.zero_shot_query = ZeroShotQuery(config)
        
    async def insert(self, docs):
        """
        跳过文档插入和索引构建步骤
        """
        logger.info("Zero-shot模式：跳过文档插入和索引构建")
        return
        
    async def query(self, query):
        """
        直接进行Zero-shot查询
        """
        return await self.zero_shot_query.query(query)


def check_dirs(opt, opt_file_path, exp_name="ZeroShot_experiment"):
    """创建结果目录"""
    result_dir = os.path.join(opt.working_dir, exp_name, "Results")
    config_dir = os.path.join(opt.working_dir, exp_name, "Configs")
    metric_dir = os.path.join(opt.working_dir, exp_name, "Metrics")
    
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    
    # 保存配置文件
    try:
        opt_name = opt_file_path[opt_file_path.rindex("/") + 1:]
        basic_name = os.path.join(opt_file_path.split("/")[0], "Config2.yaml")
        copyfile(opt_file_path, os.path.join(config_dir, opt_name))
        copyfile(basic_name, os.path.join(config_dir, "Config2.yaml"))
    except Exception as e:
        logger.warning(f"保存配置文件时出错: {e}")
    
    return result_dir


async def wrapper_query(query_dataset, zero_shot_rag, result_dir):
    """批量处理查询"""
    all_res = []
    dataset_len = len(query_dataset)
    print(dataset_len)
    
    logger.info(f"开始Zero-shot推理，共{dataset_len}个问题")
    
    for i in range(dataset_len):
        query = query_dataset[i]
        logger.info(f"正在处理问题 {i+1}/{dataset_len}...")
        
        # 进行Zero-shot查询
        res = await zero_shot_rag.query(query["question"])
        query["output"] = res
        all_res.append(query)
        
        # 每处理10个问题输出一次进度
        if (i + 1) % 10 == 0:
            logger.info(f"已处理 {i+1}/{dataset_len} 个问题")

    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, "zero_shot_results.json")
    all_res_df.to_json(save_path, orient="records", lines=True)
    logger.info(f"结果已保存到: {save_path}")
    return save_path


async def wrapper_evaluation(path, opt, result_dir):
    """评估结果"""
    logger.info("开始评估Zero-shot结果...")
    eval = Evaluator(path, opt.dataset_name)
    res_dict = await eval.evaluate()
    save_path = os.path.join(result_dir, "zero_shot_metrics.json")
    with open(save_path, "w") as f:
        f.write(str(res_dict))
    logger.info(f"评估结果已保存到: {save_path}")
    return res_dict


async def main():
    parser = argparse.ArgumentParser(description="Zero-shot推理脚本")
    parser.add_argument("-opt", type=str, help="配置文件路径")
    parser.add_argument("-dataset_name", type=str, help="数据集名称")
    parser.add_argument("--max_queries", type=int, default=None, help="最大查询数量（用于测试）")
    args = parser.parse_args()

    # 解析配置
    opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name)
    
    # 创建Zero-shot GraphRAG实例
    zero_shot_rag = ZeroShotGraphRAG(config=opt)
    
    # 创建结果目录
    result_dir = check_dirs(opt, args.opt)
    
    # 加载查询数据集
    query_dataset = RAGQueryDataset(
        data_dir=os.path.join(opt.data_root, opt.dataset_name)
    )
    
    # 限制查询数量（如果指定）
    if args.max_queries:
        query_dataset = query_dataset[:args.max_queries]
        logger.info(f"限制查询数量为: {args.max_queries}")
    
    # 跳过文档插入（Zero-shot模式）
    logger.info("Zero-shot模式：跳过文档插入和索引构建")
    
    # 执行查询
    save_path = await wrapper_query(query_dataset, zero_shot_rag, result_dir)
    
    # 评估结果
    metrics = await wrapper_evaluation(save_path, opt, result_dir)
    
    # 输出主要指标
    logger.info("=" * 50)
    logger.info("Zero-shot推理完成！")
    logger.info("=" * 50)
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())