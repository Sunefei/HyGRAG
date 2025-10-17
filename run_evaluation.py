#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG结果评估脚本
用于对RAG过程中已经生成的结果进行评估

使用方法:
python run_evaluation.py --result_path /path/to/results.json --dataset_name Popqa --output_dir /path/to/output

作者: AI Assistant
日期: 2024
"""

import os
import sys
import json
import pandas as pd
import re
import string
import argparse
import asyncio
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 尝试导入tqdm进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("提示: 安装tqdm可以显示进度条: pip install tqdm")
    TQDM_AVAILABLE = False

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir  # 当前目录就是GraphRAG项目根目录
sys.path.append(project_root)

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score
    import nltk
    from nltk import sent_tokenize
    from rouge_score import rouge_scorer, scoring
except ImportError as e:
    print(f"缺少必要的依赖包: {e}")
    print("请安装: pip install nltk rouge-score")
    sys.exit(1)

# 尝试导入原项目的LLM模块
try:
    from Option.Config2 import default_config
    from Core.Provider.BaseLLM import BaseLLM
    from Core.Provider.LLMProviderRegister import create_llm_instance
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入LLM模块: {e}")
    print("Close-set评估将使用正则表达式fallback模式")
    LLM_AVAILABLE = False

# 设置NLTK数据路径
nltk_path = "/data/zhy/nltk_data"  # 根据您的环境调整
nltk.data.path.append(nltk_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("下载NLTK数据...")
    nltk.download("punkt", download_dir=nltk_path)
    nltk.download("wordnet", download_dir=nltk_path)


class RAGEvaluator:
    """RAG结果评估器"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        
        # 尝试初始化LLM（如果模块可用）
        self.llm = None
        if LLM_AVAILABLE:
            try:
                self.config = default_config
                self.llm = create_llm_instance(self.config.llm)
                print("LLM初始化成功，将使用LLM进行close-set评估")
            except Exception as e:
                print(f"LLM初始化失败: {e}")
                print("将使用正则表达式fallback模式")
        
        # 数据集模式映射
        self.dataset_mode_map = {
            "hotpotqa": "short-form",
            "multihop-rag": "short-form",
            "Popqa": "multi-short-form",
            "ALCE": "long-asqa",
            "quality": "close-set",
        }
        
        # 根据数据集确定评估模式
        if "narrative" in dataset_name:
            self.mode = "long-narrative"
        else:
            self.mode = self.dataset_mode_map.get(dataset_name, "short-form")
        
        # 初始化评估指标
        self.short_eval_metrics = ["accuracy", "f1", "precision", "recall", "em"]
        self.close_eval_metrics = ["accuracy"]
        self.long_narrative_metrics = [
            "bleu_1", "bleu_4", "modify_bleu_4", "bleu_1_smooth", 
            "bleu_4_smooth", "modify_bleu_4_smooth", "meteor",
            "rouge_l f1", "rouge_l precision", "rouge_l recall"
        ]
        self.long_asqa_metrics = ["str_em", "str_hit", "rougeLsum", "mauve"]
        self.multi_short_eval_metrics = ["accuracy", "f1", "precision", "recall", "em"]
        
        # 初始化ROUGE评分器
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        
        print(f"数据集: {dataset_name}")
        print(f"评估模式: {self.mode}")
        print(f"使用指标: {self.get_metrics_for_mode()}")
    
    def get_metrics_for_mode(self) -> List[str]:
        """获取当前模式使用的指标"""
        if self.mode == "short-form":
            return self.short_eval_metrics
        elif self.mode == "long-narrative":
            return self.long_narrative_metrics
        elif self.mode == "long-asqa":
            return self.long_asqa_metrics
        elif self.mode == "close-set":
            return self.close_eval_metrics
        elif self.mode == "multi-short-form":
            return self.multi_short_eval_metrics
        else:
            return self.short_eval_metrics
    
    def normalize_answer(self, s: str) -> str:
        """标准化答案文本"""
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        
        def white_space_fix(text):
            return " ".join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def exact_match_score(self, prediction: str, ground_truth: str) -> bool:
        """计算精确匹配分数"""
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)
    
    def extract_option_from_output(self, model_output: str) -> str:
        """从模型输出中提取选项字母"""
        import re
        
        # 处理输出文本
        output = model_output.strip()
        
        # 方法1: 查找"答案是X"或"answer is X"的模式
        patterns = [
            r'答案是\s*([A-Z])',
            r'answer\s+is\s+([A-Z])',
            r'正确答案是\s*([A-Z])',
            r'the\s+correct\s+answer\s+is\s+([A-Z])',
            r'选择\s*([A-Z])',
            r'choose\s+([A-Z])',
            r'option\s+([A-Z])',
            r'([A-Z])是正确的',
            r'([A-Z])\s*is\s+correct',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # 方法2: 查找单独的选项字母
        # 查找行首或空格后的单个大写字母
        pattern = r'(?:^|\s)([A-Z])(?:\s|$|[.,;:])'
        matches = re.findall(pattern, output)
        if matches:
            # 返回最后一个匹配的选项（通常是最终答案）
            return matches[-1]
        
        # 方法3: 查找最后出现的A、B、C、D选项
        pattern = r'([A-D])'
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1]
        
        # 如果都没找到，返回-1表示无法提取
        return "-1"
    
    def check_word_in_string(self, s1, s2):
        # 标准化输入字符串，使其对大小写不敏感
        s1_normalized = self.normalize_answer(s1)
        s2_normalized = self.normalize_answer(s2)
        
        # 将 s2 拆分为单词列表
        words_s2 = s2_normalized.split()
        # 遍历 s2 中的每个单词
        for word in words_s2:
            # 如果任何一个单词出现在 s1 中，则返回 1
            if word in s1_normalized:
                return 1
        # 如果循环结束都没有找到匹配的单词，则返回 0
        return 0
    
    def eval_accuracy(self, prediction: str, ground_truth: str) -> int:
        """计算准确率"""
        s1 = self.normalize_answer(prediction)
        s2 = self.normalize_answer(ground_truth)
        #return self.check_word_in_string(s1, s2)
        return 1 if s2 in s1 else 0
    
    def f1_score(self, prediction: str, ground_truth: str) -> Tuple[float, float, float]:
        """计算F1分数、精确率和召回率"""
        normalized_prediction = self.normalize_answer(prediction)
        normalized_ground_truth = self.normalize_answer(ground_truth)
        
        # 处理特殊情况
        if (normalized_prediction in ["yes", "no", "noanswer"] and 
            normalized_prediction != normalized_ground_truth):
            return 0.0, 0.0, 0.0
        if (normalized_ground_truth in ["yes", "no", "noanswer"] and 
            normalized_prediction != normalized_ground_truth):
            return 0.0, 0.0, 0.0
        
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        
        if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
            return 0.0, 0.0, 0.0
        
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0, 0.0, 0.0
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1, precision, recall
    
    def bleu_1(self, prediction: str, ground_truth: str) -> float:
        """计算BLEU-1分数"""
        try:
            return sentence_bleu([ground_truth.split()], prediction.split(), weights=(1, 0, 0, 0))
        except:
            return 0.0
    
    def bleu_4(self, prediction: str, ground_truth: str) -> float:
        """计算BLEU-4分数"""
        try:
            return sentence_bleu([ground_truth.split()], prediction.split(), weights=(0, 0, 0, 1))
        except:
            return 0.0
    
    def meteor_score(self, prediction: str, ground_truth: str) -> float:
        """计算METEOR分数"""
        try:
            return meteor_score([ground_truth.split()], prediction.split())
        except:
            return 0.0
    
    def rouge_l_score(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """计算ROUGE-L分数"""
        try:
            scores = self.rouge_scorer.score(ground_truth, prediction)
            return {
                "rouge_l f1": scores["rougeL"].fmeasure,
                "rouge_l precision": scores["rougeL"].precision,
                "rouge_l recall": scores["rougeL"].recall
            }
        except:
            return {"rouge_l f1": 0.0, "rouge_l precision": 0.0, "rouge_l recall": 0.0}
    
    def short_form_evaluation(self, df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        """短文本评估（适用于multihop-rag等数据集）"""  #TODO
        print("执行短文本评估...")
        
        accuracy_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        em_list = []
        
        # 添加进度条支持
        if TQDM_AVAILABLE:
            progress_bar = tqdm(df.iterrows(), total=len(df), desc="短文本评估", unit="条")
        else:
            progress_bar = df.iterrows()
            print(f"处理 {len(df)} 条记录...")
        
        for idx, row in progress_bar:
            prediction = row["output"]
            answer = row["answer"]
            
            # 处理预测结果
            prediction = prediction.replace("|", "\n")
            prediction = prediction.split("\n")
            prediction_str = " ".join(prediction)
            
            # 处理标准答案
            answer = answer.split("|")
            if isinstance(answer, list):
                answer_str = " ".join(answer)
            else:
                answer_str = answer
            
            # 计算各项指标
            accuracy = self.eval_accuracy(prediction_str, answer_str)
            f1, precision, recall = self.f1_score(prediction_str, answer_str)
            em = self.exact_match_score(prediction_str, answer_str)
            
            accuracy_list.append(accuracy)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            em_list.append(em)
            
            # 更新进度信息（仅在没有tqdm时）
            if not TQDM_AVAILABLE and (idx + 1) % 500 == 0:
                print(f"已处理: {idx + 1}/{len(df)} 条记录")
        
        # 计算平均值
        avg_accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        avg_f1 = sum(f1_list) * 100 / len(f1_list)
        avg_precision = sum(precision_list) * 100 / len(precision_list)
        avg_recall = sum(recall_list) * 100 / len(recall_list)
        avg_em = sum(em_list) * 100 / len(em_list)
        
        # 添加指标到DataFrame
        df["accuracy"] = accuracy_list
        df["f1"] = f1_list
        df["precision"] = precision_list
        df["recall"] = recall_list
        df["em"] = em_list
        
        results = {
            "accuracy": avg_accuracy,
            "f1": avg_f1,
            "precision": avg_precision,
            "recall": avg_recall,
            "em": avg_em,
        }
        
        print(f"评估结果:")
        print(f"  Accuracy: {avg_accuracy:.4f}%")
        print(f"  Precision: {avg_precision:.4f}%")
        print(f"  Recall: {avg_recall:.4f}%")
        print(f"  F1: {avg_f1:.4f}%")
        print(f"  Exact Match: {avg_em:.4f}%")
        
        return results, df

    def multi_short_form_evaluation(self, df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        多短文本评估（适用于Popqa等数据集）
        修改说明：更新了accuracy的计算逻辑，只要预测结果中包含任意一个正确答案，即认为准确率为1。
        """
        print("执行多短文本评估...")
        
        accuracy_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        em_list = []
        
        # 添加进度条支持
        if TQDM_AVAILABLE:
            progress_bar = tqdm(df.iterrows(), total=len(df), desc="多短文本评估", unit="条")
        else:
            progress_bar = df.iterrows()
            print(f"处理 {len(df)} 条记录...")
        
        for _, row in progress_bar:
            prediction_raw = row["output"]
            answer_raw = row["answer"]
            
            # 处理预测结果，将其转换为列表
            # 兼容以 | 或 \n 分隔的情况
            prediction_list = [p.strip() for p in prediction_raw.replace("|", "\n").split("\n")]
            # 合并成一个字符串，用于后续的F1、EM等指标计算
            prediction_str = " ".join(prediction_list)
            
            # 处理标准答案，将其转换为列表
            answer_list = [a.strip() for a in answer_raw.split("|")]
            # 合并成一个字符串，用于后续的F1、EM等指标计算
            answer_str = " ".join(answer_list)
            
            # === 更新后的准确率计算逻辑 ===
            # 检查预测结果中的任何一项是否包含任意一个正确答案
            is_correct = False
            for pred_item in prediction_list:
                for answer_item in answer_list:
                    # 标准化后检查正确答案是否作为子字符串存在于预测结果中
                    normalized_pred = self.normalize_answer(pred_item)
                    normalized_answer = self.normalize_answer(answer_item)
                    if normalized_answer in normalized_pred:
                        is_correct = True
                        break
                if is_correct:
                    break
            
            # 如果找到匹配项，准确率为1，否则为0
            accuracy = 1.0 if is_correct else 0.0
            
            # 计算其他各项指标（保持原有的逻辑）
            f1, precision, recall = self.f1_score(prediction_str, answer_str)
            em = self.exact_match_score(prediction_str, answer_str)
            
            accuracy_list.append(accuracy)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            em_list.append(em)
        
        # 计算平均值
        avg_accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        avg_f1 = sum(f1_list) * 100 / len(f1_list)
        avg_precision = sum(precision_list) * 100 / len(precision_list)
        avg_recall = sum(recall_list) * 100 / len(recall_list)
        avg_em = sum(em_list) * 100 / len(em_list)
        
        # 添加指标到DataFrame
        df["accuracy"] = accuracy_list
        df["f1"] = f1_list
        df["precision"] = precision_list
        df["recall"] = recall_list
        df["em"] = em_list
        
        results = {
            "accuracy": avg_accuracy,
            "f1": avg_f1,
            "precision": avg_precision,
            "recall": avg_recall,
            "em": avg_em,
        }
        
        print(f"评估结果:")
        print(f"  Accuracy: {avg_accuracy:.4f}%")
        print(f"  Precision: {avg_precision:.4f}%")
        print(f"  Recall: {avg_recall:.4f}%")
        print(f"  F1: {avg_f1:.4f}%")
        print(f"  Exact Match: {avg_em:.4f}%")
        
        return results, df
    
    async def close_set_evaluation(self, df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        闭集评估（适用于质量评估数据集）
        按照原本Evaluation.py的模式，使用LLM从模型输出中提取选项字母，然后与答案索引进行匹配
        """
        print("执行闭集评估...")
        
        # 检查必要的列
        required_columns = ["question", "answer", "output"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"警告：缺少列 {missing_columns}，将尝试推断答案索引")
        
        # 检查是否有answer_idx列，如果没有则尝试从answer中推断
        if "answer_idx" not in df.columns:
            print("未找到answer_idx列，尝试从answer列推断...")
            df["answer_idx"] = df["answer"].apply(self._infer_answer_idx)
        
        # 使用LLM提取选项（按照原本的模式）
        if self.llm is not None:
            print("使用LLM提取选项...")
            
            # 创建进度条
            if TQDM_AVAILABLE:
                progress_bar = tqdm(df.iterrows(), total=len(df), desc="LLM提取选项", unit="条")
            else:
                progress_bar = df.iterrows()
                print(f"处理 {len(df)} 条记录...")
            
            for index, row in progress_bar:
                prompt = CLOSE_EXTRACT_OPTION_PORMPT.format(
                    question=row["question"], model_output=row["output"]
                )
                try:
                    response = await self.llm.aask(msg=prompt, format="json")
                    df.loc[index, "extract_output"] = response["predict"]
                except Exception as e:
                    if TQDM_AVAILABLE:
                        tqdm.write(f"LLM调用失败 (第{index}行): {e}")
                    else:
                        print(f"LLM调用失败 (第{index}行): {e}")
                    df.loc[index, "extract_output"] = "-1"
                
                # 更新进度信息（仅在没有tqdm时）
                if not TQDM_AVAILABLE and (index + 1) % 100 == 0:
                    print(f"已处理: {index + 1}/{len(df)} 条记录")
            
            print("LLM提取选项完成。")
        else:
            print("LLM不可用，使用正则表达式提取选项...")
            
            # 为正则表达式模式也添加进度条
            if TQDM_AVAILABLE:
                progress_bar = tqdm(df.iterrows(), total=len(df), desc="正则提取选项", unit="条")
            else:
                progress_bar = df.iterrows()
                print(f"处理 {len(df)} 条记录...")
            
            for idx, row in progress_bar:
                prediction = row["output"]
                extracted_option = self.extract_option_from_output(prediction)
                df.loc[idx, "extract_output"] = extracted_option
                
                # 更新进度信息（仅在没有tqdm时）
                if not TQDM_AVAILABLE and (idx + 1) % 500 == 0:
                    print(f"已处理: {idx + 1}/{len(df)} 条记录")
        
        # 计算准确率
        accuracy_list = []
        label_list, pred_list = self.get_label_pred_list(
            df, "extract_output", "answer_idx"
        )

        for prediction, answer in zip(pred_list, label_list):
            prediction = str(prediction).strip()
            answer = str(answer).strip()
            accuracy = self.exact_match_score(prediction, answer)
            accuracy_list.append(accuracy)

        avg_accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        df["accuracy"] = accuracy_list
        
        results = {
            "accuracy": avg_accuracy,
        }
        
        print(f"评估结果:")
        print(f"  Accuracy: {avg_accuracy:.4f}%")
        successful_extractions = len([x for x in df["extract_output"] if x != "-1"])
        print(f"  成功提取选项: {successful_extractions}/{len(df)}")
        
        return results, df
    
    def _infer_answer_idx(self, answer_text: str) -> str:
        """从答案文本中推断答案索引"""
        import re
        
        # 如果答案本身就是单个字母，直接返回
        if len(answer_text.strip()) == 1 and answer_text.strip().upper() in 'ABCD':
            return answer_text.strip().upper()
        
        # 查找答案文本中的选项字母
        pattern = r'([A-D])'
        matches = re.findall(pattern, answer_text.upper())
        if matches:
            return matches[0]
        
        # 如果都没找到，返回-1
        return "-1"
    
    def get_label_pred_list(self, df: pd.DataFrame, pred_col: str, label_col: str) -> Tuple[List, List]:
        """获取标签和预测列表"""
        label_list = df[label_col].tolist()
        pred_list = df[pred_col].tolist()
        return label_list, pred_list
    
    def long_narrative_evaluation(self, df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        """长文本叙述评估"""
        print("执行长文本叙述评估...")
        
        bleu_1_list = []
        bleu_4_list = []
        meteor_list = []
        rouge_l_f1_list = []
        rouge_l_precision_list = []
        rouge_l_recall_list = []
        
        # 添加进度条支持
        if TQDM_AVAILABLE:
            progress_bar = tqdm(df.iterrows(), total=len(df), desc="长文本叙述评估", unit="条")
        else:
            progress_bar = df.iterrows()
            print(f"处理 {len(df)} 条记录...")
        
        for idx, row in progress_bar:
            prediction = row["output"]
            answer = row["answer"]
            
            # 处理预测结果
            prediction = prediction.replace("|", "\n")
            prediction = prediction.split("\n")
            prediction_str = " ".join(prediction)
            
            # 计算各项指标
            bleu_1_score = self.bleu_1(prediction_str, answer)
            bleu_4_score = self.bleu_4(prediction_str, answer)
            meteor_score_val = self.meteor_score(prediction_str, answer)
            rouge_scores = self.rouge_l_score(prediction_str, answer)
            
            bleu_1_list.append(bleu_1_score)
            bleu_4_list.append(bleu_4_score)
            meteor_list.append(meteor_score_val)
            rouge_l_f1_list.append(rouge_scores["rouge_l f1"])
            rouge_l_precision_list.append(rouge_scores["rouge_l precision"])
            rouge_l_recall_list.append(rouge_scores["rouge_l recall"])
            
            # 更新进度信息（仅在没有tqdm时）
            if not TQDM_AVAILABLE and (idx + 1) % 500 == 0:
                print(f"已处理: {idx + 1}/{len(df)} 条记录")
        
        # 计算平均值
        avg_bleu_1 = sum(bleu_1_list) * 100 / len(bleu_1_list)
        avg_bleu_4 = sum(bleu_4_list) * 100 / len(bleu_4_list)
        avg_meteor = sum(meteor_list) * 100 / len(meteor_list)
        avg_rouge_l_f1 = sum(rouge_l_f1_list) * 100 / len(rouge_l_f1_list)
        avg_rouge_l_precision = sum(rouge_l_precision_list) * 100 / len(rouge_l_precision_list)
        avg_rouge_l_recall = sum(rouge_l_recall_list) * 100 / len(rouge_l_recall_list)
        
        # 添加指标到DataFrame
        df["bleu_1"] = bleu_1_list
        df["bleu_4"] = bleu_4_list
        df["meteor"] = meteor_list
        df["rouge_l_f1"] = rouge_l_f1_list
        df["rouge_l_precision"] = rouge_l_precision_list
        df["rouge_l_recall"] = rouge_l_recall_list
        
        results = {
            "bleu_1": avg_bleu_1,
            "bleu_4": avg_bleu_4,
            "meteor": avg_meteor,
            "rouge_l f1": avg_rouge_l_f1,
            "rouge_l precision": avg_rouge_l_precision,
            "rouge_l recall": avg_rouge_l_recall,
        }
        
        print(f"评估结果:")
        print(f"  BLEU-1: {avg_bleu_1:.4f}%")
        print(f"  BLEU-4: {avg_bleu_4:.4f}%")
        print(f"  METEOR: {avg_meteor:.4f}%")
        print(f"  ROUGE-L F1: {avg_rouge_l_f1:.4f}%")
        print(f"  ROUGE-L Precision: {avg_rouge_l_precision:.4f}%")
        print(f"  ROUGE-L Recall: {avg_rouge_l_recall:.4f}%")
        
        return results, df
    
    async def evaluate(self, result_path: str) -> Tuple[Dict[str, float], pd.DataFrame]:
        """执行评估"""
        print(f"加载结果文件: {result_path}")
        
        # 检查文件是否存在
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"结果文件不存在: {result_path}")
        
        # 加载结果数据
        try:
            # 先尝试读取为标准JSON格式
            try:
                df = pd.read_json(result_path)
            except:
                # 如果失败，尝试JSONL格式
                df = pd.read_json(result_path, lines=True) #TODO 标准GraphRAG框架格式
            print(f"成功加载 {len(df)} 条记录")
        except Exception as e:
            raise ValueError(f"无法解析结果文件: {e}")
        
        # 检查必要的列
        required_columns = ["question", "answer", "output"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"结果文件缺少必要的列: {missing_columns}")
        
        # 根据模式选择评估方法
        if self.mode == "short-form":
            return self.short_form_evaluation(df)
        elif self.mode == "long-narrative":
            return self.long_narrative_evaluation(df)
        elif self.mode == "long-asqa":
            # 这里可以添加ASQA评估逻辑
            print("ASQA评估模式暂未实现，使用短文本评估")
            return self.short_form_evaluation(df)
        elif self.mode == "close-set":
            return await self.close_set_evaluation(df)
        elif self.mode == "multi-short-form":
            return self.multi_short_form_evaluation(df)
        else:
            print("未知评估模式，使用短文本评估")
            return self.short_form_evaluation(df)


def save_results(results: Dict[str, float], df: pd.DataFrame, output_dir: str, 
                result_path: str, dataset_name: str):
    """保存评估结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(result_path))[0]
    
    # 保存详细结果
    detailed_path = os.path.join(output_dir, f"{base_name}_detailed_results.json")
    df.to_json(detailed_path, orient="records", lines=True, force_ascii=False)
    print(f"详细结果已保存到: {detailed_path}")
    
    # 保存汇总指标
    metrics_path = os.path.join(output_dir, f"{base_name}_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": dataset_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": results
        }, f, indent=2, ensure_ascii=False)
    print(f"汇总指标已保存到: {metrics_path}")
    
    # 保存人类可读的报告
    report_path = os.path.join(output_dir, f"{base_name}_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"RAG评估报告\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"样本数量: {len(df)}\n")
        f.write(f"\n评估指标:\n")
        f.write("-" * 30 + "\n")
        
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}%\n")
        
        f.write(f"\n详细统计:\n")
        f.write("-" * 30 + "\n")
        for metric in results.keys():
            if metric in df.columns:
                values = df[metric]
                f.write(f"{metric}:\n")
                f.write(f"  平均值: {values.mean():.4f}%\n")
                f.write(f"  标准差: {values.std():.4f}%\n")
                f.write(f"  最小值: {values.min():.4f}%\n")
                f.write(f"  最大值: {values.max():.4f}%\n")
                f.write(f"  中位数: {values.median():.4f}%\n\n")
    
    print(f"评估报告已保存到: {report_path}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG结果评估脚本")
    parser.add_argument("--result_path", type=str, required=True,
                       help="RAG结果文件路径 (JSON格式)")
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="数据集名称 (如: Popqa, hotpotqa等)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="输出目录 (默认: ./evaluation_results)")
    parser.add_argument("--nltk_path", type=str, default="/data/zhy/nltk_data",
                       help="NLTK数据路径")
    
    args = parser.parse_args()
    
    # 设置NLTK路径
    global nltk_path
    nltk_path = args.nltk_path
    nltk.data.path.append(nltk_path)
    
    try:
        # 创建评估器
        evaluator = RAGEvaluator(args.dataset_name)
        
        # 执行评估
        results, df = await evaluator.evaluate(args.result_path)
        
        # 保存结果
        save_results(results, df, args.output_dir, args.result_path, args.dataset_name)
        
        print("\n评估完成!")
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        sys.exit(1)


# Close-set评估提示词（从原项目Evaluation.py复制）
CLOSE_EXTRACT_OPTION_PORMPT = """
You are given a model output which is a string. The model output is a list of options. You have to extract the option letter from the model output.

# GOAL

Your goal is to extract the option letter directly from the model output. You should not rely on any external knowledge or context to answer. Simply extract the option letter as stated in the model output.

# FORMAT

Please provide your answer in the following JSON format:

- ANSWER_OPTION: the option letter extracted from the model output.

    {{
        "model_output": <answer_option>
    }}

### Example 1
-----------
# INPUT:

Question:
How much time has passed between Blake's night with Eldoria and his search for Sabrina York in his mind-world?
A: 7 years
B: 10 hours
C: 12 years
D: 1 hour

# Model Output: 
I think the answer is 7 years.

OUTPUT:
    {{
        "predict": "A"
    }}

### Example 2
-----------
# INPUT:

Question:
How much time has passed between Blake's night with Eldoria and his search for Sabrina York in his mind-world?
A: 7 years
B: 10 hours
C: 12 years
D: 1 hour

# Model Output: 
The correct answer is C.

OUTPUT:
    {{
        "predict": "C"
    }}
    
### EXAMPLE 3
-----------

# INPUT:

Question:
Donald Trump is the president of:
A: China
B: Canada
C: France
D: Spain

# Model Output: 
The correct answer is: None of the above.

OUTPUT:
    {{
        "predict": "-1"
    }}

Now please the output based on the given question and model output.

### Real Data
# INPUT:

Question:
{question}

# Model Output:
{model_output}

OUTPUT:"""


if __name__ == "__main__":
    asyncio.run(main())