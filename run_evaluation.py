#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Result Evaluation Script
Used to evaluate the results generated during the RAG process

python run_evaluation.py --result_path /path/to/results.json --dataset_name Popqa --output_dir /path/to/output

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

from tqdm import tqdm
TQDM_AVAILABLE = True

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir  
sys.path.append(project_root)

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score
    import nltk
    from nltk import sent_tokenize
    from rouge_score import rouge_scorer, scoring
except ImportError as e:
    print(": pip install nltk rouge-score")
    sys.exit(1)

# Try to import the original project's LLM module
try:
    from Option.Config2 import default_config
    from Core.Provider.BaseLLM import BaseLLM
    from Core.Provider.LLMProviderRegister import create_llm_instance
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unable to import LLM module: {e}")
    print("Close-set evaluation will use regex fallback mode")
    LLM_AVAILABLE = False

# Set NLTK data path
nltk_path = "/data/zhy/nltk_data"  # Adjust according to your environment
nltk.data.path.append(nltk_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading NLTK data...")
    nltk.download("punkt", download_dir=nltk_path)
    nltk.download("wordnet", download_dir=nltk_path)


class RAGEvaluator:
    """RAG Result Evaluator"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        
        # Try to initialize LLM (if module is available)
        self.llm = None
        if LLM_AVAILABLE:
            try:
                self.config = default_config
                self.llm = create_llm_instance(self.config.llm)
                print("LLM initialization successful, will use LLM for close-set evaluation")
            except Exception as e:
                print(f"LLM initialization failed: {e}")
                print("Will use regex fallback mode")
        
        # Dataset mode mapping
        self.dataset_mode_map = {
            "hotpotqa": "short-form",
            "multihop-rag": "short-form",
            "Popqa": "multi-short-form",
            "ALCE": "long-asqa",
            "quality": "close-set",
        }
        
        # Determine evaluation mode based on dataset
        if "narrative" in dataset_name:
            self.mode = "long-narrative"
        else:
            self.mode = self.dataset_mode_map.get(dataset_name, "short-form")
        
        # Initialize evaluation metrics
        self.short_eval_metrics = ["accuracy", "f1", "precision", "recall", "em"]
        self.close_eval_metrics = ["accuracy"]
        self.long_narrative_metrics = [
            "bleu_1", "bleu_4", "modify_bleu_4", "bleu_1_smooth", 
            "bleu_4_smooth", "modify_bleu_4_smooth", "meteor",
            "rouge_l f1", "rouge_l precision", "rouge_l recall"
        ]
        self.long_asqa_metrics = ["str_em", "str_hit", "rougeLsum", "mauve"]
        self.multi_short_eval_metrics = ["accuracy", "f1", "precision", "recall", "em"]
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        
        print(f"Dataset: {dataset_name}")
        print(f"Evaluation mode: {self.mode}")
        print(f"Using metrics: {self.get_metrics_for_mode()}")
    
    def get_metrics_for_mode(self) -> List[str]:
        """Get metrics used for current mode"""
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
        """Normalize answer text"""
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
        """Calculate exact match score"""
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)
    
    def extract_option_from_output(self, model_output: str) -> str:
        """Extract option letter from model output"""
        import re
        
        # Process output text
        output = model_output.strip()
        
        # Method 1: Look for patterns like "answer is X" or "correct answer is X"
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
        
        # Method 2: Look for standalone option letters
        # Find single uppercase letters at line start or after whitespace
        pattern = r'(?:^|\s)([A-Z])(?:\s|$|[.,;:])'
        matches = re.findall(pattern, output)
        if matches:
            # Return the last matched option (usually the final answer)
            return matches[-1]
        
        # Method 3: Find the last occurrence of A, B, C, D options
        pattern = r'([A-D])'
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1]
        
        # If none found, return -1 to indicate extraction failure
        return "-1"
    
    def check_word_in_string(self, s1, s2):
        # Normalize input strings to be case-insensitive
        s1_normalized = self.normalize_answer(s1)
        s2_normalized = self.normalize_answer(s2)
        
        # Split s2 into word list
        words_s2 = s2_normalized.split()
        # Iterate through each word in s2
        for word in words_s2:
            # If any word appears in s1, return 1
            if word in s1_normalized:
                return 1
        # If no matching words found after loop, return 0
        return 0
    
    def eval_accuracy(self, prediction: str, ground_truth: str) -> int:
        """Calculate accuracy"""
        s1 = self.normalize_answer(prediction)
        s2 = self.normalize_answer(ground_truth)
        #return self.check_word_in_string(s1, s2)
        return 1 if s2 in s1 else 0
    
    def f1_score(self, prediction: str, ground_truth: str) -> Tuple[float, float, float]:
        """Calculate F1 score, precision, and recall"""
        normalized_prediction = self.normalize_answer(prediction)
        normalized_ground_truth = self.normalize_answer(ground_truth)
        
        # Handle special cases
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
        """Calculate BLEU-1 score"""
        try:
            return sentence_bleu([ground_truth.split()], prediction.split(), weights=(1, 0, 0, 0))
        except:
            return 0.0
    
    def bleu_4(self, prediction: str, ground_truth: str) -> float:
        """Calculate BLEU-4 score"""
        try:
            return sentence_bleu([ground_truth.split()], prediction.split(), weights=(0, 0, 0, 1))
        except:
            return 0.0
    
    def meteor_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate METEOR score"""
        try:
            return meteor_score([ground_truth.split()], prediction.split())
        except:
            return 0.0
    
    def rouge_l_score(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Calculate ROUGE-L score"""
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
        """Short-form evaluation (suitable for multihop-rag and similar datasets)"""  #TODO
        print("Executing short-form evaluation...")
        
        accuracy_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        em_list = []
        
        # Add progress bar support
        if TQDM_AVAILABLE:
            progress_bar = tqdm(df.iterrows(), total=len(df), desc="Short-form evaluation", unit="items")
        else:
            progress_bar = df.iterrows()
            print(f"Processing {len(df)} records...")
        
        for idx, row in progress_bar:
            prediction = row["output"]
            answer = row["answer"]
            
            # Process prediction results
            prediction = prediction.replace("|", "\n")
            prediction = prediction.split("\n")
            prediction_str = " ".join(prediction)
            
            # Process ground truth answers
            answer = answer.split("|")
            if isinstance(answer, list):
                answer_str = " ".join(answer)
            else:
                answer_str = answer
            
            # Calculate various metrics
            accuracy = self.eval_accuracy(prediction_str, answer_str)
            f1, precision, recall = self.f1_score(prediction_str, answer_str)
            em = self.exact_match_score(prediction_str, answer_str)
            
            accuracy_list.append(accuracy)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            em_list.append(em)
            
            # Update progress information (only when tqdm is not available)
            if not TQDM_AVAILABLE and (idx + 1) % 500 == 0:
                print(f"Processed: {idx + 1}/{len(df)} records")
        
        # Calculate averages
        avg_accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        avg_f1 = sum(f1_list) * 100 / len(f1_list)
        avg_precision = sum(precision_list) * 100 / len(precision_list)
        avg_recall = sum(recall_list) * 100 / len(recall_list)
        avg_em = sum(em_list) * 100 / len(em_list)
        
        # Add metrics to DataFrame
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
        
        print(f"Evaluation results:")
        print(f"  Accuracy: {avg_accuracy:.4f}%")
        print(f"  Precision: {avg_precision:.4f}%")
        print(f"  Recall: {avg_recall:.4f}%")
        print(f"  F1: {avg_f1:.4f}%")
        print(f"  Exact Match: {avg_em:.4f}%")
        
        return results, df

    def multi_short_form_evaluation(self, df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Multi short-form evaluation (suitable for Popqa and similar datasets)
        Modification note: Updated accuracy calculation logic - if the prediction contains any correct answer, accuracy is considered 1.
        """
        print("Executing multi short-form evaluation...")
        
        accuracy_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        em_list = []
        
        # Add progress bar support
        if TQDM_AVAILABLE:
            progress_bar = tqdm(df.iterrows(), total=len(df), desc="Multi short-form evaluation", unit="items")
        else:
            progress_bar = df.iterrows()
            print(f"Processing {len(df)} records...")
        
        for _, row in progress_bar:
            prediction_raw = row["output"]
            answer_raw = row["answer"]
            
            # Process prediction results, convert to list
            # Compatible with | or \n separators
            prediction_list = [p.strip() for p in prediction_raw.replace("|", "\n").split("\n")]
            # Merge into a single string for subsequent F1, EM metrics calculation
            prediction_str = " ".join(prediction_list)
            
            # Process ground truth answers, convert to list
            answer_list = [a.strip() for a in answer_raw.split("|")]
            # Merge into a single string for subsequent F1, EM metrics calculation
            answer_str = " ".join(answer_list)
            
            # === Updated accuracy calculation logic ===
            # Check if any item in prediction contains any correct answer
            is_correct = False
            for pred_item in prediction_list:
                for answer_item in answer_list:
                    # After normalization, check if correct answer exists as substring in prediction
                    normalized_pred = self.normalize_answer(pred_item)
                    normalized_answer = self.normalize_answer(answer_item)
                    if normalized_answer in normalized_pred:
                        is_correct = True
                        break
                if is_correct:
                    break
            
            # If match found, accuracy is 1, otherwise 0
            accuracy = 1.0 if is_correct else 0.0
            
            # Calculate other metrics (keep original logic)
            f1, precision, recall = self.f1_score(prediction_str, answer_str)
            em = self.exact_match_score(prediction_str, answer_str)
            
            accuracy_list.append(accuracy)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            em_list.append(em)
        
        # Calculate averages
        avg_accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        avg_f1 = sum(f1_list) * 100 / len(f1_list)
        avg_precision = sum(precision_list) * 100 / len(precision_list)
        avg_recall = sum(recall_list) * 100 / len(recall_list)
        avg_em = sum(em_list) * 100 / len(em_list)
        
        # Add metrics to DataFrame
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
        
        print(f"Evaluation results:")
        print(f"  Accuracy: {avg_accuracy:.4f}%")
        print(f"  Precision: {avg_precision:.4f}%")
        print(f"  Recall: {avg_recall:.4f}%")
        print(f"  F1: {avg_f1:.4f}%")
        print(f"  Exact Match: {avg_em:.4f}%")
        
        return results, df
    
    async def close_set_evaluation(self, df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Closed-set evaluation (suitable for quality evaluation datasets)
        Following the original Evaluation.py pattern, use LLM to extract option letters from model output and match with answer indices
        """
        print("Executing closed-set evaluation...")
        
        # 检查必要的列
        required_columns = ["question", "answer", "output"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns {missing_columns}, will try to infer answer indices")
        
        # Check if answer_idx column exists, if not try to infer from answer column
        if "answer_idx" not in df.columns:
            print("answer_idx column not found, trying to infer from answer column...")
            df["answer_idx"] = df["answer"].apply(self._infer_answer_idx)
        
        # Use LLM to extract options (following original pattern)
        if self.llm is not None:
            print("Using LLM to extract options...")
            
            # Create progress bar
            if TQDM_AVAILABLE:
                progress_bar = tqdm(df.iterrows(), total=len(df), desc="LLM option extraction", unit="items")
            else:
                progress_bar = df.iterrows()
                print(f"Processing {len(df)} records...")
            
            for index, row in progress_bar:
                prompt = CLOSE_EXTRACT_OPTION_PORMPT.format(
                    question=row["question"], model_output=row["output"]
                )
                try:
                    response = await self.llm.aask(msg=prompt, format="json")
                    df.loc[index, "extract_output"] = response["predict"]
                except Exception as e:
                    if TQDM_AVAILABLE:
                        tqdm.write(f"LLM call failed (row {index}): {e}")
                    else:
                        print(f"LLM call failed (row {index}): {e}")
                    df.loc[index, "extract_output"] = "-1"
                
                # Update progress information (only when no tqdm)
                if not TQDM_AVAILABLE and (index + 1) % 100 == 0:
                    print(f"Processed: {index + 1}/{len(df)} records")
            
            print("LLM option extraction completed.")
        else:
            print("LLM unavailable, using regex to extract options...")
            
            # Add progress bar for regex pattern as well
            if TQDM_AVAILABLE:
                progress_bar = tqdm(df.iterrows(), total=len(df), desc="Regex option extraction", unit="items")
            else:
                progress_bar = df.iterrows()
                print(f"Processing {len(df)} records...")
            
            for idx, row in progress_bar:
                prediction = row["output"]
                extracted_option = self.extract_option_from_output(prediction)
                df.loc[idx, "extract_output"] = extracted_option
                
                # Update progress information (only when no tqdm)
                if not TQDM_AVAILABLE and (idx + 1) % 500 == 0:
                    print(f"Processed: {idx + 1}/{len(df)} records")
        
        # Calculate accuracy
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
        
        print(f"Evaluation results:")
        print(f"  Accuracy: {avg_accuracy:.4f}%")
        successful_extractions = len([x for x in df["extract_output"] if x != "-1"])
        print(f"  Successful option extractions: {successful_extractions}/{len(df)}")
        
        return results, df
    
    def _infer_answer_idx(self, answer_text: str) -> str:
        """Infer answer index from answer text"""
        import re
        
        # If answer is a single letter, return directly
        if len(answer_text.strip()) == 1 and answer_text.strip().upper() in 'ABCD':
            return answer_text.strip().upper()
        
        # Find option letters in answer text
        pattern = r'([A-D])'
        matches = re.findall(pattern, answer_text.upper())
        if matches:
            return matches[0]
        
        # If nothing found, return -1
        return "-1"
    
    def get_label_pred_list(self, df: pd.DataFrame, pred_col: str, label_col: str) -> Tuple[List, List]:
        """Get label and prediction list"""
        label_list = df[label_col].tolist()
        pred_list = df[pred_col].tolist()
        return label_list, pred_list
    
    def long_narrative_evaluation(self, df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Long-form narrative evaluation"""
        print("Executing long-form narrative evaluation...")
        
        bleu_1_list = []
        bleu_4_list = []
        meteor_list = []
        rouge_l_f1_list = []
        rouge_l_precision_list = []
        rouge_l_recall_list = []
        
        # Add progress bar support
        if TQDM_AVAILABLE:
            progress_bar = tqdm(df.iterrows(), total=len(df), desc="Long-form narrative evaluation", unit="items")
        else:
            progress_bar = df.iterrows()
            print(f"Processing {len(df)} records...")
        
        for idx, row in progress_bar:
            prediction = row["output"]
            answer = row["answer"]
            
            # Process prediction results
            prediction = prediction.replace("|", "\n")
            prediction = prediction.split("\n")
            prediction_str = " ".join(prediction)
            
            # Calculate various metrics
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
            
            # Update progress information (only when no tqdm)
            if not TQDM_AVAILABLE and (idx + 1) % 500 == 0:
                print(f"Processed: {idx + 1}/{len(df)} records")
        
        # Calculate averages
        avg_bleu_1 = sum(bleu_1_list) * 100 / len(bleu_1_list)
        avg_bleu_4 = sum(bleu_4_list) * 100 / len(bleu_4_list)
        avg_meteor = sum(meteor_list) * 100 / len(meteor_list)
        avg_rouge_l_f1 = sum(rouge_l_f1_list) * 100 / len(rouge_l_f1_list)
        avg_rouge_l_precision = sum(rouge_l_precision_list) * 100 / len(rouge_l_precision_list)
        avg_rouge_l_recall = sum(rouge_l_recall_list) * 100 / len(rouge_l_recall_list)
        
        # Add metrics to DataFrame
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
        
        print(f"Evaluation results:")
        print(f"  BLEU-1: {avg_bleu_1:.4f}%")
        print(f"  BLEU-4: {avg_bleu_4:.4f}%")
        print(f"  METEOR: {avg_meteor:.4f}%")
        print(f"  ROUGE-L F1: {avg_rouge_l_f1:.4f}%")
        print(f"  ROUGE-L Precision: {avg_rouge_l_precision:.4f}%")
        print(f"  ROUGE-L Recall: {avg_rouge_l_recall:.4f}%")
        
        return results, df
    
    async def evaluate(self, result_path: str) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Execute evaluation"""
        print(f"Loading result file: {result_path}")
        
        # Check if file exists
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Result file does not exist: {result_path}")
        
        # Load result data
        try:
            # First try to read as standard JSON format
            try:
                df = pd.read_json(result_path)
            except:
                # If failed, try JSONL format
                df = pd.read_json(result_path, lines=True) #TODO Standard GraphRAG framework format
            print(f"Successfully loaded {len(df)} records")
        except Exception as e:
            raise ValueError(f"Unable to parse result file: {e}")
        
        # Check required columns
        required_columns = ["question", "answer", "output"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Result file missing required columns: {missing_columns}")
        
        # Select evaluation method based on mode
        if self.mode == "short-form":
            return self.short_form_evaluation(df)
        elif self.mode == "long-narrative":
            return self.long_narrative_evaluation(df)
        elif self.mode == "long-asqa":
            # ASQA evaluation logic can be added here
            print("ASQA evaluation mode not yet implemented, using short-form evaluation")
            return self.short_form_evaluation(df)
        elif self.mode == "close-set":
            return await self.close_set_evaluation(df)
        elif self.mode == "multi-short-form":
            return self.multi_short_form_evaluation(df)
        else:
            print("Unknown evaluation mode, using short-form evaluation")
            return self.short_form_evaluation(df)


def save_results(results: Dict[str, float], df: pd.DataFrame, output_dir: str, 
                result_path: str, dataset_name: str):
    """Save evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(result_path))[0]
    
    # Save detailed results
    detailed_path = os.path.join(output_dir, f"{base_name}_detailed_results.json")
    df.to_json(detailed_path, orient="records", lines=True, force_ascii=False)
    print(f"Detailed results saved to: {detailed_path}")
    
    # Save summary metrics
    metrics_path = os.path.join(output_dir, f"{base_name}_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": dataset_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Summary metrics saved to: {metrics_path}")
    
    # Save human-readable report
    report_path = os.path.join(output_dir, f"{base_name}_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"RAG Evaluation Report\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Evaluation Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample Count: {len(df)}\n")
        f.write(f"\nEvaluation Metrics:\n")
        f.write("-" * 30 + "\n")
        
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}%\n")
        
        f.write(f"\nDetailed Statistics:\n")
        f.write("-" * 30 + "\n")
        for metric in results.keys():
            if metric in df.columns:
                values = df[metric]
                f.write(f"{metric}:\n")
                f.write(f"  Mean: {values.mean():.4f}%\n")
                f.write(f"  Std Dev: {values.std():.4f}%\n")
                f.write(f"  Min: {values.min():.4f}%\n")
                f.write(f"  Max: {values.max():.4f}%\n")
                f.write(f"  Median: {values.median():.4f}%\n\n")
    
    print(f"Evaluation report saved to: {report_path}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RAG Result Evaluation Script")
    parser.add_argument("--result_path", type=str, required=True,
                       help="RAG result file path (JSON format)")
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="Dataset name (e.g., Popqa, hotpotqa, etc.)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory (default: ./evaluation_results)")
    parser.add_argument("--nltk_path", type=str, default="/data/zhy/nltk_data",
                       help="NLTK data path")
    
    args = parser.parse_args()
    
    # Set NLTK path
    global nltk_path
    nltk_path = args.nltk_path
    nltk.data.path.append(nltk_path)
    
    try:
        # Create evaluator
        evaluator = RAGEvaluator(args.dataset_name)
        
        # Execute evaluation
        results, df = await evaluator.evaluate(args.result_path)
        
        # Save results
        save_results(results, df, args.output_dir, args.result_path, args.dataset_name)
        
        print("\nEvaluation completed!")
        
    except Exception as e:
        print(f"Error occurred during evaluation: {e}")
        sys.exit(1)


# Close-set evaluation prompt (copied from original project Evaluation.py)
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