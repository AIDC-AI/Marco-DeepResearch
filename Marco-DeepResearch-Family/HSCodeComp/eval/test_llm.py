import os
import json
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re 
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))


def extract_hscode_from_text(text: str) -> str:
    """Parse HScode from LaTeX boxed format"""
    if not text:
        return ""

    # Look for \boxed{} pattern
    latex_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(latex_pattern, text)
    matches=[_ for _ in matches if len(_)>=10]
    if matches:
        # Extract digits from the first match
        return re.sub(r'[^0-9]', '', str(matches[0]))

    return ""


def classify_product(record: Dict[str, Any], model_name: str) -> Optional[str]:
    """Classify a single product using question directly"""
    try:
        messages = [
            {"role": "user", "content": record["question"]}
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )

        if response and response.choices:
            return response.choices[0].message.content.strip()
        else:
            return None

    except Exception as e:
        logger.error(f"Error classifying record {record.get('id', 'unknown')}: {e}")
        return None

def load_dataset(file_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file"""
    try:
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                records.append(json.loads(line))
        return records

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return []

class TestRunner:
    """Test runner for HSCode classification"""

    def __init__(self, max_workers: int = 4):
        self.results = {}
        self.max_workers = max_workers
        self.lock = threading.Lock()

    def classify_single_record(self, record: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Classify a single record with a specific model"""
        try:
            start_time = time.time()
            llm_output = classify_product(record, model_name)
            end_time = time.time()

            parsed_hscode = extract_hscode_from_text(llm_output) if llm_output else ""
 
            return {
                "task_id": record.get("task_id", ""),
                "question": record.get("question", ""),
                "answer": record.get("hs_code", ""),
                "llm_output": llm_output if llm_output else "",
                "parsed_prediction": parsed_hscode,
                "processing_time": end_time - start_time,
                "success": bool(parsed_hscode)
            }
        except Exception as e:
            logger.error(f"Error classifying record {record.get('task_id', 'unknown')} with model {model_name}: {e}")
            return {
                "task_id": record.get("task_id", ""),
                "question": record.get("question", ""),
                "answer": record.get("hs_code", ""),
                "llm_output": "",
                "parsed_prediction": "",
                "processing_time": 0,
                "success": False,
                "error": str(e)
            }
    
    def run_test(self, records: List[Dict[str, Any]], models: List[str]):
        """Run test on records with specified models"""

        for model_idx, model_name in enumerate(models):
            print(f"Processing model {model_idx + 1}/{len(models)}: {model_name}")

            self.results[model_name] = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_record = {
                    executor.submit(self.classify_single_record, record, model_name): record
                    for record in records
                }

                with tqdm(total=len(records), desc=model_name,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

                    for future in as_completed(future_to_record):
                        record = future_to_record[future]
                        try:
                            result = future.result()

                            with self.lock:
                                self.results[model_name].append(result)

                            if result["success"]:
                                pbar.set_postfix_str(f"{result['parsed_prediction']} ({result['processing_time']:.1f}s)")
                            else:
                                pbar.set_postfix_str(f"FAILED ({result['processing_time']:.1f}s)")

                        except Exception as e:
                            logger.error(f"Error processing record {record.get('task_id', 'unknown')}: {e}")
                            with self.lock:
                                self.results[model_name].append({
                                    "task_id": record.get("task_id", ""),
                                    "question": record.get("question", ""),
                                    "answer": record.get("hs_code", ""),
                                    "llm_output": "",
                                    "parsed_prediction": "",
                                    "processing_time": 0,
                                    "success": False,
                                    "error": str(e)
                                })
                            pbar.set_postfix_str("ERROR")

                        pbar.update(1)

            self.save_single_model_results(model_name)

            successful = sum(1 for r in self.results[model_name] if r["success"])
            total = len(self.results[model_name])
            success_rate = (successful / total * 100) if total > 0 else 0
            print(f"Model {model_name}: {successful}/{total} ({success_rate:.1f}% success)")

        print("Test completed!")
    
    def save_single_model_results(self, model_name: str):
        """Save results for a single model"""
        try:
            output_file = f"{output_folder}/logs/{model_name}_results.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results[model_name], f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Failed to save results for {model_name}: {e}")
    
    
def main():
    dataset_file = "data/test_data.jsonl"
    models_to_test = ["gpt-4o"]
    global output_folder
    output_folder = "output"

    records = load_dataset(dataset_file)
    if not records:
        return

    test_runner = TestRunner(max_workers=20)
    test_runner.run_test(records, models_to_test)

    # accuracy
    def calculate_accuracy(model_name: str, results: List[Dict[str, Any]], num_records: int) -> Dict[str, Any]:
        """Calculate accuracy metrics for specific digit lengths"""
        total = num_records
        digit_lengths = [2, 4, 6, 8, 10]
        accuracy_metrics = {}

        for digits in digit_lengths:
            correct = 0
            valid_count = 0

            for i, result in enumerate(results):
                predicted =result.get("parsed_prediction", "")
                actual = str(result.get("answer", ""))

                # Only count if both have enough digits
                if len(predicted) >= digits and len(actual) >= digits:
                    valid_count += 1
                    if predicted[:digits] == actual[:digits]:
                        correct += 1

            accuracy_rate = (correct / num_records * 100) if num_records > 0 else 0
            accuracy_metrics[f"{digits}_digit"] = {
                "accuracy_percentage": round(accuracy_rate, 2),
                "correct": correct,
                "valid_count": valid_count
            }

        return {
            "model_name": model_name,
            "total_records": total,
            "digit_accuracies": accuracy_metrics
        }

    # Calculate accuracy for all models
    print("Calculating accuracy metrics...")
    acc_folder = f"{output_folder}/acc"
    os.makedirs(acc_folder, exist_ok=True)
    num_records = len(records)
    for model_name in models_to_test:
        if model_name in test_runner.results:
            model_results = test_runner.results[model_name]
            accuracy_data = calculate_accuracy(model_name, model_results, num_records)

            # Save accuracy results
            acc_file = f"{acc_folder}/{model_name}_accuracy.json"
            with open(acc_file, 'w', encoding='utf-8') as f:
                json.dump(accuracy_data, f, ensure_ascii=False, indent=2)

            # Print accuracy summary
            print(f"Accuracy for {model_name}:")
            for digit_len, metrics in accuracy_data["digit_accuracies"].items():
                print(f"  {digit_len}: {metrics['accuracy_percentage']}% ({metrics['correct']}/{metrics['valid_count']})")
            print(f"Saved to: {acc_file}")

if __name__ == "__main__":
    main()
