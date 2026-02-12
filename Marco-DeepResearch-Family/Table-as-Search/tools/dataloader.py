# Copyright (C) 2026 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tools.env_loader import load_dotenv
load_dotenv(override=True)

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List, Dict
import traceback
from rich.console import Console

console = Console()


def load_gaia_dataset(input_file: str) -> List[Dict]:
    """加载 GAIA 数据集"""
    examples = []
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[bold red]❌ Error: File not found: {input_file}[/bold red]")
        return examples
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f.readlines()):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                examples.append({
                    "task_id": item.get("id", line_num),
                    "instance_id": item.get("id", line_num),
                    "question": item.get("question", ""),
                    "true_answer": item.get("answer", ""),
                    "level": item.get("Level", "Unknown"),
                    "id": item.get("id", line_num),
                    "Annotator_Metadata": item.get("Annotator_Metadata", ""),
                })
            except json.JSONDecodeError as e:
                console.print(f"[yellow]⚠️  Warning: Skip line {line_num} (JSON error): {e}[/yellow]")
                continue
    
    return examples

def load_browsecomp_zh_dataset(input_file: str) -> List[Dict]:
    """加载 BrowseComp-ZH 数据集"""
    examples = []
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[bold red]❌ Error: File not found: {input_file}[/bold red]")
        return examples
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 处理不同的数据格式
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and "data" in data:
                items = data["data"]
            else:
                items = [data]
            
            for idx, item in enumerate(items, 1):
                # BrowseComp-ZH 使用 "Question" 和 "Answer" (大写)
                question = item.get("Question", item.get("question", item.get("problem", "")))
                answer = item.get("Answer", item.get("answer", ""))
                examples.append({
                    "task_id": f"bc_zh_{idx:04d}",
                    "instance_id": f"bc_zh_{idx:04d}",
                    "question": question,
                    "true_answer": answer,
                    "Annotator_Metadata": None,
                    "id": idx,
                })
    except Exception as e:
        console.print(f"[bold red]❌ Error loading BrowseComp-ZH dataset: {e}[/bold red]")
        traceback.print_exc()
    
    return examples


def load_browsecomp_en_dataset(input_file: str) -> List[Dict]:
    """加载 BrowseComp-EN 数据集（JSONL 格式）"""
    examples = []
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[bold red]❌ Error: File not found: {input_file}[/bold red]")
        return examples
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                # BrowseComp-EN 使用 "problem" 字段作为问题，"answer" 作为答案
                problem = item.get("problem", "")
                answer = item.get("answer", "")
                problem_topic = item.get("problem_topic", "")
                
                examples.append({
                    "task_id": line_num,
                    "instance_id": line_num,
                    "id": line_num,
                    "question": problem,
                    "true_answer": answer,
                    "problem_topic": problem_topic,
                    "Annotator_Metadata": None,
                })
            except json.JSONDecodeError as e:
                console.print(f"[yellow]⚠️  Warning: Skip line {line_num} (JSON error): {e}[/yellow]")
                continue
    
    return examples


def load_widesearch_dataset(input_file: str) -> List[Dict]:
    """加载 widesearch 数据集"""
    examples = []
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[bold red]❌ Error: File not found: {input_file}[/bold red]")
        return examples
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                instance_id = item.get("instance_id", f"ws_{line_num:04d}")
                query = item.get("query", "")
                
                if not query:
                    console.print(f"[yellow]⚠️  Warning: Skip line {line_num} (empty query)[/yellow]")
                    continue
                
                examples.append({
                    "instance_id": instance_id,
                    "task_id": instance_id,  # 使用 instance_id 作为 task_id
                    "query": query,
                    "evaluation": item.get("evaluation", ""),
                    "language": item.get("language", "en"),
                    "id": line_num,
                })
            except json.JSONDecodeError as e:
                console.print(f"[yellow]⚠️  Warning: Skip line {line_num} (JSON error): {e}[/yellow]")
                continue
    
    return examples


def load_hle_dataset(input_file: str) -> List[Dict]:
    """加载 HLE 数据集（JSONL 格式）"""
    examples = []
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[bold red]❌ Error: File not found: {input_file}[/bold red]")
        return examples
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                # HLE 数据集使用 "query" 字段作为问题，"answer" 作为答案
                query = item.get("query", "")
                answer = item.get("answer", "")
                item_id = item.get("id", item.get("index", f"hle_{line_num:04d}"))
                
                if not query:
                    console.print(f"[yellow]⚠️  Warning: Skip line {line_num} (empty query)[/yellow]")
                    continue
                
                # 构建 Annotator_Metadata，包含 HLE 特有的字段
                annotator_metadata = {
                    "category": item.get("category", ""),
                    "raw_subject": item.get("raw_subject", ""),
                    "rationale": item.get("rationale", ""),
                    "classification": item.get("classification", {}),
                    "index": item.get("index", line_num),
                }
                
                examples.append({
                    "task_id": item_id,
                    "instance_id": item_id,
                    "id": item_id,
                    "question": query,
                    "true_answer": answer,
                    "category": item.get("category", ""),
                    "raw_subject": item.get("raw_subject", ""),
                    "rationale": item.get("rationale", ""),
                    "Annotator_Metadata": annotator_metadata,
                })
            except json.JSONDecodeError as e:
                console.print(f"[yellow]⚠️  Warning: Skip line {line_num} (JSON error): {e}[/yellow]")
                continue
    
    return examples