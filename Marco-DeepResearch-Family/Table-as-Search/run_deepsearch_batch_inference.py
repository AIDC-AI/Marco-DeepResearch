#!/usr/bin/env python3
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

"""
批量推理运行脚本 - 支持并行运行多个 deepsearch 任务
复用 run_deepsearch_inference.py 的 run_inference 函数

注意：使用 multiprocessing 而非 threading 来实现真正的超时控制
因为 Python 的线程无法被强制终止，只有进程可以被 terminate/kill
"""

# ============================================================================
# ⚠️ CRITICAL: Load environment variables FIRST, before any other imports!
# ============================================================================
from tools.env_loader import load_dotenv
load_dotenv(override=True)

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import traceback
import multiprocessing
from multiprocessing import Process, Queue

# 添加当前目录到路径
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

# 导入 run_inference 函数和 clear_database_tables 函数
from run_deepsearch_inference import run_inference, clear_database_tables
from rich.console import Console
from tools.dataloader import *

console = Console()


def _run_single_task_in_process(
    example: Dict,
    main_model_id: str,
    tabular_model_id: str,
    deep_model_id: str,
    output_dir: str,
    db_name: str,
    clear_db: bool,
    use_summary_tool: bool,
    tool_response_retention_budget: Optional[int],
    max_tool_threads: Optional[int],
    use_out_key: bool,
    global_visit_limit: Optional[int],
    global_search_limit: Optional[int],
    main_max_steps: int,
    subagent_max_steps: int,
    main_enable_context_summarization: bool,
    main_context_token_threshold: int,
    tabular_enable_context_summarization: bool,
    tabular_context_token_threshold: int,
    deep_enable_context_summarization: bool,
    deep_context_token_threshold: int,
    tabular_agent_limit: Optional[int],
    deep_agent_limit: Optional[int],
    result_queue: Queue
):
    """
    在子进程中运行单个任务的函数
    
    这个函数会在独立的进程中执行，可以被主进程强制终止
    结果通过 Queue 传递回主进程
    """
    # 在子进程中重新加载环境变量
    load_dotenv(override=True)
    
    instance_id = example["instance_id"]
    task_id = example.get("task_id", instance_id)
    query = example["question"]
    true_answer = example.get('true_answer', '')
    annotator_metadata = example.get('Annotator_Metadata', {})
    
    start_time = datetime.now()
    output = None
    error = None
    run_inference_result = None
    
    process_id = os.getpid()
    print(f"[{instance_id}] 🚀 Starting task in process {process_id} at {start_time.strftime('%H:%M:%S')}")
    
    try:
        output_path = Path(output_dir)
        
        # 调用 run_inference
        output = run_inference(
            question=query,
            true_answer=true_answer,
            annotator_metadata=annotator_metadata,
            main_model_id=main_model_id,
            tabular_model_id=tabular_model_id,
            deep_model_id=deep_model_id,
            output_dir=str(output_path),
            db_name=db_name,
            clear_db=clear_db,
            use_summary_tool=use_summary_tool,
            instance_id=instance_id,
            tool_response_retention_budget=tool_response_retention_budget,
            max_tool_threads=max_tool_threads,
            use_out_key=use_out_key,
            global_visit_limit=global_visit_limit,
            global_search_limit=global_search_limit,
            main_max_steps=main_max_steps,
            subagent_max_steps=subagent_max_steps,
            main_enable_context_summarization=main_enable_context_summarization,
            main_context_token_threshold=main_context_token_threshold,
            tabular_enable_context_summarization=tabular_enable_context_summarization,
            tabular_context_token_threshold=tabular_context_token_threshold,
            deep_enable_context_summarization=deep_enable_context_summarization,
            deep_context_token_threshold=deep_context_token_threshold,
            tabular_agent_limit=tabular_agent_limit,
            deep_agent_limit=deep_agent_limit
        )
        
        # 读取 run_inference 保存的结果文件
        result_file = output_path / f"{instance_id}.json"
        if result_file.exists():
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    run_inference_result = json.load(f)
            except:
                run_inference_result = None
                
    except Exception as e:
        error = str(e)
        print(f"[{instance_id}] ❌ ERROR: {type(e).__name__}: {error}")
        traceback.print_exc()
    
    # 构建结果
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if run_inference_result:
        result = run_inference_result.copy()
        result["query"] = query
        result["error"] = error
        result["timeout"] = False
    else:
        result = {
            "instance_id": instance_id,
            "task_id": task_id,
            "query": query,
            "question": query,
            "answer": str(output) if output else None,
            "response": str(output) if output else None,
            "error": error,
            "timeout": False,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "main_model_id": main_model_id,
            "tabular_model_id": tabular_model_id,
            "deep_model_id": deep_model_id,
        }
    
    # 将结果放入队列
    result_queue.put(result)


def contains_tool_tags(text: str) -> bool:
    """检查文本中是否包含工具标签，说明任务被中断未正常完成"""
    if not text:
        return False
    
    text_str = str(text)
    
    # 检测常见的工具调用标签模式
    tool_patterns = [
        r'```json',       # ```json (代码块开始)
        #r'```python',     # ```python (代码块开始)
        r'```\s*\{',      # ``` { ... } (通用代码块)
        r'Action:\s*',    # Action: ...
        r'Thought:\s*',   # Thought: ...
        r'"name":\s*"[^"]*"',  # JSON 中的工具名称
        r"'name':\s*'[^']*'",  # JSON 中的工具名称
        r"'arguments':\s*\{",  # JSON 中的工具参数
        r'"arguments":\s*\{',  # JSON 中的工具参数
        r'Calling tools' # tool-calling keywords
    ]
    
    for pattern in tool_patterns:
        if re.search(pattern, text_str, re.IGNORECASE | re.MULTILINE):
            return True
    
    return False


def is_task_completed(instance_id: str, output_dir: Path) -> bool:
    """检查任务是否已经完成 - 通过检查 {instance_id}.json 文件是否存在"""
    result_file = output_dir / f"{instance_id}.json"
    if result_file.exists():
        # 检查文件内容是否有效（不是错误）
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                # 检查是否有有效的答案（不是错误）
                answer = result.get("answer") or result.get("response")
                if answer:
                    # 检查是否以 Error: 开头
                    if str(answer).startswith("Error:"):
                        return False
                    # 检查是否包含工具标签（说明任务被中断）
                    if contains_tool_tags(answer):
                        console.print(f"[yellow]⚠️  Task {instance_id} contains tool tags, marking as incomplete[/yellow]")
                        return False
                    return True
        except:
            # 如果文件损坏，认为未完成
            return False
    return False


def run_single_task_with_timeout(
    example: Dict,
    main_model_id: str,
    tabular_model_id: str,
    deep_model_id: str,
    output_dir: Path,
    db_name: str,
    clear_db: bool,
    use_summary_tool: bool,
    tool_response_retention_budget: Optional[int] = None,
    max_tool_threads: Optional[int] = None,
    use_out_key: bool = False,
    global_visit_limit: Optional[int] = None,
    global_search_limit: Optional[int] = None,
    main_max_steps: int = 40,
    subagent_max_steps: int = 40,
    main_enable_context_summarization: bool = False,
    main_context_token_threshold: int = 80000,
    tabular_enable_context_summarization: bool = False,
    tabular_context_token_threshold: int = 60000,
    deep_enable_context_summarization: bool = False,
    deep_context_token_threshold: int = 60000,
    tabular_agent_limit: Optional[int] = None,
    deep_agent_limit: Optional[int] = None,
    timeout_seconds: int = 3600
) -> Dict:
    """运行单个任务，带真正的超时控制（使用 multiprocessing）
    
    关键：使用 multiprocessing.Process 而非 threading
    因为 Python 线程无法被强制终止，但进程可以通过 terminate() 或 kill() 终止
    """
    instance_id = example["instance_id"]
    task_id = example.get("task_id", instance_id)
    query = example["question"]
    
    start_time = datetime.now()
    
    console.print(f"[cyan][{instance_id}] 🚀 Starting task in subprocess at {start_time.strftime('%H:%M:%S')} (timeout: {timeout_seconds}s)[/cyan]")
    
    # 创建用于进程间通信的队列
    result_queue = multiprocessing.Queue()
    
    # 创建子进程
    process = Process(
        target=_run_single_task_in_process,
        args=(
            example,
            main_model_id,
            tabular_model_id,
            deep_model_id,
            str(output_dir),
            db_name,
            clear_db,
            use_summary_tool,
            tool_response_retention_budget,
            max_tool_threads,
            use_out_key,
            global_visit_limit,
            global_search_limit,
            main_max_steps,
            subagent_max_steps,
            main_enable_context_summarization,
            main_context_token_threshold,
            tabular_enable_context_summarization,
            tabular_context_token_threshold,
            deep_enable_context_summarization,
            deep_context_token_threshold,
            tabular_agent_limit,
            deep_agent_limit,
            result_queue
        )
    )
    
    # 启动子进程
    process.start()
    
    # 等待子进程完成，带超时
    process.join(timeout=timeout_seconds)
    
    # 检查进程是否仍在运行（即超时了）
    if process.is_alive():
        console.print(f"[bold red][{instance_id}] ⏰ TIMEOUT after {timeout_seconds}s - Terminating process...[/bold red]")
        
        # 先尝试优雅终止
        process.terminate()
        # 等待一小段时间让进程清理
        process.join(timeout=10)
        
        # 如果还在运行，强制杀死
        if process.is_alive():
            console.print(f"[bold red][{instance_id}] Process did not terminate, forcing kill...[/bold red]")
            process.kill()
            process.join(timeout=5)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 返回超时结果
        return {
            "instance_id": instance_id,
            "task_id": task_id,
            "query": query,
            "question": query,
            "answer": None,
            "response": None,
            "error": f"Task timeout after {timeout_seconds} seconds (actual duration: {duration:.1f}s)",
            "timeout": True,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "main_model_id": main_model_id,
            "tabular_model_id": tabular_model_id,
            "deep_model_id": deep_model_id,
        }
    
    # 进程正常完成，从队列获取结果
    try:
        # 使用非阻塞方式获取结果，以防队列为空
        if not result_queue.empty():
            result = result_queue.get(timeout=5)
            console.print(f"[bold green][{instance_id}] ✅ Task completed successfully[/bold green]")
            return result
        else:
            # 队列为空，可能是进程异常退出
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            console.print(f"[bold yellow][{instance_id}] ⚠️ Process completed but no result in queue[/bold yellow]")
            
            # 尝试读取结果文件
            result_file = output_dir / f"{instance_id}.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    result["query"] = query
                    result["timeout"] = False
                    return result
                except:
                    pass
            
            return {
                "instance_id": instance_id,
                "task_id": task_id,
                "query": query,
                "question": query,
                "answer": None,
                "response": None,
                "error": "Process completed but returned no result",
                "timeout": False,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "main_model_id": main_model_id,
                "tabular_model_id": tabular_model_id,
                "deep_model_id": deep_model_id,
            }
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        console.print(f"[bold red][{instance_id}] ❌ Error getting result from queue: {e}[/bold red]")
        return {
            "instance_id": instance_id,
            "task_id": task_id,
            "query": query,
            "question": query,
            "answer": None,
            "response": None,
            "error": f"Error getting result: {str(e)}",
            "timeout": False,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "main_model_id": main_model_id,
            "tabular_model_id": tabular_model_id,
            "deep_model_id": deep_model_id,
        }


def run_deepsearch_batch(
    input_file: str,
    main_model_id: str,
    tabular_model_id: str,
    deep_model_id: str,
    output_dir: str,
    db_name: str,
    clear_db: bool,
    use_summary_tool: bool,
    max_workers: int = 4,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    skip_completed: bool = True,
    timeout_seconds: int = 3600,
    tool_response_retention_budget: Optional[int] = None,
    max_tool_threads: Optional[int] = None,
    use_out_key: bool = False,
    global_visit_limit: Optional[int] = None,
    global_search_limit: Optional[int] = None,
    main_max_steps: int = 40,
    subagent_max_steps: int = 40,
    main_enable_context_summarization: bool = False,
    main_context_token_threshold: int = 80000,
    tabular_enable_context_summarization: bool = False,
    tabular_context_token_threshold: int = 60000,
    deep_enable_context_summarization: bool = False,
    deep_context_token_threshold: int = 60000,
    tabular_agent_limit: Optional[int] = None,
    deep_agent_limit: Optional[int] = None
):
    """运行 deepsearch 批量推理
    
    使用 multiprocessing 实现真正的超时控制：
    - 每个任务在独立的子进程中运行
    - 超时后可以强制终止进程
    - 使用线程池来调度多个进程，实现并行执行
    """
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold cyan]🚀 Starting deepsearch batch inference (with multiprocessing timeout)[/bold cyan]")
    console.print(f"[bold cyan]Input file: {input_file}[/bold cyan]")
    console.print(f"[bold cyan]Timeout per task: {timeout_seconds}s[/bold cyan]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")
    
    # 加载数据集
    if "gaia" in input_file:
        examples = load_gaia_dataset(input_file)
    elif "browsecomp-en" in input_file:
        examples = load_browsecomp_en_dataset(input_file)
    elif "browsecomp-zh" in input_file:
        examples = load_browsecomp_zh_dataset(input_file)
    elif "hle" in input_file.lower():
        examples = load_hle_dataset(input_file)
    else:
        console.print(f"[bold red]❌ Unknown benchmark: {input_file}[/bold red]")
        return
    
    if not examples:
        console.print(f"[bold red]❌ No examples loaded from {input_file}[/bold red]")
        return
    
    # 切片处理
    if end_idx is None:
        end_idx = len(examples)
    examples = examples[start_idx:end_idx]
    
    console.print(f"[bold]Loaded {len(examples)} examples[/bold]")
    
    # 创建工作目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 跳过已完成的任务
    if skip_completed:
        original_count = len(examples)
        examples = [ex for ex in examples if not is_task_completed(ex["instance_id"], output_path)]
        skipped_count = original_count - len(examples)
        if skipped_count > 0:
            console.print(f"[yellow]⏭️  Skipped {skipped_count} already completed tasks[/yellow]")
        console.print(f"[bold]Remaining tasks: {len(examples)}[/bold]")
    
    if not examples:
        console.print(f"[bold green]✅ All tasks are already completed![/bold green]")
        return
    
    # 如果 clear_db=True，在批量执行开始前只清空一次数据库
    if clear_db:
        console.print(f"[bold yellow]🧹 Clearing database '{db_name}' before batch execution...[/bold yellow]")
        clear_database_tables(db_name)
        console.print(f"[bold green]✓ Database cleared. All tasks will use clear_db=False[/bold green]\n")
    
    # 处理任务
    results = []
    task_clear_db = False  # 批量执行时，每个任务都不再清空数据库（已在开始前清空）
    
    console.print(f"[bold cyan]🔧 Using {max_workers} parallel workers[/bold cyan]")
    console.print(f"[bold cyan]💡 Each task runs in a separate process with enforced timeout[/bold cyan]\n")
    
    # 使用线程池来并行调度多个任务（每个任务在独立进程中执行）
    # 这里线程只负责启动和监控进程，真正的任务执行在子进程中
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(
                run_single_task_with_timeout,
                example,
                main_model_id,
                tabular_model_id,
                deep_model_id,
                output_path,
                db_name,
                task_clear_db,
                use_summary_tool,
                tool_response_retention_budget,
                max_tool_threads,
                use_out_key,
                global_visit_limit,
                global_search_limit,
                main_max_steps,
                subagent_max_steps,
                main_enable_context_summarization,
                main_context_token_threshold,
                tabular_enable_context_summarization,
                tabular_context_token_threshold,
                deep_enable_context_summarization,
                deep_context_token_threshold,
                tabular_agent_limit,
                deep_agent_limit,
                timeout_seconds  # 传递超时参数
            ): example for example in examples
        }
        
        console.print(f"[bold green]✅ Submitted {len(futures)} tasks[/bold green]\n")
        
        # 使用 rich Progress 替代 tqdm
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing deepsearch tasks", total=len(examples))
            
            for future in as_completed(futures):
                example = futures[future]
                try:
                    # 这里不需要再设置 timeout，因为 run_single_task_with_timeout
                    # 内部使用 multiprocessing 已经实现了真正的超时控制
                    # 但我们加一个额外的缓冲时间以防万一
                    result = future.result(timeout=timeout_seconds + 120)
                    results.append(result)
                except Exception as e:
                    console.print(f"[bold red]❌ Error processing {example['instance_id']}: {e}[/bold red]")
                    traceback.print_exc()
                    results.append({
                        "instance_id": example["instance_id"],
                        "task_id": example.get("task_id", example["instance_id"]),
                        "query": example.get("question", ""),
                        "error": str(e),
                        "timeout": False,
                        "start_time": datetime.now().isoformat(),
                    })
                finally:
                    progress.update(task, advance=1)
    
    # 统计结果
    completed = sum(1 for r in results if (r.get("response") or r.get("answer")) and not r.get("error"))
    timeout_count = sum(1 for r in results if r.get("timeout"))
    error_count = sum(1 for r in results if r.get("error") and not r.get("timeout"))
    
    console.print(f"\n[bold green]✅ Completed deepsearch batch: {len(results)}/{len(examples)} tasks[/bold green]")
    console.print(f"[bold]  - Successfully completed: {completed}[/bold]")
    console.print(f"[bold]  - Timeout: {timeout_count}[/bold]")
    console.print(f"[bold]  - Errors: {error_count}[/bold]")
    console.print(f"[bold]Results directory: {output_path}[/bold]")


def main():
    parser = argparse.ArgumentParser(description="批量运行 deepsearch 推理框架")
    
    # 复用 run_deepsearch_inference.py 的命令行参数
    parser.add_argument(
        "--main-model-id", 
        type=str, 
        default="us.anthropic.claude-sonnet-4-20250514-v1:0", 
        help="Main Agent 的模型 ID (默认: us.anthropic.claude-sonnet-4-20250514-v1:0)"
    )
    parser.add_argument(
        "--tabular-model-id", 
        type=str, 
        default="us.anthropic.claude-sonnet-4-20250514-v1:0", 
        help="Tabular Search Agent 的模型 ID (默认: us.anthropic.claude-sonnet-4-20250514-v1:0)"
    )
    parser.add_argument(
        "--deep-model-id", 
        type=str, 
        default="us.anthropic.claude-sonnet-4-20250514-v1:0", 
        help="Deep Search Agent 的模型 ID (默认: us.anthropic.claude-sonnet-4-20250514-v1:0)"
    )
    parser.add_argument("--output-dir", "-o", type=str, default="./output", help="输出目录 (默认: ./output)")
    parser.add_argument("--db-name", "-d", type=str, default="inference_db", help="数据库名称 (默认: inference_db)")
    parser.add_argument("--clear-db", action="store_true", default=True, help="清空数据库中的所有表 (默认: True)")
    parser.add_argument("--no-clear-db", dest="clear_db", action="store_false", help="不清空数据库")
    parser.add_argument("--use-summary-tool", action="store_true", default=False, help="使用带摘要功能的网页访问工具 (JinaBackedVisitWebpageSummaryTool)")
    parser.add_argument("--use-out-key", action="store_true", default=False, help="使用 OUT_API_KEY 和 OUT_BASE_URL 而不是 OPENAI_API_KEY 和 OPENAI_BASE_URL (默认: False)")
    
    # 新增批量处理参数
    parser.add_argument("--input-file", "-i", type=str, 
                       default=str(BASE_DIR / "benchmark" / "gaia-text-only.jsonl"),
                       help="输入文件路径 (默认: benchmark/gaia-text-only.jsonl)")
    parser.add_argument("--max-workers", "-w", type=int, default=4,
                       help="最大并发数 (默认: 4)")
    parser.add_argument("--start-idx", type=int, default=0,
                       help="开始索引 (默认: 0)")
    parser.add_argument("--end-idx", type=int, default=None,
                       help="结束索引 (默认: None, 处理所有)")
    parser.add_argument("--tool-response-retention-budget", type=int, default=None, help="工具响应保留预算 (默认: 5)")
    parser.add_argument(
        "--max-tool-threads", 
        type=int, 
        default=4, 
        help="最大并行工具调用线程数，用于控制并行工具调用的并发度，避免 API rate-limit (默认: None，使用 ThreadPoolExecutor 默认值)"
    )
    parser.add_argument("--skip-completed", action="store_true", default=True,
                       help="跳过已完成的任务 (默认: True)")
    parser.add_argument("--no-skip-completed", dest="skip_completed", action="store_false",
                       help="不跳过已完成的任务")
    parser.add_argument("--timeout-seconds", type=int, default=3600,
                       help="任务超时时间（秒）(默认: 3600, 即1小时)")
    parser.add_argument(
        "--global-visit-limit",
        type=int,
        default=100,
        help="全局网页访问次数限制，所有 agent 共享此限制 (默认: None，不限制)"
    )
    parser.add_argument(
        "--global-search-limit",
        type=int,
        default=100,
        help="全局搜索次数限制，所有 agent 共享此限制 (默认: None，不限制)"
    )
    parser.add_argument(
        "--main-max-step",
        type=int,
        default=40,
        help="Main Agent 的最大步数限制 (默认: 40)"
    )
    parser.add_argument(
        "--subagent-max-step",
        type=int,
        default=40,
        help="Sub Agent (Tabular Search Agent 和 Deep Search Agent) 的最大步数限制 (默认: 40)"
    )
    
    # Main Agent Context Summarization 参数
    parser.add_argument(
        "--main-enable-context-summarization",
        action="store_true",
        default=False,
        help="为 Main Agent 启用 context summarization 功能 (默认: False)"
    )
    parser.add_argument(
        "--main-context-token-threshold",
        type=int,
        default=80000,
        help="Main Agent 的 context summarization token 阈值 (默认: 80000)"
    )
    # Tabular Search Agent Context Summarization 参数
    parser.add_argument(
        "--tabular-enable-context-summarization",
        action="store_true",
        default=False,
        help="为 Tabular Search Agent 启用 context summarization 功能 (默认: False)"
    )
    parser.add_argument(
        "--tabular-context-token-threshold",
        type=int,
        default=60000,
        help="Tabular Search Agent 的 context summarization token 阈值 (默认: 60000)"
    )
    # Deep Search Agent Context Summarization 参数
    parser.add_argument(
        "--deep-enable-context-summarization",
        action="store_true",
        default=False,
        help="为 Deep Search Agent 启用 context summarization 功能 (默认: False)"
    )
    parser.add_argument(
        "--deep-context-token-threshold",
        type=int,
        default=60000,
        help="Deep Search Agent 的 context summarization token 阈值 (默认: 60000)"
    )
    # 便捷参数：同时为所有 agent 启用 context summarization
    parser.add_argument(
        "--enable-all-context-summarization",
        action="store_true",
        default=False,
        help="为所有 agent (Main/Tabular/Deep) 同时启用 context summarization 功能 (默认: False)"
    )
    # Managed agent 调用次数限制参数
    parser.add_argument(
        "--tabular-agent-limit",
        type=int,
        default=None,
        help="Tabular Search Agent 的调用次数限制 (默认: None，不限制)"
    )
    parser.add_argument(
        "--deep-agent-limit",
        type=int,
        default=None,
        help="Deep Search Agent 的调用次数限制 (默认: None，不限制)"
    )
    
    args = parser.parse_args()
    
    # 设置环境变量（如果需要）
    if not os.getenv("OPENAI_BASE_URL"):
        console.print("[yellow]⚠️  Warning: OPENAI_BASE_URL not set in environment[/yellow]")
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[yellow]⚠️  Warning: OPENAI_API_KEY not set in environment[/yellow]")
    
    # 处理 enable-all-context-summarization 便捷参数
    main_enable_ctx_sum = args.main_enable_context_summarization or args.enable_all_context_summarization
    tabular_enable_ctx_sum = args.tabular_enable_context_summarization or args.enable_all_context_summarization
    deep_enable_ctx_sum = args.deep_enable_context_summarization or args.enable_all_context_summarization
    
    # 运行批量推理
    run_deepsearch_batch(
        input_file=args.input_file,
        main_model_id=args.main_model_id,
        tabular_model_id=args.tabular_model_id,
        deep_model_id=args.deep_model_id,
        output_dir=args.output_dir,
        db_name=args.db_name,
        clear_db=args.clear_db,
        use_summary_tool=args.use_summary_tool,
        max_workers=args.max_workers,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        skip_completed=args.skip_completed,
        timeout_seconds=args.timeout_seconds,
        tool_response_retention_budget=args.tool_response_retention_budget,
        max_tool_threads=args.max_tool_threads,
        use_out_key=args.use_out_key,
        global_visit_limit=args.global_visit_limit,
        global_search_limit=args.global_search_limit,
        main_max_steps=args.main_max_step,
        subagent_max_steps=args.subagent_max_step,
        main_enable_context_summarization=main_enable_ctx_sum,
        main_context_token_threshold=args.main_context_token_threshold,
        tabular_enable_context_summarization=tabular_enable_ctx_sum,
        tabular_context_token_threshold=args.tabular_context_token_threshold,
        deep_enable_context_summarization=deep_enable_ctx_sum,
        deep_context_token_threshold=args.deep_context_token_threshold,
        tabular_agent_limit=args.tabular_agent_limit,
        deep_agent_limit=args.deep_agent_limit
    )


if __name__ == "__main__":
    # 设置 multiprocessing 启动方法
    # 在 macOS 上使用 'spawn' (Python 3.8+ 默认)
    # 在 Linux 上可以使用 'fork' 但 'spawn' 更安全
    # 这确保子进程有干净的状态，避免潜在的问题
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        # 如果已经设置过，忽略
        pass
    
    main()
