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
推理框架运行脚本 - 使用 v3 多条件筛选模式
简化版本，可以直接运行推理
"""

# ============================================================================
# ⚠️ CRITICAL: Load environment variables FIRST, before any other imports!
# ============================================================================
from tools.env_loader import load_dotenv
load_dotenv(override=True)

import argparse
import json
import ipdb
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import traceback

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from smolagents import Model, ToolCallingAgent
#from smolagents import OpenAIServerModel
from patch.openai_sever_model import OpenAIServerModel
from smolagents.memory import AgentMemory
from patch.monitoring import AgentLogger, LogLevel
from rich.console import Console

# 导入工具
tools_path = Path(__file__).parent / "tools"
sys.path.insert(0, str(tools_path))
prompts_path = Path(__file__).parent / "prompts"
sys.path.insert(0, str(prompts_path))

from google_search_tool import GoogleSearchTool, GlobalSearchCounter
from jina_visit import JinaBackedVisitWebpageTool, JinaBackedVisitWebpageSummaryTool, GlobalVisitCounter
from db_table_code_v2 import DBTableCodeToolInterface, GlobalCreateTableCounter
from context_summary_toolcalling_agent import create_context_summarization_agent_class, SummaryStep
from pymongo import MongoClient
import threading

from prompts.widesearch_prompts.main_agent_prompt_v4 import MAIN_AGENT_INSTRUCTIONS, MAIN_AGENT_PROMPT_TEMPLATES
from prompts.widesearch_prompts.tabular_search_prompt_v4 import TABULAR_SEARCH_AGENT_DESCRIPTION, TABULAR_SEARCH_AGENT_PROMPT_TEMPLATES
from prompts.widesearch_prompts.deep_search_prompt_v4 import DEEP_SEARCH_AGENT_DESCRIPTION, DEEP_SEARCH_AGENT_PROMPT_TEMPLATES

console = Console()

# Claude Sonnet 角色转换配置
from smolagents.models import MessageRole
custom_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
    MessageRole.SYSTEM: MessageRole.USER,
}


class GlobalManagedAgentCounter:
    """
    线程安全的全局 managed agent 调用次数计数器。
    
    用于在 main agent 中限制对不同 sub-agent 的调用次数。
    每个 agent 类型（如 tabular_search_agent, deep_search_agent）有独立的限制。
    """
    
    def __init__(self, limits: dict[str, int] = None):
        """
        初始化全局 managed agent 计数器。
        
        Args:
            limits: 每个 agent 的调用次数限制字典，格式为 {agent_name: limit}
                   例如: {"tabular_search_agent": 5, "deep_search_agent": 3}
        """
        self.limits = limits or {}
        self.counts = {agent_name: 0 for agent_name in self.limits.keys()}
        self._lock = threading.Lock()
        # 统计信息：记录每次调用的详细信息
        self.call_history = {agent_name: [] for agent_name in self.limits.keys()}
    
    def try_increment(self, agent_name: str) -> bool:
        """
        尝试增加指定 agent 的调用计数。
        
        Args:
            agent_name: agent 的名称
            
        Returns:
            如果未超过限制，增加计数并返回 True；
            如果已达到限制或 agent 不在限制列表中，返回 False
        """
        with self._lock:
            # 如果该 agent 不在限制列表中，允许调用（无限制）
            if agent_name not in self.limits:
                return True
            
            # 检查是否超过限制
            if self.counts[agent_name] >= self.limits[agent_name]:
                return False
            
            # 增加计数并记录调用
            self.counts[agent_name] += 1
            self.call_history[agent_name].append({
                "call_number": self.counts[agent_name],
                "timestamp": datetime.now().isoformat()
            })
            return True
    
    def get_count(self, agent_name: str) -> int:
        """获取指定 agent 的当前调用次数"""
        with self._lock:
            return self.counts.get(agent_name, 0)
    
    def get_limit(self, agent_name: str) -> int:
        """获取指定 agent 的限制次数"""
        with self._lock:
            return self.limits.get(agent_name, float('inf'))
    
    def get_remaining(self, agent_name: str) -> int:
        """获取指定 agent 的剩余可调用次数"""
        with self._lock:
            if agent_name not in self.limits:
                return float('inf')
            return max(0, self.limits[agent_name] - self.counts[agent_name])
    
    def get_all_status(self) -> dict[str, dict]:
        """
        获取所有 agent 的状态信息。
        
        Returns:
            字典，格式为 {agent_name: {"count": int, "limit": int, "remaining": int}}
        """
        with self._lock:
            return {
                agent_name: {
                    "count": self.counts[agent_name],
                    "limit": self.limits[agent_name],
                    "remaining": max(0, self.limits[agent_name] - self.counts[agent_name])
                }
                for agent_name in self.limits.keys()
            }
    
    def get_statistics(self) -> dict:
        """
        获取统计信息，包括调用历史。
        
        Returns:
            包含所有统计信息的字典
        """
        with self._lock:
            return {
                "limits": self.limits.copy(),
                "counts": self.counts.copy(),
                "call_history": {
                    agent_name: history.copy()
                    for agent_name, history in self.call_history.items()
                }
            }
    
    def reset(self, agent_name: str = None):
        """
        重置计数器。
        
        Args:
            agent_name: 如果指定，只重置该 agent 的计数；否则重置所有
        """
        with self._lock:
            if agent_name:
                if agent_name in self.counts:
                    self.counts[agent_name] = 0
                    self.call_history[agent_name] = []
            else:
                for name in self.counts.keys():
                    self.counts[name] = 0
                    self.call_history[name] = []
    
    def __repr__(self) -> str:
        status = ", ".join([
            f"{name}: {self.counts[name]}/{self.limits[name]}"
            for name in self.limits.keys()
        ])
        return f"GlobalManagedAgentCounter({status})"


class MemoryManagedToolCallingAgent(ToolCallingAgent):
    """ToolCallingAgent with automatic memory management for managed agents.

    This agent automatically resets the memory of managed agents before each call
    to prevent context_length_exceeded errors caused by memory accumulation.
    """

    def _create_agent_copy_for_call(self, original_agent: ToolCallingAgent) -> ToolCallingAgent:
        """为并行调用创建 agent 的独立副本，确保每个调用有独立状态"""
        # 创建新实例，共享不可变对象（model, prompt_templates, tools）
        # 准备构造函数参数
        init_kwargs = {
            'tools': list(original_agent.tools.values()),
            'model': original_agent.model,  # 共享 model（通常线程安全）
            'prompt_templates': original_agent.prompt_templates,
            'max_steps': original_agent.max_steps,
            'planning_interval': original_agent.planning_interval,
            'name': original_agent.name,
            'description': original_agent.description,
            'provide_run_summary': original_agent.provide_run_summary,
            'logger': original_agent.logger,
            'managed_agents': list(original_agent.managed_agents.values()) if hasattr(original_agent, 'managed_agents') and original_agent.managed_agents else None,
        }
        
        # 复制其他可能的属性（如果它们被设置）
        if hasattr(original_agent, 'instructions'):
            init_kwargs['instructions'] = original_agent.instructions
        if hasattr(original_agent, 'final_answer_checks'):
            init_kwargs['final_answer_checks'] = original_agent.final_answer_checks
        if hasattr(original_agent, 'return_full_result'):
            init_kwargs['return_full_result'] = original_agent.return_full_result
        
        # ToolCallingAgent 特有的属性
        if hasattr(original_agent, 'stream_outputs'):
            init_kwargs['stream_outputs'] = original_agent.stream_outputs
        if hasattr(original_agent, 'max_tool_threads'):
            init_kwargs['max_tool_threads'] = original_agent.max_tool_threads
        
        # 🔧 修复：复制 tool_response_retention_budget 参数
        if hasattr(original_agent, 'tool_response_retention_budget'):
            init_kwargs['tool_response_retention_budget'] = original_agent.tool_response_retention_budget
        
        # 🔧 修复：复制 ContextSummarizationAgent 相关参数
        if hasattr(original_agent, 'context_token_threshold'):
            init_kwargs['context_token_threshold'] = original_agent.context_token_threshold
        if hasattr(original_agent, 'summary_model_name'):
            init_kwargs['summary_model_name'] = original_agent.summary_model_name
        if hasattr(original_agent, 'summary_system_prompt'):
            init_kwargs['summary_system_prompt'] = original_agent.summary_system_prompt
        if hasattr(original_agent, 'summary_user_template'):
            init_kwargs['summary_user_template'] = original_agent.summary_user_template
        if hasattr(original_agent, 'min_steps_before_summary'):
            init_kwargs['min_steps_before_summary'] = original_agent.min_steps_before_summary
        if hasattr(original_agent, 'summary_timeout'):
            init_kwargs['summary_timeout'] = original_agent.summary_timeout
        if hasattr(original_agent, 'summary_temperature'):
            init_kwargs['summary_temperature'] = original_agent.summary_temperature
        if hasattr(original_agent, 'summary_max_retries'):
            init_kwargs['summary_max_retries'] = original_agent.summary_max_retries
        
        agent_copy = type(original_agent)(**init_kwargs)
        
        # 确保 memory 是完全独立的（关键！）
        agent_copy.memory = AgentMemory(original_agent.system_prompt)
        agent_copy.step_number = 0
        agent_copy.state = {}
        
        # 🔑 重要：复制 inputs 和 output_type 属性
        # 这些属性是在 _setup_managed_agents 中设置的，用于让 managed agent 可以作为 tool 调用
        # 如果不复制，validate_tool_arguments 会失败（因为它需要 tool.inputs）
        if hasattr(original_agent, 'inputs'):
            agent_copy.inputs = original_agent.inputs.copy() if isinstance(original_agent.inputs, dict) else original_agent.inputs
        if hasattr(original_agent, 'output_type'):
            agent_copy.output_type = original_agent.output_type
        
        # ⚠️ 注意：monitor 会在 __init__ 中自动创建，所以不需要手动复制
        # monitor 使用 self.model 和 self.logger，这些我们已经正确共享了
        
        return agent_copy

    def execute_tool_call(self, tool_name: str, arguments: dict[str, str] | str) -> Any:
        # 检查是否是managed agent
        if tool_name in self.managed_agents:
            # 🔒 检查 managed agent 调用次数限制
            if hasattr(self, 'managed_agent_counter') and self.managed_agent_counter is not None:
                # 尝试增加计数
                if not self.managed_agent_counter.try_increment(tool_name):
                    # 超过限制，生成错误消息
                    current = self.managed_agent_counter.get_count(tool_name)
                    limit = self.managed_agent_counter.get_limit(tool_name)
                    remaining = self.managed_agent_counter.get_remaining(tool_name)
                    
                    # 获取所有 agent 的状态
                    all_status = self.managed_agent_counter.get_all_status()
                    
                    # 构建状态消息
                    status_lines = []
                    available_agents = []
                    for agent_name, status in all_status.items():
                        status_lines.append(
                            f"  - {agent_name}: {status['count']}/{status['limit']} calls used, "
                            f"{status['remaining']} remaining"
                        )
                        if status['remaining'] > 0:
                            available_agents.append(agent_name)
                    
                    status_msg = "\n".join(status_lines)
                    
                    # 构建错误消息
                    error_msg = (
                        f"Error: Managed agent '{tool_name}' call limit reached.\n"
                        f"You have used all {limit} allowed calls to '{tool_name}'.\n\n"
                        f"Current managed agent call status:\n{status_msg}\n\n"
                    )
                    
                    if available_agents:
                        error_msg += (
                            f"You can still call these managed agents: {', '.join(available_agents)}\n"
                            f"Please use them to gather additional information if needed.\n\n"
                        )
                    else:
                        error_msg += (
                            f"All managed agents have reached their call limits.\n"
                            f"You can no longer delegate tasks to sub-agents.\n\n"
                        )
                    
                    error_msg += (
                        f"IMPORTANT: You must now complete the task using:\n"
                        f"1. The information you have already collected\n"
                        f"2. Your own reasoning and analysis capabilities\n"
                        f"3. Any available tools (search, visit_webpage, database operations)\n\n"
                        f"Please proceed to synthesize the information and provide a final answer."
                    )
                    
                    return error_msg
            
            # 🔑 关键修复：创建 agent 的独立副本
            original_agent = self.managed_agents[tool_name]
            agent_copy = self._create_agent_copy_for_call(original_agent)
            
            # ✅ 重要：临时替换 managed_agents 字典中的引用
            # 这样 super().execute_tool_call() 会使用副本而不是原始实例
            # 注意：由于是临时替换，不会影响其他并行调用（Python 字典操作是原子性的）
            original_ref = self.managed_agents[tool_name]
            self.managed_agents[tool_name] = agent_copy
            
            try:
                # ✅ 让 super().execute_tool_call() 处理：
                # - 工具存在性检查
                # - 参数验证（validate_tool_arguments）
                # - 状态变量替换（_substitute_state_variables）
                # - 错误处理和格式化
                # 但使用的是我们的 agent_copy，确保状态独立
                return super().execute_tool_call(tool_name, arguments)
            finally:
                # ✅ 恢复原始引用（确保不影响后续调用）
                self.managed_agents[tool_name] = original_ref
        
        # 普通工具调用，使用原始方法
        return super().execute_tool_call(tool_name, arguments)


def create_model_instance(model_id: str, api_base: str = None, api_key: str = None) -> Model:
    """
    创建模型实例
    
    Args:
        model_id: 模型 ID
        api_base: API 基础 URL
        api_key: API 密钥
    
    Returns:
        模型实例
    """
    # 使用 OpenAI 兼容 API
    if api_base is None:
        api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("api_base")
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("api_key")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return OpenAIServerModel(
        model_id=model_id,
        api_base=api_base,
        api_key=api_key,
        custom_role_conversions=custom_role_conversions,
        tool_choice="auto",
    )


def create_agent_team(
    main_model: Model, 
    tabular_model: Model, 
    deep_model: Model,
    task_work_folder: str, 
    task_id: str, 
    db_name: str, 
    use_summary_tool: bool = False,
    tool_response_retention_budget: Optional[int] = None,
    max_tool_threads: Optional[int] = None,
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
    managed_agent_limits: Optional[dict[str, int]] = None
):
    """
    创建 agent 团队
    
    Args:
        main_model: Main Agent 的模型实例
        tabular_model: Tabular Search Agent 的模型实例
        deep_model: Deep Search Agent 的模型实例
        task_work_folder: 任务工作目录
        task_id: 任务 ID
        db_name: 数据库名称
        use_summary_tool: 是否使用带摘要功能的网页访问工具 (默认: False)
        tool_response_retention_budget: 工具响应保留预算
        max_tool_threads: 最大并行工具调用线程数，用于控制并行工具调用的并发度 (默认: None，使用 ThreadPoolExecutor 默认值)
        global_visit_limit: 全局网页访问次数限制，所有 agent 共享此限制 (默认: None，不限制)
        global_search_limit: 全局搜索次数限制，所有 agent 共享此限制 (默认: None，不限制)
        main_max_steps: Main Agent 的最大步数限制 (默认: 40)
        subagent_max_steps: Sub Agent (Tabular Search Agent 和 Deep Search Agent) 的最大步数限制 (默认: 40)
        main_enable_context_summarization: 是否为 Main Agent 启用 context summarization (默认: False)
        main_context_token_threshold: Main Agent 的 context summarization token 阈值 (默认: 80000)
        tabular_enable_context_summarization: 是否为 Tabular Search Agent 启用 context summarization (默认: False)
        tabular_context_token_threshold: Tabular Search Agent 的 context summarization token 阈值 (默认: 60000)
        deep_enable_context_summarization: 是否为 Deep Search Agent 启用 context summarization (默认: False)
        deep_context_token_threshold: Deep Search Agent 的 context summarization token 阈值 (默认: 60000)
        managed_agent_limits: Managed agent 调用次数限制，格式为 {agent_name: limit}
                            例如: {"tabular_search_agent": 5, "deep_search_agent": 3}
                            (默认: None，不限制)
    """
    # 创建工作目录
    os.makedirs(task_work_folder, exist_ok=True)
    os.makedirs(f'{task_work_folder}/web_page', exist_ok=True)
    
    # 创建日志记录器
    log_file = f'{task_work_folder}/agent_log.txt'
    task_logger = AgentLogger(
        file=log_file,  # 可以直接传入文件路径，AgentLogger 会自动打开
        level=LogLevel.INFO,
        console=False,  # 禁用控制台输出，只输出到文件
    )
    
    # 创建全局 managed agent 计数器（如果提供了限制）
    global_managed_agent_counter = None
    if managed_agent_limits is not None and managed_agent_limits:
        global_managed_agent_counter = GlobalManagedAgentCounter(limits=managed_agent_limits)
        console.print(f"[yellow]🔒 Managed agent call limits:[/yellow]")
        for agent_name, limit in managed_agent_limits.items():
            console.print(f"[yellow]  - {agent_name}: {limit} calls[/yellow]")
    
    # 创建全局 create_table 计数器（所有 agent 共享，限制只能调用 1 次）
    global_create_table_counter = GlobalCreateTableCounter(limit=1)
    console.print(f"[yellow]🔒 Global create_table limit: 1 (shared across all agents)[/yellow]")
    
    # 创建数据库工具
    db_tool = DBTableCodeToolInterface(
        connection_string=os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/"),
        database_name=db_name,
        mode="full",
        task_id=task_id,
        create_table_counter=global_create_table_counter  # 传入共享的全局计数器
    )
    
    # 根据参数选择使用哪个网页访问工具
    if use_summary_tool:
        # 使用带摘要功能的工具
        visit_tool_class = JinaBackedVisitWebpageSummaryTool
        console.print(f"[yellow]📝 Using JinaBackedVisitWebpageSummaryTool (with summary functionality)[/yellow]")
    else:
        # 使用原始工具
        visit_tool_class = JinaBackedVisitWebpageTool
        console.print(f"[yellow]📄 Using JinaBackedVisitWebpageTool (standard version)[/yellow]")
    
    # 创建全局访问计数器（所有 agent 共享）
    global_visit_counter = None
    if global_visit_limit is not None:
        global_visit_counter = GlobalVisitCounter(limit=global_visit_limit)
        console.print(f"[yellow]🔒 Global webpage visit limit: {global_visit_limit} (shared across all agents)[/yellow]")
    
    # 创建全局搜索计数器（所有 agent 共享）
    global_search_counter = None
    if global_search_limit is not None:
        global_search_counter = GlobalSearchCounter(limit=global_search_limit)
        console.print(f"[yellow]🔍 Global search limit: {global_search_limit} (shared across all agents)[/yellow]")
    
    # 创建网页访问工具的辅助函数
    def create_visit_tool():
        """创建网页访问工具实例"""
        tool_kwargs = {
            "jina_keys_file": os.getenv("JINA_KEYS_FILE", None),
            "work_dir": f'{task_work_folder}/web_page',
            "global_visit_counter": global_visit_counter,  # 传入共享的全局计数器
        }
        # 如果使用 summary tool，可以添加额外的配置参数
        if use_summary_tool:
            # 可以从环境变量读取 summary model 配置
            summary_model_name = os.getenv("SUMMARY_MODEL_NAME", "qwen3-next-80b-a3b-instruct")
            summary_timeout = float(os.getenv("SUMMARY_TIMEOUT", "120.0"))
            tool_kwargs.update({
                "summary_model_name": summary_model_name,
                "summary_timeout": summary_timeout
            })
        return visit_tool_class(**tool_kwargs)
    
    # 创建搜索工具的辅助函数
    def create_search_tool():
        """创建搜索工具实例"""
        return GoogleSearchTool(
            limit=10,
            global_search_counter=global_search_counter  # 传入共享的全局计数器
        )
    
    # Tabular Search Agent 工具
    tabular_search_web_tools = [
        create_search_tool(),
        create_visit_tool(),
        db_tool
    ]
    
    # Deep Search Agent 工具
    deep_search_web_tools = [
        create_search_tool(),
        create_visit_tool(),
        db_tool
    ]
    
    # Main Agent 工具
    main_agent_tools = [
        create_search_tool(),
        create_visit_tool(),
        db_tool
    ]
    
    # 创建 ContextSummarizationAgent 类（如果需要的话）
    ContextSummarizationAgent = None
    if tabular_enable_context_summarization or deep_enable_context_summarization or main_enable_context_summarization:
        ContextSummarizationAgent = create_context_summarization_agent_class(MemoryManagedToolCallingAgent)
    
    # 创建 Tabular Search Agent
    if tabular_enable_context_summarization:
        console.print(f"[yellow]🧠 Tabular Agent Context Summarization enabled with threshold: {tabular_context_token_threshold:,} tokens[/yellow]")
        tabular_search_agent = ContextSummarizationAgent(
            model=tabular_model,
            tools=tabular_search_web_tools,
            max_steps=subagent_max_steps,
            planning_interval=12,
            logger=task_logger,
            name="tabular_search_agent",
            description=TABULAR_SEARCH_AGENT_DESCRIPTION,
            provide_run_summary=False,
            tool_response_retention_budget=tool_response_retention_budget,
            max_tool_threads=max_tool_threads,
            context_token_threshold=tabular_context_token_threshold,
        )
    else:
        tabular_search_agent = MemoryManagedToolCallingAgent(
            model=tabular_model,
            tools=tabular_search_web_tools,
            max_steps=subagent_max_steps,
            planning_interval=12,
            logger=task_logger,
            name="tabular_search_agent",
            description=TABULAR_SEARCH_AGENT_DESCRIPTION,
            provide_run_summary=False,
            tool_response_retention_budget=tool_response_retention_budget,
            max_tool_threads=max_tool_threads
        )
    tabular_search_agent.prompt_templates['managed_agent'] = TABULAR_SEARCH_AGENT_PROMPT_TEMPLATES['managed_agent']
    # 设置system_prompt模板，确保agent自己调用工具时使用包含格式说明的prompt
    if 'system_prompt' in TABULAR_SEARCH_AGENT_PROMPT_TEMPLATES:
        tabular_search_agent.prompt_templates['system_prompt'] = TABULAR_SEARCH_AGENT_PROMPT_TEMPLATES['system_prompt']
    
    # 创建 Deep Search Agent
    if deep_enable_context_summarization:
        console.print(f"[yellow]🧠 Deep Agent Context Summarization enabled with threshold: {deep_context_token_threshold:,} tokens[/yellow]")
        deep_search_agent = ContextSummarizationAgent(
            model=deep_model,
            tools=deep_search_web_tools,
            max_steps=subagent_max_steps,
            planning_interval=12,
            logger=task_logger,
            name="deep_search_agent",
            description=DEEP_SEARCH_AGENT_DESCRIPTION,
            provide_run_summary=False,
            tool_response_retention_budget=tool_response_retention_budget,
            max_tool_threads=max_tool_threads,
            context_token_threshold=deep_context_token_threshold,
        )
    else:
        deep_search_agent = MemoryManagedToolCallingAgent(
            model=deep_model,
            tools=deep_search_web_tools,
            max_steps=subagent_max_steps,
            planning_interval=12,
            logger=task_logger,
            name="deep_search_agent",
            description=DEEP_SEARCH_AGENT_DESCRIPTION,
            provide_run_summary=False,
            tool_response_retention_budget=tool_response_retention_budget,
            max_tool_threads=max_tool_threads
        )
    deep_search_agent.prompt_templates['managed_agent'] = DEEP_SEARCH_AGENT_PROMPT_TEMPLATES['managed_agent']
    # 设置system_prompt模板，确保agent自己调用工具时使用包含格式说明的prompt
    if 'system_prompt' in DEEP_SEARCH_AGENT_PROMPT_TEMPLATES:
        deep_search_agent.prompt_templates['system_prompt'] = DEEP_SEARCH_AGENT_PROMPT_TEMPLATES['system_prompt']
    
    # 创建 Main Agent
    # 根据 main_enable_context_summarization 参数选择使用哪个类
    if main_enable_context_summarization:
        # 使用带有 context summarization 功能的 agent
        console.print(f"[yellow]🧠 Main Agent Context Summarization enabled with threshold: {main_context_token_threshold:,} tokens[/yellow]")
        main_agent = ContextSummarizationAgent(
            model=main_model,
            tools=main_agent_tools,
            max_steps=main_max_steps,
            planning_interval=8,
            logger=task_logger,
            managed_agents=[tabular_search_agent, deep_search_agent],
            instructions=MAIN_AGENT_INSTRUCTIONS,
            tool_response_retention_budget=tool_response_retention_budget,
            max_tool_threads=max_tool_threads,
            context_token_threshold=main_context_token_threshold,
            # prompt_templates=MAIN_AGENT_PROMPT_TEMPLATES,
        )
    else:
        # 使用原始的 MemoryManagedToolCallingAgent
        main_agent = MemoryManagedToolCallingAgent(
            model=main_model,
            tools=main_agent_tools,
            max_steps=main_max_steps,
            planning_interval=8,
            logger=task_logger,
            managed_agents=[tabular_search_agent, deep_search_agent],
            instructions=MAIN_AGENT_INSTRUCTIONS,
            tool_response_retention_budget=tool_response_retention_budget,
            max_tool_threads=max_tool_threads
            # prompt_templates=MAIN_AGENT_PROMPT_TEMPLATES,
        )
    main_agent.prompt_templates['managed_agent'] = MAIN_AGENT_PROMPT_TEMPLATES['managed_agent']
    if 'system_prompt' in MAIN_AGENT_PROMPT_TEMPLATES:
        main_agent.prompt_templates['system_prompt'] = MAIN_AGENT_PROMPT_TEMPLATES['system_prompt']
    
    # 🔑 为 main agent 设置 managed_agent_counter 属性
    if global_managed_agent_counter is not None:
        main_agent.managed_agent_counter = global_managed_agent_counter
    
    return main_agent, {}, log_file, task_logger, global_managed_agent_counter


def clear_database_tables(db_name: str, connection_string: str = None):
    """
    清空数据库中的所有表（集合）
    
    Args:
        db_name: 数据库名称
        connection_string: MongoDB 连接字符串，如果为 None 则从环境变量获取
    """
    if connection_string is None:
        connection_string = os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/")
    
    try:
        console.print(f"[yellow]🧹 Clearing all tables in database '{db_name}'...[/yellow]")
        
        # 连接数据库
        client = MongoClient(connection_string)
        db = client[db_name]
        
        # 获取所有集合名称
        collection_names = db.list_collection_names()
        
        if not collection_names:
            console.print(f"[green]✓ Database '{db_name}' is already empty[/green]")
            client.close()
            return
        
        # 删除所有集合（除了系统集合）
        dropped_count = 0
        for collection_name in collection_names:
            # 跳过系统集合（以 system. 开头的集合）
            if not collection_name.startswith("system."):
                try:
                    db[collection_name].drop()
                    dropped_count += 1
                    console.print(f"  [dim]Dropped collection: {collection_name}[/dim]")
                except Exception as e:
                    console.print(f"  [yellow]Warning: Failed to drop collection '{collection_name}': {e}[/yellow]")
        
        client.close()
        
        if dropped_count > 0:
            console.print(f"[green]✓ Successfully cleared {dropped_count} table(s) from database '{db_name}'[/green]")
        else:
            console.print(f"[green]✓ No tables to clear in database '{db_name}'[/green]")
            
    except Exception as e:
        console.print(f"[bold red]❌ Error clearing database tables: {e}[/bold red]")
        console.print(f"[yellow]Warning: Continuing with inference despite cleanup error...[/yellow]")


def run_inference(
    question: str, 
    main_model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
    tabular_model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
    deep_model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
    output_dir: str = "./output", 
    db_name: str = "inference_db", 
    clear_db: bool = True, 
    use_summary_tool: bool = False,
    instance_id: Optional[str] = None,
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
    """
    运行推理
    
    Args:
        question: 要回答的问题
        main_model_id: Main Agent 的模型 ID (默认: us.anthropic.claude-sonnet-4-20250514-v1:0)
        tabular_model_id: Tabular Search Agent 的模型 ID (默认: us.anthropic.claude-sonnet-4-20250514-v1:0)
        deep_model_id: Deep Search Agent 的模型 ID (默认: us.anthropic.claude-sonnet-4-20250514-v1:0)
        output_dir: 输出目录
        db_name: 数据库名称
        clear_db: 是否清空数据库（默认: True）
        use_summary_tool: 是否使用带摘要功能的网页访问工具（默认: False）
        instance_id: 实例 ID，如果提供则使用它作为输出文件名（默认: None，使用 task_id）
        tool_response_retention_budget: 工具响应保留预算
        max_tool_threads: 最大并行工具调用线程数，用于控制并行工具调用的并发度，避免 API rate-limit (默认: None，使用 ThreadPoolExecutor 默认值)
        use_out_key: 是否使用 OUT_API_KEY 和 OUT_BASE_URL（默认: False，使用 OPENAI_API_KEY 和 OPENAI_BASE_URL）
        global_visit_limit: 全局网页访问次数限制，所有 agent 共享此限制 (默认: None，不限制)
        global_search_limit: 全局搜索次数限制，所有 agent 共享此限制 (默认: None，不限制)
        main_max_steps: Main Agent 的最大步数限制 (默认: 40)
        subagent_max_steps: Sub Agent (Tabular Search Agent 和 Deep Search Agent) 的最大步数限制 (默认: 40)
        main_enable_context_summarization: 是否为 Main Agent 启用 context summarization (默认: False)
        main_context_token_threshold: Main Agent 的 context summarization token 阈值 (默认: 80000)
        tabular_enable_context_summarization: 是否为 Tabular Search Agent 启用 context summarization (默认: False)
        tabular_context_token_threshold: Tabular Search Agent 的 context summarization token 阈值 (默认: 60000)
        deep_enable_context_summarization: 是否为 Deep Search Agent 启用 context summarization (默认: False)
        deep_context_token_threshold: Deep Search Agent 的 context summarization token 阈值 (默认: 60000)
        tabular_agent_limit: Tabular Search Agent 的调用次数限制 (默认: None，不限制)
        deep_agent_limit: Deep Search Agent 的调用次数限制 (默认: None，不限制)
    """
    # task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    task_id = instance_id
    
    console.print(f"\n[bold green]{'='*80}[/bold green]")
    console.print(f"[bold green]🚀 Starting inference: {task_id}[/bold green]")
    console.print(f"[bold green]Question: {question[:100]}...[/bold green]")
    console.print(f"[bold green]{'='*80}[/bold green]\n")
    
    # 创建模型实例
    # 根据 use_out_key 参数选择使用哪组 API key 和 base URL
    if use_out_key:
        api_base = os.getenv("OUT_BASE_URL") or os.getenv("api_base")
        api_key = os.getenv("OUT_API_KEY") or os.getenv("api_key")
        key_name = "OUT_API_KEY"
    else:
        api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("api_base")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("api_key")
        key_name = "OPENAI_API_KEY"
    
    if not api_key:
        console.print(f"[bold red]❌ Error: {key_name} not found in environment variables[/bold red]")
        return None
    
    try:
        console.print(f"[{task_id}] Creating model instances...")
        console.print(f"  Main Agent Model: {main_model_id}")
        console.print(f"  Tabular Search Agent Model: {tabular_model_id}")
        console.print(f"  Deep Search Agent Model: {deep_model_id}")
        console.print(f"  Using API Key: {key_name}")
        
        main_model = create_model_instance(main_model_id, api_base, api_key)
        tabular_model = create_model_instance(tabular_model_id, api_base, api_key)
        deep_model = create_model_instance(deep_model_id, api_base, api_key)
        
        console.print(f"[{task_id}] ✓ Model instances created successfully")
    except Exception as e:
        console.print(f"[bold red]❌ Error creating model instances: {e}[/bold red]")
        return None
    
    # 清空数据库中的所有表（如果启用）
    if clear_db:
        clear_database_tables(db_name)
    
    # 创建工作目录
    work_folder = f'{output_dir}/work'
    task_work_folder = f'{work_folder}/{task_id}'
    os.makedirs(task_work_folder, exist_ok=True)
    
    # 如果提供了 instance_id，使用它作为 task_id（用于 collection 命名）
    # 否则使用生成的 task_id
    task_id_for_db = instance_id if instance_id else task_id
    
    console.print(f"[{task_id}] Creating agent team...")
    if use_summary_tool:
        console.print(f"[{task_id}] Using webpage summary tool (with targeted information extraction)")
    task_logger = None
    global_managed_agent_counter = None
    try:
        # 构建 managed_agent_limits 字典
        managed_agent_limits = {}
        if tabular_agent_limit is not None:
            managed_agent_limits["tabular_search_agent"] = tabular_agent_limit
        if deep_agent_limit is not None:
            managed_agent_limits["deep_search_agent"] = deep_agent_limit
        
        agent, mcp_clients, log_file, task_logger, global_managed_agent_counter = create_agent_team(
            main_model=main_model,
            tabular_model=tabular_model,
            deep_model=deep_model,
            task_work_folder=task_work_folder, 
            task_id=task_id_for_db,  # 使用 instance_id 作为 task_id，这样 collection 名称会是 {instance_id}_{collection_name}
            db_name=db_name, 
            use_summary_tool=use_summary_tool,
            tool_response_retention_budget=tool_response_retention_budget,
            max_tool_threads=max_tool_threads,
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
            managed_agent_limits=managed_agent_limits if managed_agent_limits else None
        )
    except Exception as e:
        console.print(f"[bold red][{task_id}] Error creating agent team: {e}[/bold red]")
        traceback.print_exc()
        # 如果创建失败，尝试关闭可能已创建的 logger
        if task_logger:
            try:
                task_logger.close()
            except:
                pass
        return None
    
    console.print(f"[{task_id}] Agent team created successfully")
    
    # 构建问题
    augmented_question = f"""You have a question that needs to be answered. This question requires you to search and find the answer.

Question: {question}
"""
    
    start_time = datetime.now()
    output = None
    
    try:
        console.print(f"[{task_id}] Running agent...")
        output = agent.run(augmented_question)
        console.print(f"[bold green][{task_id}] ✅ Agent completed successfully[/bold green]")
    except Exception as e:
        console.print(f"[bold red][{task_id}] ❌ ERROR: {type(e).__name__}[/bold red]")
        console.print(f"[bold red]Error details: {e}[/bold red]")
        traceback.print_exc()
        output = f"Error: {str(e)}"
    finally:
        console.print(f"[{task_id}] Cleaning up resources...")
        
        # 关闭日志记录器的文件句柄
        if task_logger:
            try:
                task_logger.close()
            except Exception as e:
                console.print(f"[yellow]Warning: Error closing log file: {e}[/yellow]")
        
        # 保存结果
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = {
            "task_id": task_id,
            "question": question,
            "answer": str(output) if output else None,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "main_model_id": main_model_id,
            "tabular_model_id": tabular_model_id,
            "deep_model_id": deep_model_id,
        }
        
        # 添加 managed agent 调用统计信息
        if global_managed_agent_counter is not None:
            result["managed_agent_statistics"] = global_managed_agent_counter.get_statistics()
        
        # 如果提供了 instance_id，使用它作为文件名；否则使用 task_id
        if instance_id:
            # 添加 instance_id 到结果中
            result["instance_id"] = instance_id
            output_file = f'{output_dir}/{instance_id}.json'
        else:
            output_file = f'{output_dir}/result_{task_id}.json'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 只有成功完成的任务才保存文件（不是错误）
        output_str = str(output) if output else ""
        if output and not output_str.startswith("Error:"):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            console.print(f"[bold green]✅ Results saved to: {output_file}[/bold green]")
        else:
            console.print(f"[yellow]⚠️  Task failed or error occurred, not saving result file[/yellow]")
        
        console.print(f"[bold]Answer:[/bold] {output}")
        
        return output


def main():
    parser = argparse.ArgumentParser(description="运行推理框架")
    parser.add_argument("--question", "-q", type=str, required=True, help="要回答的问题")
    parser.add_argument("--tool-response-retention-budget", type=int, default=None, help="工具响应保留预算 (默认: 5)")
    parser.add_argument(
        "--max-tool-threads", 
        type=int, 
        default=None, 
        help="最大并行工具调用线程数，用于控制并行工具调用的并发度，避免 API rate-limit (默认: None，使用 ThreadPoolExecutor 默认值)"
    )
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
    parser.add_argument("--instance-id", type=str, default=None, help="实例 ID，如果提供则使用它作为输出文件名 (默认: None)")
    parser.add_argument("--use-out-key", action="store_true", default=False, help="使用 OUT_API_KEY 和 OUT_BASE_URL 而不是 OPENAI_API_KEY 和 OPENAI_BASE_URL (默认: False)")
    parser.add_argument(
        "--global-visit-limit",
        type=int,
        default=None,
        help="全局网页访问次数限制，所有 agent 共享此限制 (默认: None，不限制)"
    )
    parser.add_argument(
        "--global-search-limit",
        type=int,
        default=None,
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
        default=50,
        help="Tabular Search Agent 的调用次数限制 (默认: None，不限制)"
    )
    parser.add_argument(
        "--deep-agent-limit",
        type=int,
        default=50,
        help="Deep Search Agent 的调用次数限制 (默认: None，不限制)"
    )
    
    args = parser.parse_args()
    
    # 处理便捷参数：如果启用了 enable_all_context_summarization，则为所有 agent 启用
    main_enable_ctx_sum = args.main_enable_context_summarization or args.enable_all_context_summarization
    tabular_enable_ctx_sum = args.tabular_enable_context_summarization or args.enable_all_context_summarization
    deep_enable_ctx_sum = args.deep_enable_context_summarization or args.enable_all_context_summarization
    
    run_inference(
        question=args.question,
        main_model_id=args.main_model_id,
        tabular_model_id=args.tabular_model_id,
        deep_model_id=args.deep_model_id,
        output_dir=args.output_dir,
        db_name=args.db_name,
        clear_db=args.clear_db,
        use_summary_tool=args.use_summary_tool,
        instance_id=args.instance_id,
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
    main()

