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
Context Summarization ToolCalling Agent

一个带有自动上下文摘要功能的 ToolCallingAgent，当上下文 token 数量超过阈值时，
自动执行 context summarization，清空之前的 history 并将 summarization 作为最新的历史。

使用方法:
    # 在 run_widesearch_inference.py 中使用:
    from tools.context_summary_toolcalling_agent import create_context_summarization_agent_class, SummaryStep
    
    # 创建带有 context summarization 功能的 agent 类
    ContextSummarizationAgent = create_context_summarization_agent_class(MemoryManagedToolCallingAgent)
    
    # 使用这个类创建 agent
    main_agent = ContextSummarizationAgent(
        model=main_model,
        tools=main_agent_tools,
        context_token_threshold=80000,  # 80k token 阈值
        ...
    )
"""

import time
from typing import Any, Generator, Type, Optional
from dataclasses import dataclass

from loguru import logger as loguru_logger

from smolagents import ToolCallingAgent, Tool, Model
from smolagents.memory import MemoryStep, TaskStep, ActionStep, PlanningStep, AgentMemory
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import TokenUsage, Timing, LogLevel

# 导入 request_api_detail
try:
    from .my_utils import request_api_detail
except ImportError:
    try:
        from my_utils import request_api_detail
    except ImportError:
        request_api_detail = None


# =============================================================================
# SummaryStep: 用于存储 context summarization 结果的 MemoryStep
# =============================================================================
@dataclass
class SummaryStep(MemoryStep):
    """存储 context summarization 结果的步骤"""
    original_task: str
    summary: str
    summarized_steps_count: int
    tokens_before_summary: int = 0
    
    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        """将 SummaryStep 转换为消息列表"""
        return [
            ChatMessage(
                role=MessageRole.USER,
                content=[{
                    "type": "text",
                    "text": f"""[上下文摘要 - 已压缩 {self.summarized_steps_count} 个历史步骤]

**原始任务:** {self.original_task}

**之前工作的摘要:**
{self.summary}

---
请基于以上摘要继续完成任务。如果摘要中的信息不完整，你可能需要重新搜索或验证相关信息。
"""
                }]
            )
        ]
    
    def dict(self):
        return {
            "original_task": self.original_task,
            "summary": self.summary,
            "summarized_steps_count": self.summarized_steps_count,
            "tokens_before_summary": self.tokens_before_summary,
        }


# =============================================================================
# Default Summarization Prompt (中文版)
# =============================================================================
DEFAULT_SUMMARY_SYSTEM_PROMPT = """你是一个专业的 AI Agent 上下文摘要专家。你的任务是将 Agent 的历史交互记录压缩成结构化的摘要，以便 Agent 能够在有限的上下文窗口中继续高效地完成任务。

## 摘要输出格式要求

你的摘要必须严格按照以下四个部分组织，每个部分用清晰的标题分隔：

### 一、交互历史与计划回顾
总结 Agent 之前的主要交互过程：
- 执行了哪些搜索和网页访问操作
- 调用了哪些工具，得到了什么结果
- 之前制定的计划（Plan）是什么
- 遇到了哪些问题或错误
- **有哪些可以待插入或者更新的数据，但是还没有插入数据库或者更新数据库，请记录他们的信息千万不能遗漏，要尽可能多的记录**

### 二、已收集的表格数据
详细列出 Agent 已经插入或更新到数据库表格中的数据：
- 使用 `add_records` 或 `update_records` 工具添加/更新了哪些记录
- 每条记录的关键字段值是什么
- 如果可能，以表格形式呈现已收集的数据
- **务必保留所有具体的数值、名称、日期等结构化数据，这些是最重要的信息**

### 三、更新后的计划
基于当前进展，给出接下来应该执行的计划：
- 任务完成了多少，还剩多少
- 接下来需要搜索或验证什么信息
- 需要填充表格的哪些字段
- 建议的下一步行动

### 四、其他重要信息
记录其他对继续任务有帮助的信息：
- 重要的发现或结论
- 需要注意的约束条件
- 失败的尝试（避免重复）
- 任何其他上下文信息

## 重要原则

1. **数据优先**：所有具体的数据值（数字、名称、日期、金额等）必须完整保留，不能遗漏或概括
2. **结构清晰**：严格按照四个部分组织输出
3. **简洁高效**：在保留关键信息的前提下，尽量精简描述
4. **可操作性**：摘要应该让 Agent 能够立即继续工作，不需要重新搜索已有的信息
"""

DEFAULT_SUMMARY_USER_TEMPLATE = """请对以下 Agent 的历史交互记录进行摘要。Agent 正在执行一个数据收集任务，需要你生成结构化的摘要以便继续工作。

=== 原始任务 ===
{task}

=== 历史交互记录 ===
{history}

=== 历史记录结束 ===

请严格按照系统提示中规定的四个部分（交互历史与计划回顾、已收集的表格数据、更新后的计划、其他重要信息）生成摘要。

特别注意：
1. 所有已经添加到表格中的数据必须完整列出，不能遗漏
2. 所有具体的数值、名称、日期等必须原样保留
3. 如果有表格数据，请以 Markdown 表格格式呈现
"""


# =============================================================================
# Mixin 类: 添加 Context Summarization 功能
# =============================================================================
class ContextSummarizationMixin:
    """
    Context Summarization Mixin，可以添加到任何 ToolCallingAgent 子类。
    
    这个 Mixin 提供自动 context summarization 功能：
    - 跟踪每次模型调用的 input token 数量
    - 当 token 数量超过阈值时，自动生成摘要并替换历史
    - 使用 qwen3-next-80b 模型进行摘要生成
    
    使用方法:
        class MyAgent(ContextSummarizationMixin, ToolCallingAgent):
            pass
    """
    
    # 需要子类设置的属性
    context_token_threshold: int = 80000
    summary_model_name: str = "qwen3-next-80b-a3b-instruct"
    summary_system_prompt: str = DEFAULT_SUMMARY_SYSTEM_PROMPT
    summary_user_template: str = DEFAULT_SUMMARY_USER_TEMPLATE
    min_steps_before_summary: int = 5
    summary_timeout: float = 180.0
    summary_temperature: float = 0.0
    summary_max_retries: int = 5
    
    # 统计信息
    _summarization_count: int = 0
    _total_tokens_saved: int = 0
    
    def _init_summarization(
        self,
        context_token_threshold: int = 80000,
        summary_model_name: str = "qwen3-next-80b-a3b-instruct",
        summary_system_prompt: str = None,
        summary_user_template: str = None,
        min_steps_before_summary: int = 5,
        summary_timeout: float = 180.0,
        summary_temperature: float = 0.0,
        summary_max_retries: int = 5,
        # 保留旧参数兼容性（已弃用）
        summarization_model: Model = None,
    ):
        """初始化 summarization 相关配置"""
        self.context_token_threshold = context_token_threshold
        self.summary_model_name = summary_model_name
        self.summary_system_prompt = summary_system_prompt or DEFAULT_SUMMARY_SYSTEM_PROMPT
        self.summary_user_template = summary_user_template or DEFAULT_SUMMARY_USER_TEMPLATE
        self.min_steps_before_summary = min_steps_before_summary
        self.summary_timeout = summary_timeout
        self.summary_temperature = summary_temperature
        self.summary_max_retries = summary_max_retries
        
        self._summarization_count = 0
        self._total_tokens_saved = 0
        
        # 检查 request_api_detail 是否可用
        if request_api_detail is None:
            self.logger.log(
                "[ContextSummarization] ⚠️ 严重警告: request_api_detail 不可用! "
                "Context summarization 功能将无法正常工作!",
                level=LogLevel.ERROR
            )
            loguru_logger.error("[ContextSummarization] ⚠️ request_api_detail 不可用! Context summarization 功能将无法正常工作!")
        
        self.logger.log(
            f"[ContextSummarization] Initialized with token threshold: {context_token_threshold:,}, "
            f"summary model: {summary_model_name}, max_retries: {summary_max_retries}",
            level=LogLevel.INFO
        )
        loguru_logger.info(f"[ContextSummarization] 初始化完成 | token阈值: {context_token_threshold:,} | 模型: {summary_model_name} | 最大重试: {summary_max_retries}")
    
    def _get_last_input_tokens(self) -> int:
        """获取最近一次模型调用的 input token 数量"""
        for step in reversed(self.memory.steps):
            if isinstance(step, (ActionStep, PlanningStep)):
                if step.token_usage is not None:
                    return step.token_usage.input_tokens
        return 0
    
    def _count_action_steps(self) -> int:
        """统计当前 memory 中的 ActionStep 数量"""
        return sum(1 for step in self.memory.steps if isinstance(step, ActionStep))
    
    def _should_summarize(self) -> bool:
        """判断是否需要执行 context summarization"""
        last_input_tokens = self._get_last_input_tokens()
        action_step_count = self._count_action_steps()
        
        should_summarize = (
            last_input_tokens > self.context_token_threshold and 
            action_step_count >= self.min_steps_before_summary
        )
        
        loguru_logger.debug(f"[ContextSummarization] 检查是否需要摘要 | input_tokens: {last_input_tokens:,} | threshold: {self.context_token_threshold:,} | steps: {action_step_count} | 需要摘要: {should_summarize}")
        
        if should_summarize:
            self.logger.log(
                f"[ContextSummarization] Triggering: "
                f"input_tokens={last_input_tokens:,} > threshold={self.context_token_threshold:,}, "
                f"steps={action_step_count}",
                level=LogLevel.INFO
            )
            loguru_logger.warning(f"[ContextSummarization] 🚨 触发上下文摘要! | input_tokens: {last_input_tokens:,} > threshold: {self.context_token_threshold:,} | steps: {action_step_count}")
        
        return should_summarize
    
    def _history_to_text(self) -> str:
        """将当前 memory 中的历史转换为文本格式"""
        history_parts = []
        
        for i, step in enumerate(self.memory.steps):
            if isinstance(step, TaskStep):
                history_parts.append(f"[任务] {step.task}")
            
            elif isinstance(step, SummaryStep):
                history_parts.append(f"[之前的摘要] {step.summary}")
            
            elif isinstance(step, PlanningStep):
                history_parts.append(f"[计划] {step.plan}")
            
            elif isinstance(step, ActionStep):
                parts = [f"[步骤 {step.step_number}]"]
                
                if step.model_output:
                    output = str(step.model_output)
                    if len(output) > 2000:
                        output = output[:2000] + "... (已截断)"
                    parts.append(f"Agent 思考: {output}")
                
                if step.tool_calls:
                    for tc in step.tool_calls:
                        args_str = str(tc.arguments)
                        if len(args_str) > 500:
                            args_str = args_str[:500] + "... (已截断)"
                        parts.append(f"工具调用: {tc.name}({args_str})")
                
                if step.observations:
                    obs = step.observations
                    if len(obs) > 3000:
                        obs = obs[:3000] + "... (已截断)"
                    parts.append(f"观察结果: {obs}")
                
                if step.error:
                    parts.append(f"错误: {step.error}")
                
                history_parts.append("\n".join(parts))
        
        return "\n\n---\n\n".join(history_parts)
    
    def _call_summary_api_once(self, prompt: str) -> tuple[Optional[str], Optional[str], Optional[dict]]:
        """
        单次调用 qwen3-next-80b 模型生成摘要
        
        Args:
            prompt: 完整的摘要提示词
            
        Returns:
            (摘要文本, 错误信息, usage信息) - 成功时返回 (摘要, None, usage)，失败时返回 (None, 错误信息, None)
            usage 包含: prompt_tokens, completion_tokens, total_tokens
        """
        if request_api_detail is None:
            loguru_logger.error("[ContextSummarization] request_api_detail 不可用，无法调用摘要 API")
            return None, "request_api_detail 不可用", None
        
        # 构建消息
        messages = [
            {"role": "system", "content": self.summary_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        loguru_logger.info(f"[ContextSummarization] 开始调用摘要 API | 模型: {self.summary_model_name} | prompt长度: {len(prompt)}")
        start_time = time.time()
        
        try:
            # 直接调用 API（request_api_detail 内部已有 retry 机制）
            response, status_code, _ = request_api_detail(
                message=messages,
                temperature=self.summary_temperature,
                sample_num=1,
                index=0,
                model_name=self.summary_model_name,
                retry_num=1,  # 单次调用不重试，由外层 _call_summary_api_with_retry 控制重试
                retry_duration=1,
            )
            
            elapsed_time = time.time() - start_time
            
            if status_code != 200 or response is None:
                error_msg = f"API 返回错误 (status_code={status_code})"
                loguru_logger.error(f"[ContextSummarization] API 调用失败 | status_code: {status_code} | 耗时: {elapsed_time:.2f}s")
                return None, error_msg, None
            
            # 从响应中提取内容和 usage
            summary_content, usage = self._extract_api_response(response)
            
            if not summary_content:
                loguru_logger.error("[ContextSummarization] 无法从 API 响应中提取内容")
                return None, "无法从响应中提取内容", None
            
            # 构建日志信息，包含精确的 token 数量
            completion_tokens = usage.get('completion_tokens', 'N/A') if usage else 'N/A'
            prompt_tokens = usage.get('prompt_tokens', 'N/A') if usage else 'N/A'
            total_tokens = usage.get('total_tokens', 'N/A') if usage else 'N/A'
            
            self.logger.log(
                f"[ContextSummarization] API 调用成功. "
                f"Model: {self.summary_model_name}, "
                f"Elapsed: {elapsed_time:.2f}s, "
                f"Summary: {len(summary_content)} chars / {completion_tokens} tokens",
                level=LogLevel.INFO
            )
            loguru_logger.success(
                f"[ContextSummarization] ✅ API 调用成功 | 模型: {self.summary_model_name} | 耗时: {elapsed_time:.2f}s | "
                f"摘要: {len(summary_content)} chars / {completion_tokens} tokens | "
                f"prompt_tokens: {prompt_tokens} | total_tokens: {total_tokens}"
            )
            
            return summary_content, None, usage
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"API 调用异常: {str(e)}"
            loguru_logger.exception(f"[ContextSummarization] API 调用异常 | 耗时: {elapsed_time:.2f}s | 错误: {e}")
            return None, error_msg, None
    
    def _call_summary_api_with_retry(self, prompt: str) -> tuple[Optional[str], Optional[dict]]:
        """
        使用 retry 机制调用 qwen3-next-80b 模型生成摘要
        
        Args:
            prompt: 完整的摘要提示词
            
        Returns:
            (摘要文本, usage信息) - 如果所有重试都失败则返回 (None, None)
            usage 包含: prompt_tokens, completion_tokens, total_tokens
        """
        last_error = None
        loguru_logger.info(f"[ContextSummarization] 开始生成摘要 (最多重试 {self.summary_max_retries} 次)")
        
        for attempt in range(1, self.summary_max_retries + 1):
            self.logger.log(
                f"[ContextSummarization] 尝试生成摘要 (第 {attempt}/{self.summary_max_retries} 次)...",
                level=LogLevel.INFO
            )
            loguru_logger.info(f"[ContextSummarization] 第 {attempt}/{self.summary_max_retries} 次尝试...")
            
            summary_text, error_msg, usage = self._call_summary_api_once(prompt)
            
            if summary_text:
                if attempt > 1:
                    self.logger.log(
                        f"[ContextSummarization] 第 {attempt} 次尝试成功!",
                        level=LogLevel.INFO
                    )
                    loguru_logger.success(f"[ContextSummarization] 第 {attempt} 次尝试成功!")
                return summary_text, usage
            
            last_error = error_msg
            self.logger.log(
                f"[ContextSummarization] 第 {attempt} 次尝试失败: {error_msg}",
                level=LogLevel.ERROR
            )
            loguru_logger.warning(f"[ContextSummarization] 第 {attempt} 次尝试失败: {error_msg}")
            
            # 如果不是最后一次尝试，等待一段时间后重试
            if attempt < self.summary_max_retries:
                wait_time = min(5 * attempt, 30)  # 递增等待时间，最多30秒
                self.logger.log(
                    f"[ContextSummarization] 等待 {wait_time} 秒后重试...",
                    level=LogLevel.INFO
                )
                loguru_logger.info(f"[ContextSummarization] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        
        self.logger.log(
            f"[ContextSummarization] ❌ 所有 {self.summary_max_retries} 次尝试均失败! "
            f"最后一次错误: {last_error}",
            level=LogLevel.ERROR
        )
        loguru_logger.error(f"[ContextSummarization] ❌ 所有 {self.summary_max_retries} 次尝试均失败! 最后错误: {last_error}")
        return None, None
    
    def _extract_api_response(self, response: dict) -> tuple[Optional[str], Optional[dict]]:
        """
        从 API 响应中提取内容和 usage 信息
        
        Returns:
            (content, usage) - content 是摘要文本，usage 是 token 使用信息字典
            usage 包含: prompt_tokens, completion_tokens, total_tokens
        """
        try:
            content = None
            usage = None
            
            # 提取 content
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"].strip()
            
            # 提取 usage
            if "usage" in response:
                usage = response["usage"]
                loguru_logger.debug(
                    f"[ContextSummarization] Token 使用情况: "
                    f"prompt_tokens={usage.get('prompt_tokens', 'N/A')}, "
                    f"completion_tokens={usage.get('completion_tokens', 'N/A')}, "
                    f"total_tokens={usage.get('total_tokens', 'N/A')}"
                )
            
            if not content:
                self.logger.log(
                    "[ContextSummarization] 无法从响应中提取内容",
                    level=LogLevel.ERROR
                )
                return None, None
            
            return content, usage
            
        except Exception as e:
            self.logger.log(
                f"[ContextSummarization] 提取响应内容时出错: {str(e)}",
                level=LogLevel.ERROR
            )
            loguru_logger.exception(f"[ContextSummarization] 提取响应内容时出错: {e}")
            return None, None
    
    def _generate_context_summary(self) -> tuple[str, Optional[dict]]:
        """
        调用 qwen3-next-80b 模型生成当前 context 的摘要
        
        注意：只使用 qwen3-next-80b 模型，不使用其他模型作为 fallback。
        如果调用失败会自动重试（最多 summary_max_retries 次）。
        
        Returns:
            (摘要文本, usage信息) - usage 包含 prompt_tokens, completion_tokens, total_tokens
        """
        # 检查 request_api_detail 是否可用
        if request_api_detail is None:
            error_msg = (
                "[摘要生成失败: request_api_detail 不可用]\n\n"
                "请确保 my_utils 模块正确安装并且 request_api_detail 函数可用。\n\n"
                f"原始任务: {self.task}"
            )
            self.logger.log(
                "[ContextSummarization] ❌ 无法生成摘要: request_api_detail 不可用!",
                level=LogLevel.ERROR
            )
            loguru_logger.error("[ContextSummarization] ❌ 无法生成摘要: request_api_detail 不可用!")
            return error_msg, None
        
        history_text = self._history_to_text()
        loguru_logger.debug(f"[ContextSummarization] 历史文本长度: {len(history_text)} 字符")
        
        # 构建用户提示词
        user_prompt = self.summary_user_template.format(
            task=self.task or "未知任务",
            history=history_text
        )
        
        self.logger.log(
            f"[ContextSummarization] 正在使用 {self.summary_model_name} 生成上下文摘要...",
            level=LogLevel.INFO
        )
        loguru_logger.info(f"[ContextSummarization] 正在使用 {self.summary_model_name} 生成上下文摘要... (prompt 长度: {len(user_prompt)})")
        
        # 使用带 retry 机制的 API 调用
        summary_text, usage = self._call_summary_api_with_retry(user_prompt)
        
        if summary_text:
            completion_tokens = usage.get('completion_tokens', 'N/A') if usage else 'N/A'
            self.logger.log(
                f"[ContextSummarization] 摘要生成完成, 长度: {len(summary_text)} 字符 / {completion_tokens} tokens",
                level=LogLevel.INFO
            )
            loguru_logger.success(f"[ContextSummarization] 摘要生成完成 | 长度: {len(summary_text)} 字符 / {completion_tokens} tokens")
            return summary_text, usage
        else:
            # 所有重试都失败，返回错误信息
            error_msg = (
                f"[摘要生成失败: 经过 {self.summary_max_retries} 次重试后仍然失败]\n\n"
                f"请检查网络连接和 API 服务状态。\n\n"
                f"原始任务: {self.task}"
            )
            self.logger.log(
                f"[ContextSummarization] ❌ 摘要生成最终失败，已重试 {self.summary_max_retries} 次",
                level=LogLevel.ERROR
            )
            loguru_logger.error(f"[ContextSummarization] ❌ 摘要生成最终失败，已重试 {self.summary_max_retries} 次")
            return error_msg, None
    
    def _perform_context_summarization(self):
        """执行 context summarization"""
        # 记录当前状态
        tokens_before = self._get_last_input_tokens()
        steps_count = len(self.memory.steps)
        original_task = self.task
        
        self.logger.log(
            f"[ContextSummarization] 开始压缩: "
            f"input_tokens={tokens_before:,}, 步骤数={steps_count}",
            level=LogLevel.INFO
        )
        loguru_logger.info(f"[ContextSummarization] 🔄 开始上下文压缩 | input_tokens: {tokens_before:,} | 步骤数: {steps_count}")
        
        # 生成摘要
        summary, usage = self._generate_context_summary()
        
        # 提取精确的 token 信息
        summary_completion_tokens = usage.get('completion_tokens', 0) if usage else 0
        summary_prompt_tokens = usage.get('prompt_tokens', 0) if usage else 0
        summary_total_tokens = usage.get('total_tokens', 0) if usage else 0
        
        # 清空 memory
        self.memory.reset()
        loguru_logger.debug("[ContextSummarization] Memory 已重置")
        
        # 添加 SummaryStep
        summary_step = SummaryStep(
            original_task=original_task,
            summary=summary,
            summarized_steps_count=steps_count,
            tokens_before_summary=tokens_before
        )
        self.memory.steps.append(summary_step)
        
        # 重新添加 TaskStep
        self.memory.steps.append(TaskStep(task=self.task))
        
        # 更新统计
        self._summarization_count += 1
        self._total_tokens_saved += tokens_before
        
        self.logger.log(
            f"[ContextSummarization] ✅ 完成! (#{self._summarization_count}) "
            f"压缩了 {steps_count} 个步骤, "
            f"压缩前 input_tokens: {tokens_before:,}, "
            f"摘要: {len(summary):,} 字符 / {summary_completion_tokens:,} tokens",
            level=LogLevel.INFO
        )
        loguru_logger.success(
            f"[ContextSummarization] ✅ 上下文压缩完成! (第 {self._summarization_count} 次) | "
            f"压缩了 {steps_count} 个步骤 | 压缩前 tokens: {tokens_before:,} | "
            f"摘要: {len(summary):,} 字符 / {summary_completion_tokens:,} tokens | "
            f"摘要 API 使用: prompt={summary_prompt_tokens:,}, completion={summary_completion_tokens:,}, total={summary_total_tokens:,}"
        )
    
    def get_summarization_stats(self) -> dict:
        """获取 summarization 统计信息"""
        return {
            "summarization_count": self._summarization_count,
            "total_tokens_saved": self._total_tokens_saved,
            "current_steps": len(self.memory.steps),
            "context_token_threshold": self.context_token_threshold,
            "summary_model_name": self.summary_model_name,
        }


# =============================================================================
# 工厂函数：创建带有 Context Summarization 功能的 Agent 类
# =============================================================================
def create_context_summarization_agent_class(base_class: Type[ToolCallingAgent]) -> Type:
    """
    创建一个带有 context summarization 功能的 Agent 类。
    
    Args:
        base_class: 基类，通常是 ToolCallingAgent 或其子类（如 MemoryManagedToolCallingAgent）
    
    Returns:
        一个新的类，继承自 base_class 并添加了 context summarization 功能
    
    使用示例:
        # 在 run_widesearch_inference.py 中
        from tools.context_summary_toolcalling_agent import create_context_summarization_agent_class
        
        ContextSummarizationAgent = create_context_summarization_agent_class(MemoryManagedToolCallingAgent)
        
        main_agent = ContextSummarizationAgent(
            model=main_model,
            tools=main_agent_tools,
            context_token_threshold=80000,
            ...
        )
    """
    
    class ContextSummarizationAgent(ContextSummarizationMixin, base_class):
        """
        带有自动 context summarization 的 Agent。
        
        当上下文的 token 数量超过阈值时，自动执行 context summarization，
        清空之前的 history 并将 summarization 作为最新的历史。
        
        Args:
            tools: 工具列表
            model: 模型实例
            context_token_threshold: token 阈值，超过此值触发 summarization（默认 80000）
            summary_model_name: 用于生成摘要的模型名称（默认 qwen3-next-80b-a3b-instruct）
            summary_system_prompt: 摘要系统提示词
            summary_user_template: 摘要用户提示词模板
            min_steps_before_summary: 最少执行多少步后才允许 summarization（默认 5）
            summary_timeout: 摘要 API 调用超时时间（默认 180 秒）
            summary_temperature: 摘要生成温度（默认 0.0）
            summary_max_retries: 摘要生成失败时的最大重试次数（默认 5）
            **kwargs: 传递给父类的其他参数
        """
        
        def __init__(
            self,
            tools: list[Tool],
            model: Model,
            context_token_threshold: int = 80000,
            summary_model_name: str = "qwen3-next-80b-a3b-instruct",
            summary_system_prompt: str = None,
            summary_user_template: str = None,
            min_steps_before_summary: int = 5,
            summary_timeout: float = 180.0,
            summary_temperature: float = 0.0,
            summary_max_retries: int = 5,
            # 保留旧参数兼容性（已弃用）
            summarization_model: Model = None,
            **kwargs
        ):
            # 先调用父类初始化
            super().__init__(tools=tools, model=model, **kwargs)
            
            # 初始化 summarization 功能
            self._init_summarization(
                context_token_threshold=context_token_threshold,
                summary_model_name=summary_model_name,
                summary_system_prompt=summary_system_prompt,
                summary_user_template=summary_user_template,
                min_steps_before_summary=min_steps_before_summary,
                summary_timeout=summary_timeout,
                summary_temperature=summary_temperature,
                summary_max_retries=summary_max_retries,
                summarization_model=summarization_model,
            )
        
        def _step_stream(self, memory_step: ActionStep) -> Generator:
            """重写 _step_stream，在每步开始前检查是否需要 context summarization"""
            # 在每步开始前检查是否需要 summarization
            if self._should_summarize():
                self._perform_context_summarization()
            
            # 调用父类的 _step_stream 继续执行
            yield from super()._step_stream(memory_step)
    
    # 设置类名以便调试
    ContextSummarizationAgent.__name__ = f"ContextSummarization{base_class.__name__}"
    ContextSummarizationAgent.__qualname__ = f"ContextSummarization{base_class.__name__}"
    
    return ContextSummarizationAgent


# =============================================================================
# 便捷类：直接继承 ToolCallingAgent（如果不需要 MemoryManagedToolCallingAgent）
# =============================================================================
class ContextSummarizationToolCallingAgent(ContextSummarizationMixin, ToolCallingAgent):
    """
    直接继承 ToolCallingAgent 的 Context Summarization Agent。
    
    如果你需要继承其他类（如 MemoryManagedToolCallingAgent），
    请使用 create_context_summarization_agent_class() 工厂函数。
    """
    
    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        context_token_threshold: int = 80000,
        summary_model_name: str = "qwen3-next-80b-a3b-instruct",
        summary_system_prompt: str = None,
        summary_user_template: str = None,
        min_steps_before_summary: int = 5,
        summary_timeout: float = 180.0,
        summary_temperature: float = 0.0,
        summary_max_retries: int = 5,
        summarization_model: Model = None,
        **kwargs
    ):
        super().__init__(tools=tools, model=model, **kwargs)
        self._init_summarization(
            context_token_threshold=context_token_threshold,
            summary_model_name=summary_model_name,
            summary_system_prompt=summary_system_prompt,
            summary_user_template=summary_user_template,
            min_steps_before_summary=min_steps_before_summary,
            summary_timeout=summary_timeout,
            summary_temperature=summary_temperature,
            summary_max_retries=summary_max_retries,
            summarization_model=summarization_model,
        )
    
    def _step_stream(self, memory_step: ActionStep) -> Generator:
        if self._should_summarize():
            self._perform_context_summarization()
        yield from super()._step_stream(memory_step)


# =============================================================================
# 测试和示例
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Context Summarization ToolCalling Agent")
    print("=" * 60)
    print()
    print("使用方法:")
    print("  1. 导入工厂函数:")
    print("     from tools.context_summary_toolcalling_agent import \\")
    print("         create_context_summarization_agent_class, SummaryStep")
    print()
    print("  2. 创建 agent 类:")
    print("     ContextSummarizationAgent = create_context_summarization_agent_class(")
    print("         MemoryManagedToolCallingAgent")
    print("     )")
    print()
    print("  3. 创建 agent 实例:")
    print("     agent = ContextSummarizationAgent(")
    print("         model=model,")
    print("         tools=tools,")
    print("         context_token_threshold=80000,")
    print("         summary_model_name='qwen3-next-80b-a3b-instruct',")
    print("         ...") 
    print("     )")
    print()
    print("功能特性:")
    print(f"  - 默认 token 阈值: 80,000")
    print(f"  - 使用 qwen3-next-80b-a3b-instruct 模型生成摘要")
    print(f"  - 自动压缩超长上下文")
    print(f"  - 保留任务和关键数据")
    print(f"  - 跟踪摘要统计信息")
    print()
    print("摘要输出格式:")
    print("  一、交互历史与计划回顾")
    print("  二、已收集的表格数据")
    print("  三、更新后的计划")
    print("  四、其他重要信息")
