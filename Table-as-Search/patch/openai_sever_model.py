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

from smolagents.models import ApiModel
from typing import Any, Generator
from smolagents.models import ChatMessage, ChatMessageStreamDelta, TokenUsage, ChatMessageToolCallStreamDelta, Tool
import logging

logger = logging.getLogger(__name__)


def _clean_blank_text_fields(messages: list[dict]) -> list[dict]:
    """
    Clean blank text fields from messages to prevent API errors.
    
    Handles both simple string content and complex content block formats:
    - Simple: {"role": "user", "content": ""} -> {"role": "user", "content": " "}
    - Complex: {"role": "user", "content": [{"type": "text", "text": ""}]} -> remove or set to " "
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Cleaned list of messages
    """
    cleaned_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned_messages.append(msg)
            continue
        
        msg_copy = msg.copy()
        content = msg_copy.get("content")
        
        if content is None:
            cleaned_messages.append(msg_copy)
            continue
        
        # Handle simple string content
        if isinstance(content, str):
            if content.strip() == "":
                # Replace empty string with a space to avoid blank text field error
                msg_copy["content"] = " "
                logger.debug(f"Replaced empty string content with space in message with role: {msg.get('role')}")
        # Handle list of content blocks (e.g., [{"type": "text", "text": ""}] or [{"text": ""}])
        elif isinstance(content, list):
            cleaned_content = []
            for block in content:
                if isinstance(block, dict):
                    block_copy = block.copy()
                    # Check if block has a "text" field (handles both {"text": ""} and {"type": "text", "text": ""} formats)
                    if "text" in block_copy and isinstance(block_copy["text"], str):
                        if block_copy["text"].strip() == "":
                            # Replace empty text with a space
                            block_copy["text"] = " "
                            logger.debug(f"Replaced empty text in content block with space")
                    cleaned_content.append(block_copy)
                else:
                    cleaned_content.append(block)
            msg_copy["content"] = cleaned_content
        # Handle dict content (e.g., {"text": ""})
        elif isinstance(content, dict):
            content_copy = content.copy()
            if "text" in content_copy and isinstance(content_copy["text"], str):
                if content_copy["text"].strip() == "":
                    content_copy["text"] = " "
                    logger.debug(f"Replaced empty text in content dict with space")
            msg_copy["content"] = content_copy
        
        cleaned_messages.append(msg_copy)
    
    return cleaned_messages

class OpenAIServerModel(ApiModel):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to forward to the underlying OpenAI API completion call, for instance `temperature`.
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ):
        self.client_kwargs = {
            **(client_kwargs or {}),
            "api_key": api_key,
            "base_url": api_base,
            "organization": organization,
            "project": project,
        }
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'smolagents[openai]'`"
            ) from e

        return openai.OpenAI(**self.client_kwargs)

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        import time
        import random
        
        try:
            import openai
        except ImportError:
            openai = None
        
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._apply_rate_limit()
        
        # enable_thinking 是某些自定义 API 的参数，需要通过 extra_body 传递
        extra_body = {}
        enable_thinking = completion_kwargs.pop('enable_thinking', None)
        reasoning_effort = completion_kwargs.pop('reasoning_effort', None)
        thinking_budget = completion_kwargs.pop('thinking_budget', None)
        if enable_thinking is not None:
            extra_body['enable_thinking'] = enable_thinking
        if reasoning_effort is not None:
            extra_body['reasoning_effort'] = reasoning_effort
        if thinking_budget is not None:
            extra_body['thinking_budget'] = thinking_budget
        
        # Clean blank text fields from messages before sending
        if "messages" in completion_kwargs:
            completion_kwargs["messages"] = _clean_blank_text_fields(completion_kwargs["messages"])
        
        # Add retry mechanism for BadRequestError with blank text field
        max_retries = 10
        for attempt in range(max_retries):
            try:
                create_kwargs = {**completion_kwargs, "stream": True, "stream_options": {"include_usage": True}}
                if extra_body:
                    create_kwargs["extra_body"] = extra_body
                stream = self.client.chat.completions.create(**create_kwargs)
                # If we successfully create the stream, break and proceed
                break
            except Exception as e:
                # Check if it's a BadRequestError with blank text field or IndexError
                error_str = str(e)
                is_bad_or_limit_request = (
                    openai is not None and isinstance(e, openai.BadRequestError)
                )
                is_blank_or_limit_error = is_bad_or_limit_request # and "blank" in error_str.lower()
                is_index_error = (
                    isinstance(e, IndexError) or
                    "IndexError" in str(type(e).__name__) or
                    "list index out of range" in error_str.lower() or
                    "list out of range" in error_str.lower()
                )
                is_token_usage_compute_error = "unsupported operand type(s) for +: 'int' and 'NoneType'" in error_str
                
                if (is_blank_or_limit_error or is_index_error or is_token_usage_compute_error) and attempt < max_retries - 1:
                    # Clean messages again before retry (in case they were modified)
                    if "messages" in completion_kwargs:
                        completion_kwargs["messages"] = _clean_blank_text_fields(completion_kwargs["messages"])
                    # Sleep 2-3 seconds before retry
                    sleep_time = random.uniform(10.0, 20.0)
                    if is_blank_or_limit_error:
                        error_type = "blank text field or rate limit"
                    elif is_index_error:
                        error_type = "IndexError"
                    elif is_token_usage_compute_error:
                        error_type = "Token Usage Compute Error"
                    else:
                        error_type = "Unknown Error"
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_retries} due to {error_type} error: {error_str[:200]}. "
                        f"Sleeping {sleep_time:.2f} seconds before retry."
                    )
                    time.sleep(sleep_time)
                    continue
                else:
                    # Re-raise if not a retryable error or max retries reached
                    raise
        
        # Now iterate over the stream
        for event in stream:
            if event.usage:
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    yield ChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=[
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        import time
        import random
        
        try:
            import openai
        except ImportError:
            openai = None
        
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._apply_rate_limit()
        # 某些情况下上游不会设置 stop，这里安全弹出
        completion_kwargs.pop('stop', None)
        
        # enable_thinking 是某些自定义 API 的参数，需要通过 extra_body 传递
        extra_body = {}
        enable_thinking = completion_kwargs.pop('enable_thinking', None)
        reasoning_effort = completion_kwargs.pop('reasoning_effort', None)
        thinking_budget = completion_kwargs.pop('thinking_budget', None)
        if enable_thinking is not None:
            extra_body['enable_thinking'] = enable_thinking
        if reasoning_effort is not None:
            extra_body['reasoning_effort'] = reasoning_effort
        if thinking_budget is not None:
            extra_body['thinking_budget'] = thinking_budget
        
        # Clean blank text fields from messages before sending
        if "messages" in completion_kwargs:
            completion_kwargs["messages"] = _clean_blank_text_fields(completion_kwargs["messages"])
        
        # Add retry mechanism for BadRequestError with blank text field
        max_retries = 10
        for attempt in range(max_retries):
            try:
                create_kwargs = {**completion_kwargs}
                if extra_body:
                    create_kwargs["extra_body"] = extra_body
                response = self.client.chat.completions.create(**create_kwargs)
                return ChatMessage.from_dict(
                    response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
                    raw=response,
                    token_usage=TokenUsage(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                    ),
                )
            except Exception as e:
                # Check if it's a BadRequestError with blank text field or IndexError
                error_str = str(e)
                is_token_usage_compute_error = "unsupported operand type(s) for +: 'int' and 'NoneType'" in error_str

                is_bad_or_limit_request = (
                    openai is not None and isinstance(e, openai.BadRequestError)
                )
                is_blank_or_limit_error = is_bad_or_limit_request # and "blank" in error_str.lower()
                is_index_error = (
                    isinstance(e, IndexError) or
                    "IndexError" in str(type(e).__name__) or
                    "list index out of range" in error_str.lower() or
                    "list out of range" in error_str.lower()
                )
                
                if (is_blank_or_limit_error or is_index_error or is_token_usage_compute_error) and attempt < max_retries - 1:
                    # Clean messages again before retry (in case they were modified)
                    if "messages" in completion_kwargs:
                        completion_kwargs["messages"] = _clean_blank_text_fields(completion_kwargs["messages"])
                    # Sleep 2-3 seconds before retry
                    sleep_time = random.uniform(10.0, 20.0)
                    #error_type = "blank text field" if is_blank_or_limit_error else "IndexError"
                    if is_blank_or_limit_error:
                        error_type = "blank text field or rate limit"
                    elif is_index_error:
                        error_type = "IndexError"
                    elif is_token_usage_compute_error:
                        error_type = "Token Usage Compute Error"
                    else:
                        error_type = "Unknown Error"
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_retries} due to {error_type} error: {error_str[:200]}. "
                        f"Sleeping {sleep_time:.2f} seconds before retry."
                    )
                    time.sleep(sleep_time)
                    continue
                else:
                    # Re-raise if not a retryable error or max retries reached
                    raise
