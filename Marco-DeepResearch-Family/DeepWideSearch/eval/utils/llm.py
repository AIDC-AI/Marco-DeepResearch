# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Any, Iterable, List, Optional, Union
from dotenv import load_dotenv
import os
from loguru import logger
from openai import OpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from tenacity import retry, stop_after_attempt, wait_incrementing

from eval.utils.schema import LLMOutputItem, ModelResponse, ToolCall
from my_utils import request_api
import ipdb
from dataclasses import asdict, dataclass



@retry(stop=stop_after_attempt(8), wait=wait_incrementing(8, 8))
def openai_complete(
    base_url: str,
    api_key: Optional[str],
    messages: Iterable[dict],
    tools: Optional[Iterable[dict]] = None,
    model_name: str = "gpt-4o-2024-05-13",
    retry_if_empty: bool = False,
    **generate_kwargs,
) -> Optional[ChatCompletionMessage]:
    """Complete a prompt with OpenAI APIs."""

    def create_openai_client(base_url, api_key):
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=300,
        )

    openai_client = create_openai_client(base_url, api_key)
    logger.debug(f"messages: {messages}")
    logger.debug(f"tools: {tools}")
    logger.debug(generate_kwargs)
    completion = openai_client.chat.completions.create(
        messages=messages,  # type: ignore
        model=model_name,
        tools=tools,  # type: ignore
        **generate_kwargs,
    )
    message = None

    try:
        message = completion.choices[0].message
    except Exception as e:
        logger.warning(f"Error during completion: {e}")
        return None

    if retry_if_empty and not message.content and not message.tool_calls:
        raise RuntimeError(
            "[openai_complete] Got message, but content and toolcalls is empty, retry"
        )

    return message


@dataclass
class APIResponse:
    content: str

def llm_completion(
    messages: Union[str, List[dict]],
    tools: Optional[List[dict]] = None,
    model_config_name: str = "gpt-4o",
) -> Optional[ChatCompletionMessage]:
    """Complete a prompt with given LLM, raise error if the request failed."""

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    load_dotenv(override=True)
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("api_base")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("api_key")
    
    response = openai_complete(
        base_url=base_url,
        api_key=api_key,
        messages=messages,
        tools=tools,
        model_name=model_config_name,
        retry_if_empty=True
    )
    response = APIResponse(content=response.content)
    return response


def transform_model_response(response: Any | None) -> ModelResponse:
    out = ModelResponse()
    if response is None:
        out.error_marker = {"message": "Calling LLM failed."}
        return out

    # Set fields.
    item = LLMOutputItem(content=response.content)
    # Convert into dict to get optional fields.
    resp_dict = response.model_dump()
    if resp_dict.get("reasoning_content"):
        item.reasoning_content = resp_dict["reasoning_content"]
    if resp_dict.get("signature"):
        item.signature = resp_dict["signature"]

    if response.tool_calls:
        item.tool_calls = []
        for tool_call in response.tool_calls:
            item.tool_calls.append(
                ToolCall(
                    tool_name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                    # TODO: Randomly generate the ID if not provided.
                    tool_call_id=tool_call.id,
                )
            )
    out.outputs.append(item)
    return out
