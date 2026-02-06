#!/usr/bin/env python
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

# coding=utf-8

"""
Enhanced JSON parsing utilities for smolagents.
This module provides a more robust version of parse_json_blob that first attempts
to extract JSON from markdown code blocks, then falls back to the original logic.
"""

import json
import re


def parse_json_blob(json_blob: str) -> tuple[dict[str, str], str]:
    """
    Enhanced JSON blob parser that first tries to extract JSON from markdown code blocks,
    then falls back to the original smolagents logic.
    
    This function attempts two methods:
    1. First, tries to extract JSON from markdown code blocks (```json ... ```)
    2. If that fails, falls back to the original smolagents logic (finding first { and last })
    
    Args:
        json_blob: The input string that may contain JSON
        
    Returns:
        A tuple of (json_data, remaining_text) where:
        - json_data: The parsed JSON dictionary
        - remaining_text: The text before the JSON blob
        
    Raises:
        ValueError: If both methods fail to extract valid JSON
    """
    # Method 1: Try to extract JSON from markdown code blocks
    # Pattern matches: ```json ... ``` or ``` ... ``` (with optional json language tag)
    markdown_patterns = [
        r"```json\s*\n(.*?)\n```",  # ```json\n...\n```
        r"```json\s*(.*?)```",      # ```json...```
        r"```\s*\n(.*?)\n```",      # ```\n...\n``` (fallback for any code block)
        r"```\s*(.*?)```",           # ```...``` (fallback for any code block)
    ]
    
    for pattern in markdown_patterns:
        # Use finditer to get both the match content and its position
        for match_obj in re.finditer(pattern, json_blob, re.DOTALL):
            # Extract the captured group (the JSON content inside the code block)
            json_str = match_obj.group(1).strip()
            if json_str:
                try:
                    json_data = json.loads(json_str, strict=False)
                    # Extract text before the code block
                    remaining_text = json_blob[:match_obj.start()].strip()
                    return json_data, remaining_text
                except json.JSONDecodeError:
                    # This match wasn't valid JSON, try next match
                    continue
    
    # Method 2: Fall back to original smolagents logic
    try:
        first_accolade_index = json_blob.find("{")
        if first_accolade_index == -1:
            raise ValueError("The model output does not contain any JSON blob.")
        
        # Find all closing braces
        closing_braces = [a.start() for a in re.finditer(r"}", json_blob)]
        if not closing_braces:
            raise ValueError("The model output does not contain a complete JSON blob.")
        
        last_accolade_index = closing_braces[-1]
        json_str = json_blob[first_accolade_index : last_accolade_index + 1]
        json_data = json.loads(json_str, strict=False)
        return json_data, json_blob[:first_accolade_index]
    except IndexError:
        raise ValueError("The model output does not contain any JSON blob.")
    except json.JSONDecodeError as e:
        place = e.pos
        if place > 0 and place < len(json_blob) and json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        # Provide more context about the error
        error_context_start = max(0, place - 20)
        error_context_end = min(len(json_blob), place + 20)
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[error_context_start:error_context_end]}'."
        )

