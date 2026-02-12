# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

model_config = {
    "model_config_name": {
        "model_name": "MODEL_NAME",
        "base_url": "YOUR_BASE_URL",
        "api_key": "YOUR_API_KEY",
    },
    "k2": {
        "model_name": "kimi-k2-250711",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "max_tokens": 32768,
        },
    },
    "doubao-1.6": {
        "model_name": "doubao-seed-1-6-250615",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "thinking": {"type": "enabled"},
            "max_tokens": 65535,
        },
    },
    "deepseek-r1": {
        "model_name": "deepseek-r1",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "max_tokens": 65535,
        },
    },
    "doubao-1.6-non-thinking": {  # for eval
        "model_name": "doubao-seed-1-6-250615",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "thinking": {"type": "disabled"},
            "max_tokens": 65535,
        },
    },
    "claude37-sonnet-thinking": {
        "model_name": "gcp-claude37-sonnet",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "temperature": 1,
            "extra_body": {"thinking": {"type": "enabled", "budget_tokens": 4096}},
            "max_tokens": 10240,
        },
        "is_claude_thinking": True,
    },
    "claude4-sonnet-thinking": {
        "model_name": "claude4-sonnet",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "temperature": 1,
            "extra_body": {"thinking": {"type": "enabled", "budget_tokens": 32768}},
            "max_tokens": 64000,
        },
        "is_claude_thinking": True,
    },
    "o3-medium": {
        "model_name": "o3-2025-04-16",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "max_tokens": 65535,
            "reasoning_effort": "medium",
        },
        "default_system_prompt": "Formatting re-enabled",
    },
    "gemini-2.5-pro": {
        "model_name": "gemini-2.5-pro-preview-06-05",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "max_tokens": 65535,
        },
    },
    "default_eval_config": {
        "model_name": "gpt-4.1-2025-04-14",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "max_tokens": 10240,
        },
        "temperature": 0,
    },
}
