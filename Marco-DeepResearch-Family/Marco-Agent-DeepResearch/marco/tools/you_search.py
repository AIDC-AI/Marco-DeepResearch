import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from .base import BaseTool, register_tool


logger = logging.getLogger(__name__)


DEFAULT_API_URL = "https://api.you.com/mcp?profile=free"


@register_tool("you_search")
class YouSearchTool(BaseTool):

    name = "you_search"
    description = "Search the web via You.com to find relevant information and URLs. Supports authenticated (YOUCOM_API_KEY) and keyless (free tier) modes."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (one intent per query, phrased naturally).",
            },
            "count": {
                "type": "integer",
                "description": "Number of results to return (default 10, max 30).",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: int = 20,
        max_retries: int = 5,
        max_workers: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("YOUCOM_API_KEY", "")
        self.api_url = api_url or os.getenv(
            "YOUCOM_API_URL",
            "https://api.you.com/mcp" if self.api_key else DEFAULT_API_URL,
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max_workers

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            try:
                parsed: dict = json.loads(params)
            except json.JSONDecodeError:
                return "[You.com Search] Invalid JSON parameters."
        else:
            parsed = params

        query = parsed.get("query", "")
        if not query:
            return "[You.com Search] `query` is required."

        count = min(int(parsed.get("count", 10)), 30)

        logger.info(f"[You.com Search] query={query!r} count={count}")

        return self._search(query, count)

    def _search(self, query: str, count: int = 10) -> str:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "you-search",
                "arguments": {
                    "query": query,
                    "count": count,
                },
            },
            "id": 1,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                result = self._parse_sse_response(response.text)
                if result is None:
                    return f"[You.com Search] Failed to parse response for query: {query}"

                results = self._extract_results(result)
                return self._format(query, results)

            except Exception as e:
                logger.warning(
                    f"[You.com Search] query={query!r} attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(1)

        return f"[You.com Search] Failed after {self.max_retries} attempts for query: {query}"

    @staticmethod
    def _parse_sse_response(text: str) -> Optional[dict]:
        """Parse SSE (Server-Sent Events) response and extract the result JSON."""
        for line in text.splitlines():
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "result" in data:
                        return data["result"]
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _extract_results(result: dict) -> List[Dict[str, str]]:
        """Extract search results from the MCP response content."""
        parsed = []
        content_list = result.get("content", [])
        for content in content_list:
            if content.get("type") == "text":
                try:
                    text_data = json.loads(content["text"])
                    for section in ("web", "news", "knowledge"):
                        for item in text_data.get("results", {}).get(section, []):
                            parsed.append({
                                "title": item.get("title", ""),
                                "link": item.get("url", ""),
                                "snippet": item.get("description", item.get("snippet", "")),
                            })
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        return parsed

    @staticmethod
    def _format(query: str, results: List[Dict[str, str]]) -> str:
        if not results:
            return f"[You.com Search] Query: {query}\nNo results found."

        lines = [f"[You.com Search] Query: {query} ({len(results)} results)"]
        for i, item in enumerate(results, 1):
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            lines.append("")
            lines.append(f"{i}. {title}|{link}")
            if snippet:
                lines.append(snippet)
        return "\n".join(lines)