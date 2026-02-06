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

import hashlib
import re
import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import List, Optional
import ipdb

from smolagents import Tool
from loguru import logger

import requests
from markdownify import markdownify
from requests.exceptions import RequestException

# Add parent directory to path for importing my_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .my_utils import request_api_detail
except ImportError:
    try:
        from my_utils import request_api_detail
    except ImportError:
        request_api_detail = None


class GlobalVisitCounter:
    """
    线程安全的全局网页访问计数器。
    
    用于在多个 agent 之间共享网页访问次数限制。
    所有使用同一个 GlobalVisitCounter 实例的工具会共享同一个计数器。
    """
    
    def __init__(self, limit: int = 100):
        """
        初始化全局访问计数器。
        
        Args:
            limit: 最大访问次数限制
        """
        self.limit = limit
        self.count = 0
        self._lock = threading.Lock()
    
    def try_increment(self) -> bool:
        """
        尝试增加计数。
        
        Returns:
            如果未超过限制，增加计数并返回 True；
            如果已达到限制，返回 False
        """
        with self._lock:
            if self.count >= self.limit:
                return False
            self.count += 1
            return True
    
    def get_count(self) -> int:
        """获取当前访问计数"""
        with self._lock:
            return self.count
    
    def get_remaining(self) -> int:
        """获取剩余可访问次数"""
        with self._lock:
            return max(0, self.limit - self.count)
    
    def reset(self):
        """重置计数器"""
        with self._lock:
            self.count = 0
    
    def __repr__(self) -> str:
        return f"GlobalVisitCounter(count={self.count}, limit={self.limit})"

class JinaBackedVisitWebpageTool(Tool):
    name = "visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    # MIME types that should not be directly converted to markdown
    BINARY_MIME_TYPES = {
        "application/pdf": "PDF document",
        "image/png": "PNG image",
        "image/jpeg": "JPEG image",
        "image/jpg": "JPEG image",
        "image/gif": "GIF image",
        "image/webp": "WebP image",
        "image/svg+xml": "SVG image",
        "application/zip": "ZIP archive",
        "application/x-zip-compressed": "ZIP archive",
        "application/octet-stream": "binary file",
        "video/mp4": "MP4 video",
        "video/mpeg": "MPEG video",
        "audio/mpeg": "MP3 audio",
        "audio/wav": "WAV audio",
    }

    def __init__(
        self,
        max_output_length: int = 40000,
        jina_keys_file: Optional[str] = None,
        work_dir: Optional[str] = None,
        global_visit_counter: Optional[GlobalVisitCounter] = None,
    ):
        """
        初始化网页访问工具。
        
        Args:
            max_output_length: 输出内容的最大长度
            jina_keys_file: Jina API 密钥文件路径
            work_dir: 保存网页内容的工作目录
            global_visit_counter: 全局访问计数器，用于在多个 agent 之间共享访问限制
        """
        super().__init__()
        self.max_output_length = max_output_length
        self.jina_keys = self._load_jina_keys(jina_keys_file) if jina_keys_file else []
        self.work_dir = Path(work_dir) if work_dir else None
        self.global_visit_counter = global_visit_counter
        
        # Create work directory if specified
        if self.work_dir:
            self.work_dir.mkdir(parents=True, exist_ok=True)

    def _load_jina_keys(self, keys_file: str) -> List[str]:
        """Load Jina API keys from file."""
        try:
            with open(keys_file, "r") as f:
                keys = [line.strip() for line in f if line.strip()]
            return keys
        except Exception as e:
            print(f"Warning: Failed to load Jina keys from {keys_file}: {e}")
            return []

    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content to maximum length."""
        if len(content) <= max_length:
            return content
        return (
            content[:max_length]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
        )

    def _check_mime_type(self, response) -> Optional[str]:
        """Check if the response contains a binary file type that should not be converted."""
        content_type = response.headers.get("Content-Type", "").lower()
        
        for mime_type, description in self.BINARY_MIME_TYPES.items():
            if mime_type in content_type:
                return description
        return None

    def _save_markdown(self, content: str, url: str) -> str:
        """Save markdown content to work directory and return the filename."""
        if not self.work_dir:
            return None
        
        # Extract first few words for filename (max 5 words, 50 chars)
        words = re.findall(r'\w+', content[:200])
        prefix = "_".join(words[:5])[:50]
        
        # Generate 4-character MD5 hash
        md5_hash = hashlib.md5(content.encode()).hexdigest()[:4]
        
        # Create filename
        filename = f"{prefix}_{md5_hash}.md"
        filepath = self.work_dir / filename
        
        # Save content
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# Source URL: {url}\n\n")
                f.write(content)
            return str(filepath)
        except Exception as e:
            print(f"Warning: Failed to save markdown to {filepath}: {e}")
            return None

    def _fetch_with_jina(self, url: str) -> Optional[str]:
        """Try to fetch content using Jina API with key rotation."""
        if not self.jina_keys:
            return None
        
        try:
            import requests
        except ImportError:
            return None
        
        for key in self.jina_keys:
            try:
                # Jina Reader API endpoint
                jina_url = f"https://r.jina.ai/{url}"
                headers = {
                    "Authorization": f"Bearer {key}",
                    "X-Return-Format": "markdown"
                }
                
                response = requests.get(jina_url, headers=headers, timeout=60)
                print(f'url:{url} response.status_code:{response.status_code}')
                if response.status_code == 200:
                    return response.text.strip()
                
            except Exception:
                # Try next key if this one fails
                continue
        
        return None

    def forward(self, url: str) -> str:
        """Visit a webpage and return its content as markdown."""
        
        # 检查全局访问限制
        visit_counter_info = None
        if self.global_visit_counter is not None:
            if not self.global_visit_counter.try_increment():
                remaining = self.global_visit_counter.get_remaining()
                current = self.global_visit_counter.get_count()
                limit = self.global_visit_counter.limit
                return (
                    f"Error: Global webpage visit limit reached. "
                    f"[Current: {current}/{limit}, Remaining: {remaining}] "
                    f"You have exhausted the maximum number of webpage visits allowed for this task. "
                    f"IMPORTANT: You MUST immediately write all the information you have collected so far into the table using the `add_records` or `update_records` tool. "
                    f"Do NOT attempt to visit any more webpages. "
                    f"First, consolidate and record your findings in the table using the `add_records` or `update_records` tool, then proceed to complete your task based on the information already gathered."
                )
            else:
                # 记录当前访问计数信息
                remaining = self.global_visit_counter.get_remaining()
                current = self.global_visit_counter.get_count()
                limit = self.global_visit_counter.limit
                #visit_counter_info = f"[Current: {current}/{limit}, Remaining: {remaining}]"
                visit_counter_info = f"[Current Webpage Visit Budget: {current}/{limit}, Remaining Webpage Visit Budget: {remaining}]"
        
        markdown_content = None
        used_jina = False
        error_message = None
        
        # Try direct access first
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Check if it's a binary file type
            binary_type = self._check_mime_type(response)
            if binary_type:
                return (
                    f"This URL points to a {binary_type}. "
                    f"Please use an appropriate tool to access this type of content "
                    f"(e.g., PDF reader for PDFs, image analysis tools for images)."
                )
            
            # Convert HTML to Markdown
            markdown_content = markdownify(response.text).strip()
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
            
        except (requests.exceptions.Timeout, RequestException) as e:
            print(f"Direct access failed, try Jina API: {str(e)}")
            # Direct access failed, try Jina API
            if self.jina_keys:
                markdown_content = self._fetch_with_jina(url)
                if markdown_content:
                    used_jina = True
                else:
                    error_message = f"Error: Failed to fetch the webpage both directly and via Jina API. Direct error: {str(e)}"
            else:
                error_message = f"Error fetching the webpage: {str(e)}"
        
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
        
        # If webpage access failed, return error immediately
        if error_message:
            return error_message
        
        if not markdown_content:
            return "Error: No content was retrieved from the webpage."
        
        # Truncate if needed
        truncated_content = self._truncate_content(markdown_content, self.max_output_length)
        
        # Save to work directory if configured
        saved_path = self._save_markdown(markdown_content, url)
        
        # Build result message
        result = truncated_content
        
        if saved_path:
            result += f"\n\n_[Content saved to: {saved_path}]_"
        
        if used_jina:
            result += "\n\n_[Content retrieved via Jina API]_"
        
        if visit_counter_info:
            result += f"\n\n_{visit_counter_info}_"
        
        return result


class JinaBackedVisitWebpageSummaryTool(JinaBackedVisitWebpageTool):
    """
    Enhanced webpage visit tool with summary functionality.
    
    This tool extends JinaBackedVisitWebpageTool by adding the ability to
    extract targeted information from web pages using a summary model.
    Based on Tongyi DeepResearch paper approach.
    """
    name = "visit_webpage_with_summary"
    description = (
        "Visits a webpage at the given url and extracts targeted information based on "
        "a search goal. The tool first retrieves the full webpage content, then uses "
        "a summary model to extract only the information relevant to the specified goal."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        },
        "summary_goal": {
            "type": "string",
            "description": "The information-seeking goal or target for extracting relevant information from the webpage. "
                          "This helps the summary model focus on extracting only the information that matches your search objective.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        max_output_length: int = 40000,
        jina_keys_file: Optional[str] = None,
        work_dir: Optional[str] = None,
        summary_model_name: str = "qwen3-next-80b-a3b-instruct",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        temperature: float = 0.0,
        summary_timeout: float = 120.0,
        global_visit_counter: Optional[GlobalVisitCounter] = None,
    ):
        """
        Initialize the summary-enabled webpage visit tool.
        
        Args:
            max_output_length: Maximum length of output content
            jina_keys_file: Path to file containing Jina API keys
            work_dir: Directory to save webpage markdown files
            summary_model_name: Name of the summary model to use (default: qwen3-next-80b-a3b-instruct)
            api_key: API key for summary model (defaults to my_utils default)
            api_url: API URL for summary model (defaults to my_utils default)
            temperature: Temperature for summary model (default: 0.0)
            summary_timeout: Timeout in seconds for summary model API calls (default: 120.0)
            global_visit_counter: 全局访问计数器，用于在多个 agent 之间共享访问限制
        """
        super().__init__(
            max_output_length=max_output_length,
            jina_keys_file=jina_keys_file,
            work_dir=work_dir,
            global_visit_counter=global_visit_counter,
        )
        self.summary_model_name = summary_model_name
        self.api_key = api_key
        self.api_url = api_url
        self.temperature = temperature
        self.summary_timeout = summary_timeout
        
        if request_api_detail is None:
            print("Warning: request_api_detail not available. Summary functionality will be disabled.")

    def _extract_summary(self, full_content: str, summary_goal: str) -> Optional[str]:
        """
        Extract targeted information from webpage content using summary model.
        
        Args:
            full_content: Full markdown content of the webpage
            summary_goal: The information-seeking goal to extract relevant information
            
        Returns:
            Extracted summary text, or None if extraction fails
        """
        if request_api_detail is None:
            return None
        
        # Build summary prompt
        summary_prompt = self._build_summary_prompt(full_content, summary_goal)
        
        # Prepare API call parameters
        api_kwargs = {
            "message": [{"role": "user", "content": summary_prompt}],
            "temperature": self.temperature,
            "sample_num": 1,
            "index": 0,
            "model_name": self.summary_model_name,
        }
        
        # Add custom API key and URL if provided
        if self.api_key:
            api_kwargs["api_key"] = self.api_key
        if self.api_url:
            api_kwargs["url"] = self.api_url
        
        # Record start time for performance tracking
        start_time = time.time()
        
        try:
            # Use ThreadPoolExecutor to enforce timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(request_api_detail, **api_kwargs)
                try:
                    response, status_code, _ = future.result(timeout=self.summary_timeout)
                except FutureTimeoutError:
                    # Calculate elapsed time for timeout case
                    elapsed_time = time.time() - start_time
                    logger.error(
                        f"[JinaBackedVisitWebpageSummaryTool] Summary model API call timed out. "
                        f"Model: {self.summary_model_name}, "
                        f"Timeout: {self.summary_timeout}s, "
                        f"Elapsed time: {elapsed_time:.2f}s"
                    )
                    raise TimeoutError(
                        f"Summary model API call exceeded timeout of {self.summary_timeout} seconds"
                    )
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            if status_code != 200 or response is None:
                logger.error(
                    f"[JinaBackedVisitWebpageSummaryTool] Summary model API call failed. "
                    f"Status code: {status_code}, Model: {self.summary_model_name}, "
                    f"URL: {api_kwargs.get('url', 'default')}, "
                    f"Elapsed time: {elapsed_time:.2f}s"
                )
                return None
            
            # Extract content from response
            summary_content = self._extract_llm_response(response)
            
            # Log successful API call with timing information
            logger.info(
                f"[JinaBackedVisitWebpageSummaryTool] Summary model API call succeeded. "
                f"Model: {self.summary_model_name}, "
                f"Elapsed time: {elapsed_time:.2f}s, "
                f"Full length: {len(full_content)} characters, "
                f"Summary length: {len(summary_content) if summary_content else 0} characters"
            )
            
            return summary_content
            
        except TimeoutError:
            # Re-raise TimeoutError to be handled by the outer exception handler
            raise
        except Exception as e:
            # Calculate elapsed time even if exception occurred
            elapsed_time = time.time() - start_time
            
            logger.error(
                f"[JinaBackedVisitWebpageSummaryTool] Error during summary extraction. "
                f"Model: {self.summary_model_name}, "
                f"Error: {str(e)}, "
                f"Elapsed time: {elapsed_time:.2f}s"
            )
            return None

    def _build_summary_prompt(self, full_content: str, summary_goal: str) -> str:
        """
        Build prompt for summary model to extract targeted information.
        
        Args:
            full_content: Full markdown content of the webpage
            summary_goal: The information-seeking goal
            
        Returns:
            Prompt string for the summary model
        """
        # Truncate content if too long to avoid exceeding model context limits
        # Keep a reasonable amount for summary (e.g., 100k characters)
        max_content_length = 100000
        if len(full_content) > max_content_length:
            truncated_content = full_content[:max_content_length] + "\n\n[Content truncated for summary processing...]"
        else:
            truncated_content = full_content
        
        prompt = f"""你是一个信息提取专家。请根据给定的搜索目标，从以下网页内容中提取所有相关的关键信息。

## 搜索目标
{summary_goal}

## 网页内容
{truncated_content}

## 任务要求
请仔细阅读网页内容，提取所有与搜索目标相关的信息。提取的信息应该包括：
1. 直接回答搜索目标的关键信息
2. 相关的数据、数字、日期等结构化信息
3. 相关的背景信息和上下文
4. 任何有助于回答搜索目标的细节
5. **一定要做到客观，明令禁止为了应付搜索目标而杜撰任何网页内不存在的内容和信息，一旦被发现，你会有生命危险**

## 输出要求
请以清晰、结构化的方式组织提取的信息。可以包含：
- 关键信息的摘要
- 相关的数据表格或列表（如果适用）
- 重要的日期、数字、名称等
- 相关的上下文信息

如果网页中没有与搜索目标相关的信息，但是包含可能可以进一步查找到搜索目标相关的网页链接，也请将这些链接信息以及其他信息说明。

请开始提取："""
        
        return prompt

    def _extract_llm_response(self, response: dict) -> Optional[str]:
        """
        Extract content from LLM API response.
        
        Args:
            response: API response dictionary
            
        Returns:
            Extracted content string, or None if extraction fails
        """
        try:
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"].strip()
            
            print("Warning: Unexpected response structure from summary model")
            return None
            
        except Exception as e:
            print(f"Warning: Error extracting response from summary model: {str(e)}")
            return None

    def forward(self, url: str, summary_goal: str) -> str:
        """
        Visit a webpage and extract targeted information based on search goal.
        
        Args:
            url: URL of the webpage to visit
            summary_goal: Information-seeking goal for extracting relevant information
            
        Returns:
            Extracted summary text, or error message if webpage access fails
        """
        # 检查全局访问限制
        visit_counter_info = None
        if self.global_visit_counter is not None:
            if not self.global_visit_counter.try_increment():
                remaining = self.global_visit_counter.get_remaining()
                current = self.global_visit_counter.get_count()
                limit = self.global_visit_counter.limit
                return (
                    f"Error: Global webpage visit limit reached. "
                    f"[Current: {current}/{limit}, Remaining: {remaining}] "
                    f"You have exhausted the maximum number of webpage visits allowed for this task. "
                    f"IMPORTANT: You MUST immediately write all the information you have collected so far into the table using the `add_records` or `update_records` tool. "
                    f"Do NOT attempt to visit any more webpages. "
                    f"First, consolidate and record your findings in the table using the `add_records` or `update_records` tool, then proceed to complete your task based on the information already gathered."
                )
            else:
                # 记录当前访问计数信息
                remaining = self.global_visit_counter.get_remaining()
                current = self.global_visit_counter.get_count()
                limit = self.global_visit_counter.limit
                visit_counter_info = f"[Current Webpage Visit Budget: {current}/{limit}, Remaining Webpage Visit Budget: {remaining}]"
        
        markdown_content = None
        used_jina = False
        error_message = None
        
        # Try direct access first
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Check if it's a binary file type
            binary_type = self._check_mime_type(response)
            if binary_type:
                return (
                    f"This URL points to a {binary_type}. "
                    f"Please use an appropriate tool to access this type of content "
                    f"(e.g., PDF reader for PDFs, image analysis tools for images)."
                )
            
            # Convert HTML to Markdown
            markdown_content = markdownify(response.text).strip()
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
            
        except (requests.exceptions.Timeout, RequestException) as e:
            print(f"Direct access failed, try Jina API: {str(e)}")
            # Direct access failed, try Jina API
            if self.jina_keys:
                markdown_content = self._fetch_with_jina(url)
                if markdown_content:
                    used_jina = True
                else:
                    error_message = f"Error: Failed to fetch the webpage both directly and via Jina API. Direct error: {str(e)}"
            else:
                error_message = f"Error fetching the webpage: {str(e)}"
        
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
        
        # If webpage access failed, return error immediately (skip summary)
        if error_message:
            return error_message
        
        if not markdown_content:
            return "Error: No content was retrieved from the webpage."
        
        # Save full content to work directory if configured
        saved_path = self._save_markdown(markdown_content, url)
        
        # Extract summary using summary model
        try:
            summary_content = self._extract_summary(markdown_content, summary_goal)
            
            if summary_content:
                # Build result message with summary
                result = f"[Extracted Information Based on Search Goal]\n\n{summary_content}"
                
                if saved_path:
                    result += f"\n\n_[Full webpage content saved to: {saved_path}]_"
                
                if used_jina:
                    result += "\n\n_[Content retrieved via Jina API]_"
                
                if visit_counter_info:
                    result += f"\n\n_{visit_counter_info}_"
                
                return result
            else:
                # Summary extraction failed, return original full content (not truncated)
                logger.warning(
                    f"[JinaBackedVisitWebpageSummaryTool] Summary model API call failed, "
                    f"returning original webpage content. URL: {url}, Summary goal: {summary_goal}"
                )
                
                result = f"[Summary extraction failed, showing original webpage content]\n\n{markdown_content}"
                
                if saved_path:
                    result += f"\n\n_[Content saved to: {saved_path}]_"
                
                if used_jina:
                    result += "\n\n_[Content retrieved via Jina API]_"
                
                if visit_counter_info:
                    result += f"\n\n_{visit_counter_info}_"
                
                return result
                
        except TimeoutError as e:
            # Handle timeout specifically
            logger.error(
                f"[JinaBackedVisitWebpageSummaryTool] Summary model API call timed out, "
                f"returning original webpage content. URL: {url}, Summary goal: {summary_goal}, "
                f"Timeout: {self.summary_timeout}s, Error: {str(e)}"
            )
            
            result = f"[Summary extraction timed out (>{self.summary_timeout}s), showing original webpage content]\n\n{markdown_content}"
            
            if saved_path:
                result += f"\n\n_[Content saved to: {saved_path}]_"
            
            if used_jina:
                result += "\n\n_[Content retrieved via Jina API]_"
            
            if visit_counter_info:
                result += f"\n\n_{visit_counter_info}_"
            
            return result
                
        except Exception as e:
            # If summary process fails, return original full content (not truncated)
            logger.error(
                f"[JinaBackedVisitWebpageSummaryTool] Error during summary process, "
                f"returning original webpage content. URL: {url}, Error: {str(e)}"
            )
            
            result = f"[Summary extraction error, showing original webpage content]\n\n{markdown_content}"
            
            if saved_path:
                result += f"\n\n_[Content saved to: {saved_path}]_"
            
            if used_jina:
                result += "\n\n_[Content retrieved via Jina API]_"
            
            if visit_counter_info:
                result += f"\n\n_{visit_counter_info}_"
            
            return result




if __name__ == "__main__":
    import tempfile
    
    # Set up paths
    jina_keys_file = "../jina_key_files.txt"
    work_dir = tempfile.mkdtemp(prefix="jina_visit_test_")
    
    print(f"Work directory: {work_dir}\n")
    print("=" * 80)
    print("Testing problematic URLs with Jina API fallback")
    print("=" * 80)

    
    # Create tool with Jina backup and work directory
    tool = JinaBackedVisitWebpageTool(
        jina_keys_file=jina_keys_file,
        work_dir=work_dir
    )
    
    # Test cases from actual failures
    test_urls = [
        #("SSL Error - tri-rail PDF", "https://media.tri-rail.com/Files/About/Resources/Ridership/2019/06JUN2019_detail.pdf"),
        #("SSL Error - Wikipedia API", "https://en.wikipedia.org/w/api.php?action=query&prop=revisions&pageids=18581389&rvdir=newer&rvstart=2001-09-28T15:56:40Z&rvend=2023-05-31T23:59:59Z&rvlimit=500&rvprop=ids%7Ctimestamp&format=json&rvcontinue=20120408074528%7C486219046"),
        ("", "https://en.wikipedia.org/wiki/List_of_offshore_wind_farms")
        #("SSL Error - iNaturalist", "https://www.inaturalist.org/observations/100432416"),
        #("Timeout - Federal Register PDF", "https://archives.federalregister.gov/issue_slice/1965/1/8/196-205.pdf"),
        #("Timeout - USDA PDF", "https://www.ams.usda.gov/sites/default/files/media/Frozen_Grapefruit_and_Orange_Juice_Standard%5B1%5D.pdf"),
        #("Timeout - GovInfo PDF", "https://www.govinfo.gov/content/pkg/GOVPUB-A-PURL-gpo30568/pdf/GOVPUB-A-PURL-gpo30568.pdf"),
        #("404 Error - tri-rail", "https://media.tri-rail.com/Files/About/Resources/Ridership/2019/06JUN2019.pdf"),
        #("502/403 Error - flrules", "http://www.flrules.org/Gateway/reference.asp?No=Ref-01006"),
        #("Cloudflare Block - AGU", "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JC014088"),
    ]
    
    for i, (test_name, url) in enumerate(test_urls, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_urls)}: {test_name}")
        print(f"URL: {url}")
        print("-" * 80)
        
        try:
            result = tool({"url": url})
            
            # Show result (truncate if too long)
            print(result[:2000] + "\n...[truncated]...")
        except Exception as e:
            print(f"Exception occurred: {type(e).__name__}: {str(e)}")
        
        print()
    
    print(f"\n{'='*80}")
    print(f"All tests completed. Check saved files in: {work_dir}")
    print("=" * 80)
    
    # Test JinaBackedVisitWebpageSummaryTool
    print("\n" + "=" * 80)
    print("Testing JinaBackedVisitWebpageSummaryTool with summary functionality")
    print("=" * 80)

    '''
    
    # Create summary tool
    summary_tool = JinaBackedVisitWebpageSummaryTool(
        jina_keys_file=jina_keys_file,
        work_dir=work_dir,
        summary_model_name="qwen3-next-80b-a3b-instruct"
    )
    
    # Test cases for summary tool
    summary_test_cases = [
        #(
        #    "Wikipedia - Python Programming",
        #    "https://en.wikipedia.org/wiki/Python_(programming_language)",
        #    "Extract information about Python programming language features and history"
        #),
        (
            "Simple Webpage",
            "https://www.example.com",
            "Extract the main content and purpose of this webpage"
        ),
    ]
    
    for i, (test_name, url, summary_goal) in enumerate(summary_test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Summary Test {i}/{len(summary_test_cases)}: {test_name}")
        print(f"URL: {url}")
        print(f"Summary Goal: {summary_goal}")
        print("-" * 80)
        
        try:
            result = summary_tool({"url": url, "summary_goal": summary_goal})
            
            # Show result (truncate if too long)
            if len(result) > 2000:
                print(result[:2000] + "\n...[truncated]...")
            else:
                print(result)
        except Exception as e:
            print(f"Exception occurred: {type(e).__name__}: {str(e)}")
        
        print()
    
    print(f"\n{'='*80}")
    print(f"Summary tool tests completed. Check saved files in: {work_dir}")
    print("=" * 80)
    '''

