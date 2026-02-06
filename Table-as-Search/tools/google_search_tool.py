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
Google Search Tool: 基于 AI-HUB 定制化搜索引擎的 Google 搜索工具
支持自动重试和回退到 Jina Search
"""
import os
import json
import threading
# import ipdb  # 注释掉，避免多进程环境下的导入问题
from datetime import datetime
import requests
from typing import Dict, List, Any, Optional
from smolagents import Tool


class GlobalSearchCounter:
    """
    线程安全的全局搜索次数计数器。
    
    用于在多个 agent 之间共享搜索次数限制。
    所有使用同一个 GlobalSearchCounter 实例的工具会共享同一个计数器。
    """
    
    def __init__(self, limit: int = 100):
        """
        初始化全局搜索计数器。
        
        Args:
            limit: 最大搜索次数限制
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
        """获取当前搜索计数"""
        with self._lock:
            return self.count
    
    def get_remaining(self) -> int:
        """获取剩余可搜索次数"""
        with self._lock:
            return max(0, self.limit - self.count)
    
    def reset(self):
        """重置计数器"""
        with self._lock:
            self.count = 0
    
    def __repr__(self) -> str:
        return f"GlobalSearchCounter(count={self.count}, limit={self.limit})"


class GoogleSearchTool(Tool):
    """
    Google 搜索工具，基于 AI-HUB 定制化搜索引擎
    
    使用 Google Custom Search API 进行稳定的搜索
    """
    
    name = "google_search"
    description = """
使用 Google Custom Search API 进行网络搜索。
这是一个稳定可靠的搜索工具，基于 AI-HUB 定制化搜索引擎。

Args:
    query: 要搜索的查询文本
    
Returns:
    搜索结果，包含标题、URL 和摘要

注意由于 Google Custom Search 的要求，请慎重使用同时带有多个引号的精确查询 query，因为 Google Search API 要求它们同时精确匹配，通常可能会导致无结果或者搜索失败。
"""
    inputs = {
        "query": {
            "type": "string",
            "description": "要搜索的查询文本"
        }
    }
    output_type = "string"
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        api_base: Optional[str] = None,
        limit: int = 10, 
        max_retries: int = 3,
        global_search_counter: Optional["GlobalSearchCounter"] = None
    ):
        """
        初始化 Google 搜索工具
        
        Args:
            api_key: API 密钥，如果为 None 则从环境变量 SEARCH_API_KEY 读取
            api_base: API base url
            limit: 返回的搜索结果数量（默认10）
            max_retries: Google Search 失败时的最大重试次数（默认3）
            global_search_counter: 全局搜索计数器，用于在多个 agent 之间共享搜索限制
        """
        super().__init__()
        
        self.api_key = api_key or os.environ.get("SEARCH_API_KEY")
        self.base_url = api_base or os.environ.get("SEARCH_API_BASE")
        self.service = "google"
        self.limit = limit
        self.max_retries = max_retries
        self.global_search_counter = global_search_counter
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        # 延迟导入 JinaSearchTool，避免循环依赖
        self._jina_tool = None
    
    def search(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行搜索请求
        
        Args:
            query: 搜索查询词
            params: 额外的搜索参数
            
        Returns:
            搜索结果字典
        """
        # 构建请求 URL
        url = f"{self.base_url}/customsearch/{self.service}/search"
        
        # 构建请求体
        payload = {"q": query}
        if params:
            payload.update(params)
        
        # 发送请求
        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": str(e),
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }
    
    def _get_jina_tool(self):
        """延迟加载 JinaSearchTool，避免循环依赖"""
        if self._jina_tool is None:
            # 延迟导入，避免循环依赖
            # 尝试不同的导入路径以适应不同的调用场景
            try:
                from tools.jina_search_tool import JinaSearchTool
            except ImportError:
                # 如果从 scripts 目录导入失败，尝试直接导入
                from jina_search_tool import JinaSearchTool
            self._jina_tool = JinaSearchTool(limit=self.limit)
        return self._jina_tool
    
    def forward(self, query: str) -> str:
        """
        执行搜索并返回格式化的结果
        
        策略：
        1. 先尝试使用 Google Search API（参考 web_search_agent_test.py 的实现）
        2. 如果失败，重试最多 max_retries 次
        3. 如果所有重试都失败，回退到 Jina Search
        
        Args:
            query: 搜索查询
            
        Returns:
            格式化的搜索结果字符串
        """
        # 检查全局搜索限制
        search_counter_info = None
        if self.global_search_counter is not None:
            if not self.global_search_counter.try_increment():
                remaining = self.global_search_counter.get_remaining()
                current = self.global_search_counter.get_count()
                limit = self.global_search_counter.limit
                return (
                    f"Error: Global search limit reached. "
                    f"[Current: {current}/{limit}, Remaining: {remaining}] "
                    f"You have exhausted the maximum number of searches allowed for this task. "
                    f"IMPORTANT: You MUST immediately write all the information you have collected so far into the table using the `add_records` or `update_records` tool. "
                    f"Do NOT attempt to perform any more searches. "
                    f"First, consolidate and record your findings in the table using the `add_records` or `update_records` tool, then proceed to complete your task based on the information already gathered."
                )
            else:
                # 记录当前搜索计数信息
                remaining = self.global_search_counter.get_remaining()
                current = self.global_search_counter.get_count()
                limit = self.global_search_counter.limit
                search_counter_info = f"[Current Google Search Budget: {current}/{limit}, Remaining Google Search Budget: {remaining}]"
        
        # 第一步：尝试 Google Search，最多重试 max_retries 次
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # 执行搜索（参考 web_search_agent_test.py 的 search 方法）
                result = self.search(query)
                #raise Exception("手动禁止 Google Search 调用，测试 JINA search tool")
                
                # 检查错误
                if result.get("error"):
                    error_msg = result.get("message", "Unknown error")
                    status_code = result.get("status_code")
                    last_error = f"Google Search API error (Status {status_code}): {error_msg}" if status_code else f"Google Search API error: {error_msg}"
                    
                    # 如果是最后一次尝试，跳出循环
                    if attempt == self.max_retries:
                        break
                    continue
                
                # 提取搜索结果
                results = []
                
                # 从 organic 字段提取结果
                if "organic" in result and result["organic"]:
                    for item in result["organic"][:self.limit]:
                        results.append({
                            'title': item.get('title', 'No title'),
                            'url': item.get('link', 'No URL'),
                            'snippet': item.get('snippet', 'No description')
                        })
                
                # 如果成功获取到结果，返回格式化的结果
                if results:
                    formatted_result = "=" * 80 + '\n'.join([self.format_result(result, index) for index, result in enumerate(results)]) + '\n' + "=" * 80
                    if search_counter_info:
                        formatted_result += f"\n\n_{search_counter_info}_"
                    return formatted_result
                else:
                    # 没有结果，也视为失败，继续重试或回退
                    # 打印调试信息：查看API返回的完整响应
                    #debug_info = f"Google Search returned no results. API response keys: {list(result.keys())}, Full response: {json.dumps(result, indent=2)[:500]}"
                    debug_info = f"Google Search returned no results. You MUST refine your search query and strategy and try again."
                    last_error = debug_info
                    if attempt == self.max_retries:
                        break
                    continue
                    
            except Exception as e:
                last_error = f"Google Search exception: {str(e)}"
                if attempt == self.max_retries:
                    break
                continue
        
        # 第二步：如果 Google Search 失败，回退到 Jina Search
        try:
            jina_tool = self._get_jina_tool()
            jina_result = jina_tool.forward(query)
            # 在结果前添加标记，表明使用了 Jina Search
            fallback_result = f"[回退到 Jina Search]\n{jina_result}"
            if search_counter_info:
                fallback_result += f"\n\n_{search_counter_info}_"
            return fallback_result
        except Exception as e:
            # 如果 Jina Search 也失败，抛出包含两个错误的异常
            raise Exception(
                f"Google Search Failed or Return No Results. You MUST refine your search query and strategy and try again."
            )

    def format_result(self, result: Dict[str, Any], index: str) -> str:
        """
        格式化搜索结果用于显示
        
        Args:
            result: 搜索结果
            search_type: 搜索类型
            
        Returns:
            格式化后的字符串
        """
        if result.get("error"):
            return f"❌ 错误: {result.get('message')}"
        
        output = []
        #output.append("=" * 80)
        #output.append("-" * 80)
        # 根据搜索类型显示不同的结果
        self._format_organic_results(result, output, index)
        #output.append("=" * 80)
        return "\n".join(output)
    
    def _format_organic_results(self, result: Dict[str, Any], output: List[str], index: str):
        """格式化普通搜索结果"""
        output.append(f"\n  [{index}] {result.get('title', 'N/A')}")
        output.append(f"      链接: {result.get('url', 'N/A')}")
        output.append(f"      摘要: {result.get('snippet', 'N/A')}")


def run_test_suite():
    """运行完整的测试套件"""
    
    print("\n" + "=" * 80)
    print("🚀 Web Search Agent 测试套件")
    print("=" * 80)
    
    # 从环境变量或配置获取 API Key 和 API Base URL
    api_key = os.environ.get("SEARCH_API_KEY")
    api_base = os.environ.get("SEARCH_API_BASE")
    
    # 初始化搜索代理
    agent = GoogleSearchTool(api_key=api_key, api_base=api_base)
    
    print(f"\n✅ 初始化完成")
    print(f"   - API Base: {api_base or 'Not set'}")
    print(f"   - 服务: Google Custom Search")
    print(f"   - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 测试用例定义
    test_cases = [
        #{
        #    "name": "普通搜索 - 中文",
        #    "type": "search",
        #    "query": "罗斯福",
        #    "params": None
        #},
        #{
        #    "name": "普通搜索 - 英文",
        #    "type": "search",
        #    "query": "artificial intelligence",
        #    "params": None
        #},
        # 原始查询（可能太严格，导致无结果）
        #{
        #    "name": "原始查询 - 多个精确短语",
        #    "type": "search",
        #    "query": 'research publication "novel baked goods" "unconventional ingredients" "enhanced consumer value"',
        #    "params": None
        #},
        # 改进方案1：去掉部分引号，使用更灵活的搜索
        {
            "name": "改进查询1 - 部分精确匹配",
            "type": "search",
            "query": 'research publication "novel baked goods" unconventional ingredients enhanced consumer value',
            "params": None
        },
        # 改进方案2：只保留最关键的精确短语
        {
            "name": "改进查询2 - 简化精确匹配",
            "type": "search",
            "query": 'research publication novel baked goods unconventional ingredients',
            "params": None
        },
        # 改进方案3：使用AND操作符（如果API支持）
        {
            "name": "改进查询3 - 使用关键词组合",
            "type": "search",
            "query": "research publication novel baked goods unconventional ingredients consumer value",
            "params": None
        }
    ]
    
    # 执行测试
    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"📝 测试 {i}/{len(test_cases)}: {test['name']}")
        print(f"{'=' * 80}")
        
        try:
            # 执行搜索 - 使用 search() 获取原始字典结果用于测试
            raw_result = agent.search(
                query=test['query'],
                params=test['params']
            )
            
            # 使用 forward() 获取格式化的字符串结果用于显示
            formatted_result = agent.forward(
                query=test['query']
            )
            print(formatted_result)
            
            # 记录结果（使用原始字典结果）
            results.append({
                "test": test['name'],
                "type": test['type'],
                "query": test['query'],
                "success": not raw_result.get("error", False) if isinstance(raw_result, dict) else False,
                "credits": raw_result.get("credits", 0) if isinstance(raw_result, dict) else 0,
                "result_count": len(raw_result.get("organic", raw_result.get("images", raw_result.get("news", raw_result.get("videos", raw_result.get("shopping", raw_result.get("places", raw_result.get("suggestions", [])))))))) if isinstance(raw_result, dict) else 0
            })
            
            # 保存原始结果到文件
            output_dir = "search_results"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{output_dir}/{test['type']}_{test['query'][:20].replace(' ', '_')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(raw_result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 原始结果已保存到: {filename}")
            
        except Exception as e:
            print(f"\n❌ 测试失败: {str(e)}")
            results.append({
                "test": test['name'],
                "type": test['type'],
                "query": test['query'],
                "success": False,
                "error": str(e)
            })
    
    # 生成测试报告
    print("\n\n" + "=" * 80)
    print("📊 测试报告")
    print("=" * 80)
    
    total = len(results)
    success = sum(1 for r in results if r.get("success", False))
    failed = total - success
    total_credits = sum(r.get("credits", 0) for r in results)
    
    print(f"\n总测试数: {total}")
    print(f"成功: {success} ✅")
    print(f"失败: {failed} ❌")
    print(f"成功率: {(success/total*100):.1f}%")
    print(f"消耗总积分: {total_credits} 💰")
    
    print("\n详细结果:")
    for i, result in enumerate(results, 1):
        status = "✅" if result.get("success", False) else "❌"
        print(f"  {i}. {status} {result['test']} - 查询: '{result['query']}'")
        if result.get("success"):
            print(f"     结果数: {result.get('result_count', 0)}, 积分: {result.get('credits', 0)}")
        else:
            print(f"     错误: {result.get('error', 'Unknown error')}")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 保存测试报告
    report_filename = f"search_results/test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total": total,
                "success": success,
                "failed": failed,
                "success_rate": f"{(success/total*100):.1f}%",
                "total_credits": total_credits
            },
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n📝 测试报告已保存到: {report_filename}")


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 80)
    print("🌐 Web Search Agent 测试工具")
    print("   基于 AI-HUB 定制化搜索引擎")
    print("=" * 80)
    
    run_test_suite()


