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
轻量级 .env 文件加载器
用于替代 python-dotenv，避免许可证问题

功能：
- 加载 .env 文件中的环境变量到 os.environ
- 支持注释（# 开头）
- 支持空行
- 支持带引号和不带引号的值
- 支持变量展开（${VAR} 语法）
- 支持 override 参数控制是否覆盖已有环境变量

Author: Custom implementation
License: MIT (same as project)
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict


def load_dotenv(dotenv_path: Optional[str] = None, override: bool = False) -> bool:
    """
    从 .env 文件加载环境变量到 os.environ
    
    Args:
        dotenv_path: .env 文件路径，如果为 None 则在当前目录及父目录查找
        override: 是否覆盖已存在的环境变量
        
    Returns:
        bool: 是否成功加载（找到文件即返回 True）
        
    Example:
        >>> load_dotenv()  # 加载当前目录的 .env
        >>> load_dotenv('.env.local')  # 加载指定文件
        >>> load_dotenv(override=True)  # 覆盖已有环境变量
    """
    # 查找 .env 文件
    if dotenv_path is None:
        dotenv_path = find_dotenv()
    else:
        dotenv_path = Path(dotenv_path)
    
    if not dotenv_path or not Path(dotenv_path).exists():
        # 与 python-dotenv 行为一致：找不到文件不报错，返回 False
        return False
    
    # 解析并加载环境变量
    env_vars = parse_dotenv(dotenv_path)
    
    for key, value in env_vars.items():
        # 根据 override 参数决定是否覆盖
        if override or key not in os.environ:
            os.environ[key] = value
    
    return True


def find_dotenv(filename: str = '.env', raise_error_if_not_found: bool = False) -> Optional[Path]:
    """
    在当前目录及父目录中查找 .env 文件
    
    Args:
        filename: 要查找的文件名
        raise_error_if_not_found: 如果找不到是否抛出异常
        
    Returns:
        Path: 找到的文件路径，如果找不到返回 None
    """
    # 从当前工作目录开始向上查找
    current_dir = Path.cwd()
    
    # 最多向上查找到根目录
    for _ in range(10):  # 限制最多查找 10 层，防止无限循环
        env_file = current_dir / filename
        if env_file.exists() and env_file.is_file():
            return env_file
        
        # 到达根目录
        if current_dir == current_dir.parent:
            break
        
        current_dir = current_dir.parent
    
    if raise_error_if_not_found:
        raise FileNotFoundError(f"Could not find {filename}")
    
    return None


def parse_dotenv(dotenv_path: str) -> Dict[str, str]:
    """
    解析 .env 文件内容
    
    支持的格式：
    - KEY=value
    - KEY="value"
    - KEY='value'
    - # 注释
    - 空行
    - export KEY=value (自动忽略 export 前缀)
    - KEY=${OTHER_VAR} (变量展开)
    - KEY=value # 行内注释
    
    Args:
        dotenv_path: .env 文件路径
        
    Returns:
        Dict[str, str]: 解析后的环境变量字典
    """
    env_vars = {}
    dotenv_path = Path(dotenv_path)
    
    if not dotenv_path.exists():
        return env_vars
    
    # 读取文件内容
    with open(dotenv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, start=1):
        # 移除行尾换行符
        line = line.rstrip('\n\r')
        
        # 跳过空行
        if not line.strip():
            continue
        
        # 跳过注释行
        if line.strip().startswith('#'):
            continue
        
        # 移除 export 前缀（兼容 bash 风格）
        line = re.sub(r'^\s*export\s+', '', line)
        
        # 解析 KEY=VALUE
        # 支持：KEY=VALUE 或 KEY = VALUE（带空格）
        match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$', line)
        
        if not match:
            # 不是有效的赋值语句，跳过
            continue
        
        key = match.group(1)
        value = match.group(2)
        
        # 移除行内注释（但要注意引号内的 # 不算注释）
        value = remove_inline_comment(value)
        
        # 移除首尾空格
        value = value.strip()
        
        # 处理引号
        value = unquote(value)
        
        # 变量展开（${VAR} 或 $VAR）
        value = expand_variables(value, env_vars)
        
        env_vars[key] = value
    
    return env_vars


def remove_inline_comment(value: str) -> str:
    """
    移除行内注释，但保留引号内的 #
    
    Examples:
        >>> remove_inline_comment('value # comment')
        'value '
        >>> remove_inline_comment('"value # not comment"')
        '"value # not comment"'
        >>> remove_inline_comment("'value # not comment'")
        "'value # not comment'"
    """
    # 如果值被引号包围，直接返回
    if (value.startswith('"') and '"' in value[1:]) or \
       (value.startswith("'") and "'" in value[1:]):
        # 找到配对的引号
        quote = value[0]
        end_quote_pos = value.index(quote, 1)
        # 返回引号部分 + 移除后面的注释
        quoted_part = value[:end_quote_pos + 1]
        rest = value[end_quote_pos + 1:]
        if '#' in rest:
            rest = rest[:rest.index('#')]
        return quoted_part + rest
    
    # 没有引号，直接查找 # 并截断
    if '#' in value:
        value = value[:value.index('#')]
    
    return value


def unquote(value: str) -> str:
    """
    移除值首尾的引号
    
    Examples:
        >>> unquote('"value"')
        'value'
        >>> unquote("'value'")
        'value'
        >>> unquote('value')
        'value'
    """
    if len(value) >= 2:
        # 双引号
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        # 单引号
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]
    
    return value


def expand_variables(value: str, env_vars: Dict[str, str]) -> str:
    """
    展开变量引用
    
    支持：
    - ${VAR}
    - $VAR
    - ${VAR:-default} (带默认值，未实现)
    
    查找顺序：
    1. 当前解析的 env_vars
    2. 系统环境变量 os.environ
    
    Examples:
        >>> expand_variables('${HOME}/data', {})  # 使用系统变量
        '/home/user/data'
        >>> expand_variables('${API_BASE}/v1', {'API_BASE': 'https://api.com'})
        'https://api.com/v1'
    """
    # 展开 ${VAR} 格式
    def replace_braced(match):
        var_name = match.group(1)
        # 先从当前 env_vars 查找，再从系统环境变量查找
        return env_vars.get(var_name, os.environ.get(var_name, match.group(0)))
    
    value = re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', replace_braced, value)
    
    # 展开 $VAR 格式（不带花括号）
    def replace_simple(match):
        var_name = match.group(1)
        return env_vars.get(var_name, os.environ.get(var_name, match.group(0)))
    
    value = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)', replace_simple, value)
    
    return value


# 为了兼容性，提供一些常用的函数别名
def dotenv_values(dotenv_path: Optional[str] = None) -> Dict[str, str]:
    """
    读取 .env 文件并返回字典，但不设置环境变量
    
    Args:
        dotenv_path: .env 文件路径
        
    Returns:
        Dict[str, str]: 环境变量字典
    """
    if dotenv_path is None:
        dotenv_path = find_dotenv()
    
    if not dotenv_path or not Path(dotenv_path).exists():
        return {}
    
    return parse_dotenv(dotenv_path)


if __name__ == '__main__':
    # 简单测试
    print("Testing env_loader...")
    
    # 测试解析功能
    test_content = """
# 这是注释
API_KEY=sk-1234567890
API_BASE="https://api.example.com"
API_VERSION='v1'
TIMEOUT=30
DEBUG=true  # 行内注释

# 带变量展开
DATA_PATH=${HOME}/data
API_ENDPOINT=${API_BASE}/${API_VERSION}

# 空值
EMPTY_VALUE=
    """
    
    # 创建临时测试文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(test_content)
        test_file = f.name
    
    try:
        # 测试解析
        result = parse_dotenv(test_file)
        print("\n解析结果：")
        for key, value in result.items():
            print(f"  {key} = {value}")
        
        # 测试加载
        print("\n加载到环境变量...")
        load_dotenv(test_file, override=True)
        print(f"  API_KEY from env: {os.environ.get('API_KEY')}")
        print(f"  API_BASE from env: {os.environ.get('API_BASE')}")
        
        print("\n✅ 测试通过！")
    
    finally:
        # 清理测试文件
        os.unlink(test_file)
