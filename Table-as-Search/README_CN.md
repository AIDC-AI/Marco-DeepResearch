# Table-as-Search: 深度与广度信息搜索智能体框架

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-论文-red.svg)](https://arxiv.org/abs/PLACEHOLDER)

<div align="center">

⭐ _**MarcoPolo 团队**_ ⭐

[_**阿里巴巴国际数字商业**_](https://aidc-ai.com)

📝 [**论文**](https://arxiv.org/abs/PLACEHOLDER) | 🤗 [**数据集**](https://huggingface.co/datasets/AIDC-AI/DeepWideSearch) | 🔧 [**框架**](https://github.com/AIDC-AI/Marco-DeepWideSearch-Agent/Table-as-Search) | 🌍 [**English**](./README.md)

</div>

---

## 📌 项目简介

**Table-as-Search** 是一个生产级的智能体框架，专为解决需要以下两方面能力的**深度与广度信息搜索任务**而设计：
- 🔍 **深度推理**：多跳检索
- 🌐 **广度规模**：跨多个实体的信息收集

该框架在具有挑战性的**深度与广度信息搜索**任务中显著优于单智能体和多智能体 ReAct 基线。该框架实现了层次化多智能体架构，包含专门用于不同搜索策略的特化智能体，适用于市场分析、竞争情报、商业开发研究等真实应用场景。

<img src="../assets/TaS-overview.png" alt="TaS Overview" width="600" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">

---

## 🎯 核心特性

### 🏗️ 架构设计

- **层次化多智能体系统**
  - 🎭 **主智能体**：编排整体任务分解与结果聚合
  - 📊 **表格搜索智能体**：处理跨多个候选实体的广度规模实体收集
  - 🔎 **深度搜索智能体**：对特定实体进行详细属性提取

- **高级工具集成**
  - 🌐 Google 搜索（支持速率限制和重试机制）
  - 📄 网页访问与内容提取（基于 Jina AI）
  - 🗄️ 数据库支持的表格管理（MongoDB）

### 🚀 性能与可靠性

- ⚡ **并行处理**：多 worker 批量推理支持
- ⏱️ **超时管理**：进程级超时控制，确保可靠执行
- 🔄 **重试机制**：失败自动重试，可配置重试限制
- 💾 **状态持久化**：基于数据库的断点续传支持
- 📊 **丰富监控**：详细的日志和进度跟踪（Rich 库）

---

## 📁 仓库结构

```
Table-as-Search/
├── run_widesearch_inference.py       # 单任务 WideSearch 推理
├── run_deepsearch_inference.py       # 单任务 DeepSearch 推理
├── run_widesearch_batch_inference.py # 批量 WideSearch（并行处理）
├── run_deepsearch_batch_inference.py # 批量 DeepSearch（并行处理）
│
├── benchmark/                        # 基准测试数据集
│   ├── widesearch.jsonl              # WideSearch 基准（160 个实例）
│   ├── gaia-text-only.jsonl          # GAIA 基准子集
│   └── browsecomp-zh-decrypted.json  # BrowseComp-zh 基准
│
├── tools/                            # 核心工具实现
│   ├── google_search_tool.py         # Google 搜索（支持速率限制）
│   ├── jina_visit.py                 # 网页访问与提取
│   ├── db_table_code_v2.py           # 数据库表格管理
│   ├── dataloader.py                 # 基准数据加载工具
│   ├── context_summary_toolcalling_agent.py  # 上下文摘要智能体
│   └── env_loader.py                 # 自定义 .env 文件加载器（许可证安全）
│
├── prompts/                          # 智能体提示词模板
│   ├── widesearch_prompts/           # WideSearch 任务提示词
│   │   ├── main_agent_prompt_v4.py
│   │   ├── tabular_search_prompt_v4.py
│   │   └── deep_search_prompt_v4.py
│   └── deepsearch_prompts/           # DeepSearch 任务提示词
│       ├── main_agent_prompt_v3_multi_condition.py
│       ├── tabular_search_agent_prompt_v3_multi_condition.py
│       └── deep_search_agent_prompt_v3_multi_condition.py
│
├── patch/                            # 自定义补丁和增强
│   ├── openai_sever_model.py        # 增强的 OpenAI 兼容模型
│   ├── monitoring.py                # 高级日志和监控
│   └── utils.py                     # 工具函数
│
├── scripts/                          # 执行脚本
│   ├── widesearch/
│   │   └── run_ws_gemini_2.5_flash.sh
│   └── deepsearch/
│       └── run_bczh_gemini_2.5_flash.sh
│
└── requirements.txt                  # Python 依赖
```

---

## 🛠️ 安装

### 前置要求

- Python 3.10 或更高版本
- MongoDB（用于数据库支持的表格管理）
- `.env` 文件中列出你自己的 API 密钥：
  - OpenAI API 密钥（或兼容的 LLM API）
  - Google 搜索 API 密钥（Serper 或类似服务）
  - Jina AI API 密钥（用于网页访问）

### 步骤 1：克隆仓库

```bash
git clone https://github.com/AIDC-AI/Marco-DeepWideSearch-Agent
cd Marco-DeepWideSearch-Agent/Table-as-Search
```

### 步骤 2：安装依赖

```bash
# 安装主要依赖
pip install -r requirements.txt
```

### 步骤 3：配置环境变量

在 `Table-as-Search` 目录中创建 `.env` 文件：

```bash
# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_BASE_URL=https://api.openai.com/v1

# Google 搜索 API 配置
SEARCH_API_KEY=your_serper_api_key_here
SEARCH_API_BASE=XXXXXX

# Jina AI 配置（用于网页访问）
JINA_KEYS_FILE=path/to/jina_keys.txt  # 可选：包含多个 Jina API 密钥的文件
```

---

## 🚀 快速开始

### 运行单任务推理

#### WideSearch 任务

```bash
python run_widesearch_inference.py \
    --query "列出 2025 年排名前 10 的编程语言及其创建者、创建年份和主要用途" \
    --instance-id "test_001" \
    --main-model-id "gpt-4o" \
    --tabular-model-id "gpt-4o" \
    --deep-model-id "gpt-4o" \
    --output-dir ./outputs/widesearch \
    --db-name widesearch_test
```

#### DeepSearch 任务（BrowseComp-zh）

```bash
python run_deepsearch_inference.py \
    --query "找出2024年诺贝尔物理学奖得主的详细信息，包括获奖理由、主要研究领域和代表性论文" \
    --instance-id "deep_001" \
    --main-model-id "gpt-4o" \
    --tabular-model-id "gpt-4o" \
    --deep-model-id "gpt-4o" \
    --output-dir ./outputs/deepsearch \
    --db-name deepsearch_test
```

### 运行批量推理

要使用并行处理在基准数据集上进行评估，请参考 [scripts](scripts) 目录下的脚本。

---

## 📊 主要参数

### 模型配置

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--main-model-id` | 主编排模型 | 必需 |
| `--tabular-model-id` | 广度规模实体搜索模型 | 与主模型相同 |
| `--deep-model-id` | 深度属性提取模型 | 与主模型相同 |

### 执行控制

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--max-workers` | 并行 worker 数量 | 1 |
| `--timeout-seconds` | 单任务超时时间（秒） | 3600 |
| `--skip-completed` | 跳过已完成的任务 | False |
| `--clear-db` | 运行前清空数据库 | False |
| `--start-idx` | 批处理起始索引 | 0 |
| `--end-idx` | 批处理结束索引 | 全部 |

### 智能体配置

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--main-max-steps` | 主智能体最大步数 | 30 |
| `--tabular-max-steps` | 表格智能体最大步数 | 20 |
| `--deep-max-steps` | 深度智能体最大步数 | 15 |
| `--max-tool-call-retries` | 工具失败最大重试次数 | 3 |

---

## 🔧 高级用法

### 使用数据库支持的表格系统

```python
from tools.db_table_code_v2 import DBTableCodeToolInterface
from pymongo import MongoClient

# 初始化 MongoDB 连接
client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGODB_DATABASE")]

# 创建表格工具
table_tool = DBTableCodeToolInterface(
    db=db,
    name_prefix="task_001",
    description="管理结构化表格以进行信息收集"
)

# 在智能体中使用
agent = ToolCallingAgent(
    model=model,
    tools=[table_tool],
    max_steps=20
)
```

## 🧩 架构详情

### 层次化多智能体工作流

```
┌─────────────────────────────────────────────────────────────┐
│                        主智能体                              │
│  • 任务分解                                                   │
│  • 子智能体编排                                               │
│  • 结果聚合                                                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────────┐ ┌──────────────────┐
│ 表格智能体        │ │ 深度智能体        │
│ • 广度搜索        │ │ • 深度推理        │
│ • 实体列表        │ │ • 属性提取        │
│ • 快速扫描        │ │                  │
└────────┬─────────┘ └────────┬─────────┘
         │                    │
         └────────┬───────────┘
                  ▼
         ┌────────────────────┐
         │    工具生态系统     │
         │  • Google 搜索     │
         │  • 网页访问         │
         │  • 表格管理器       │
         │  • 摘要生成         │
         └────────────────────┘
```

### 智能体职责

#### 🎭 主智能体
- 分析用户查询以确定搜索范围
- 决定何时使用表格搜索或深度搜索策略
- 管理子智能体之间的信息流
- 将结果聚合为最终的结构化输出

#### 📊 表格搜索智能体
- 识别并收集候选实体的完整列表
- 执行广泛、浅层的搜索以最大化覆盖范围
- 针对召回率而非精确率进行优化

#### 🔎 深度搜索智能体
- 提取特定实体的详细属性
- 执行多跳推理以查找复杂信息
- 针对精确率和完整性进行优化

---

## 📚 相关项目

本框架是 **Marco Search Agent** 项目系列的一部分：

- 📊 **[DeepWideSearch](../DeepWideSearch)**：基准测试与评估指标
- 🏷️ **[HSCodeComp](../HSCodeComp)**：层次化规则应用基准测试
- 🤖 **[our_smolagents](./our_smolagents)**：自定义智能体框架（基于 HuggingFace smolagents 修改）

---

## 🤝 贡献

我们欢迎在以下方面的贡献：

- 🔧 **工具实现**：添加新的搜索引擎、数据源或提取工具
- 🎯 **智能体策略**：改进任务分解和编排逻辑
- 📊 **评估**：添加新的基准测试或评估指标
- 🐛 **Bug 修复**：报告问题或提交修复
- 📖 **文档**：改进指南和示例

---

## 🛡️ 许可证

本项目采用 **Apache-2.0 许可证**。

### 第三方组件

- **our_smolagents**：修改自 [HuggingFace smolagents](https://github.com/huggingface/smolagents)（Apache-2.0）
- **Jina AI**：通过 Jina Reader API 进行网页内容提取
- **MongoDB**：数据库后端（Server Side Public License）

---

## 🙏 致谢

本项目基于优秀的开源工作：

- 🤗 **HuggingFace smolagents**：核心智能体框架基础
- 🌐 **Jina AI**：网页内容提取和阅读器 API

我们感谢相关团队对开源社区的贡献。

---

## 📧 联系方式

如有问题、建议或合作意向：

- 👨‍💻 **兰天**：[GitHub](https://github.com/gmftbyGMFTBY)
- 👨‍🔬 **王龙跃**：[个人主页](https://www.longyuewang.com/)

---

## ⚠️ 免责声明

本框架设计用于研究和评估目的。在生产环境使用时：

- 🔒 确保 API 密钥的适当管理和安全性
- 💰 监控 API 使用量和成本
- 🌐 遵守网站服务条款和 robots.txt
- 📊 在下游使用前验证输出质量
- ⚖️ 遵守所在地区的数据隐私法规

数据集和基准测试可能包含公开可访问的网页数据。如果您认为任何内容侵犯了您的权利，请及时联系我们。

---

## 📊 引用格式

如果您在研究中使用本框架，请引用 🤗：

```bibtex
```
