#!/bin/bash
# Marco Search 运行脚本

cd "$(dirname "$0")"

# 主模型（Agent 推理）
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://localhost:40001/v1"
export MODEL_NAME="your-model-name"

# 搜索工具（Google via Serper）
export GOOGLE_SEARCH_KEY="your-serper-api-key"

# 可选: 使用 You.com 搜索替代 Serper（keyless 或设置 YOUCOM_API_KEY）
# - 无 API Key: 使用免费 keyless 模式（https://api.you.com/mcp?profile=free）
# - 有 API Key: export YOUCOM_API_KEY="your-youcom-api-key"
# - 切换: 在 configs/default.yaml 中将 tools.enabled 改为 [you_search, visit]
# export YOUCOM_API_KEY=""

# 网页访问工具（requests 失败时 fallback 到 Jina Reader）
export JINA_API_KEY="your-jina-api-key"

# 网页摘要模型（Visit 工具内部使用，可与主模型不同）
export SUMMARY_MODEL_OPENAI_API_KEY="your-summary-api-key"
export SUMMARY_MODEL_OPENAI_BASE_URL="your-summary-api-base"
export SUMMARY_MODEL_NAME="your-summary-model-name"

# 可用 Benchmark: gaia, xbench-DeepSearch-2510, xbench-DeepSearch-2505,
#   browsecomp_zh, browsecomp_en_sample, hle_sample

# 运行配置
BENCHMARK="gaia"
PROFILE="accuracy"                # profile: efficiency, accuracy（留空使用默认）
WORKERS=5                        # 并发数
ROLLOUT=8                         # rollout 次数
OUTPUT_DIR="output"               # 结果保存路径

# 构建命令
CMD="python -u run_benchmark.py run -b $BENCHMARK -w $WORKERS -n $ROLLOUT -o $OUTPUT_DIR"
[ -n "$PROFILE" ] && CMD="$CMD -p $PROFILE"

echo "Running: $CMD"
$CMD
