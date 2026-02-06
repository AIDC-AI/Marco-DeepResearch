#!/bin/bash

RETRY="false"
OUTPUT_FOLDER=outputs/output_widesearch/gemini-2.5-flash
DB_NAME=widesearch_db_v1_gemini-2-5-flash
MAIN_MODEL_NAME=gemini-2.5-flash
# thinking mode efficient sub-agent models support qwen3-235b-a22b and qwen3-next-80b-a3b-thinking
TABULAR_MODEL_NAME=gemini-2.5-flash
DEEP_MODEL_NAME=gemini-2.5-flash

# 根据 RETRY 标签设置参数
if [ "$RETRY" = "true" ]; then
    # RETRY="try": 清空数据库，删除输出文件夹，禁用跳过已完成
    rm -rf $OUTPUT_FOLDER
    CLEAR_DB_FLAG="--clear-db"
    SKIP_COMPLETED_FLAG="--no-skip-completed"
elif [ "$RETRY" = "false" ]; then
    # RETRY="false": 维持数据库，不删除输出文件夹，启用跳过已完成
    CLEAR_DB_FLAG="--no-clear-db"
    SKIP_COMPLETED_FLAG="--skip-completed"
else
    # 默认行为：维持数据库，启用跳过已完成
    CLEAR_DB_FLAG="--no-clear-db"
    SKIP_COMPLETED_FLAG="--skip-completed"
fi

python run_widesearch_batch_inference.py \
    --input-file ./benchmark/widesearch/widesearch.jsonl \
    --output-dir $OUTPUT_FOLDER \
    --db-name $DB_NAME \
    --main-model-id $MAIN_MODEL_NAME \
    --tabular-model-id $TABULAR_MODEL_NAME \
    --deep-model-id $DEEP_MODEL_NAME \
    --max-workers 4 \
    --timeout-seconds 3600 \
    --start-idx 0 \
    --end-idx 200 \
    $CLEAR_DB_FLAG \
    $SKIP_COMPLETED_FLAG
