#!/bin/bash

RETRY="false"
OUTPUT_FOLDER=outputs/output_deepsearch/browsecomp-zh/gemini-2.5-flash
DB_NAME=deepsearch_bc_zh_db_v1_gemini-2-5-flash
MAIN_MODEL_NAME=gemini-2.5-flash
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

# --input-file can be replaced with other data files under ./benchmark/
python run_deepsearch_batch_inference.py \
    --input-file ./benchmark/browsecomp-zh-decrypted.json \
    --output-dir $OUTPUT_FOLDER \
    --db-name $DB_NAME \
    --main-model-id $MAIN_MODEL_NAME \
    --tabular-model-id $TABULAR_MODEL_NAME \
    --deep-model-id $DEEP_MODEL_NAME \
    --max-workers 4 \
    --timeout-seconds 1800 \
    --start-idx 0 \
    --end-idx 300 \
    $CLEAR_DB_FLAG \
    $SKIP_COMPLETED_FLAG
