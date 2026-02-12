#!/bin/bash

llm_judge_model="gpt-4o"
trial_num=4
# Please put the model-generated responses into this folder [RESULT_PATH]; the folder structure could be found in README
RESULT_PATH="/Users/qihui/Desktop/DeepWideSearch/data_convert/result_20250918"
EVAL_RESULT_SAVE_PATH="eval_result"
evaluated_model=$1
thread_num=$2

# initialize output path
OUTPUT_PATH="eval_data/${evaluated_model}"
if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p $OUTPUT_PATH
    echo "Created directory: $OUTPUT_PATH"
else
    echo "Directory already exists: $OUTPUT_PATH"
fi
result_path="${RESULT_PATH}/${evaluated_model}"

# copy the ground-truth tables
mkdir $OUTPUT_PATH/overall_20250916_tables
cp -r data/overall_20250916_tables/* ${OUTPUT_PATH}/overall_20250916_tables

# conver the response and ground-truth into the WideSearch eval script format                                                              
python scripts/convert_output2ws_format.py \
    --query_path data/overall_20250916.jsonl \
    --result_path $result_path \
    --output_path $OUTPUT_PATH

python scripts/run_eval_batching.py \
    --query_path ${OUTPUT_PATH}/overall_20250916.jsonl \
    --answer_root_path ${OUTPUT_PATH}/overall_20250916_tables \
    --response_root $OUTPUT_PATH \
    --eval_model_config_name $llm_judge_model \
    --model_config_name $evaluated_model \
    --result_save_root $EVAL_RESULT_SAVE_PATH \
    --trial_num $trial_num \
    --thread_num $thread_num \
    --use_cache

