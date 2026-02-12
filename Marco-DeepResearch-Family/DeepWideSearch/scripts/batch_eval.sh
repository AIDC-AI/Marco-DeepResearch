#!/bin/bash

models=("o3-mini" "claude-sonnet-4" "gemini-2.5-pro" "qwen-max" "deepseek-r1" "deepseek-v3" "kimi-k2" "qwen3-235b-a22b" "qwen3-235b-a22b-instruct" "qwen3-32b" "gpt-4o" "owl_claude-sonnet-4" "owl_gemini-2.5-pro" "owl_gemini-2.5-pro" "smolagents_claude-sonnet-4" "smolgents_gemini-2.5-pro" "smolagents_gpt-5" "websailor_gpt-5" "websailor_claude4" "websailor_gemini-2.5-pro")
model=("gpt-4o")
for model in "${models[@]}"
do
    echo "============================"
    echo "evaluate $model begin"
    echo "============================"
    ./scripts/eval.sh $model 1
done 
