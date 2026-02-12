import json
import os
from pathlib import Path
import ipdb
from argparse import ArgumentParser

def process_query_file(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line.strip())
            # 只保留指定字段，并重命名question为query
            processed = {
                "instance_id": data["instance_id"],
                "query": data["question"],
                "evaluation": data["evaluation"],
                "language": data["language"],
                "entity": data['entity'],
                "language": data['language'],
                'topic': data['topic']
            }
            f_out.write(json.dumps(processed, ensure_ascii=False) + '\n')

def process_response_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.endswith('.jsonl'):
            continue
        if not any(suffix in filename for suffix in ['iter1', 'iter2', 'iter3', 'iter4']):
            continue
        input_path = os.path.join(input_dir, filename)
        with open(input_path, 'r', encoding='utf-8') as f_in:
            
            for line in f_in:
                data = json.loads(line.strip())
                
                # 确保messages存在且非空
                if not data.get("messages") or not isinstance(data["messages"], list) or len(data["messages"]) == 0:
                    continue
                    
                # 提取最后一个message的content作为response
                last_message = data["messages"][-1]
                response_content = last_message.get("content", "")
                processed = {
                    "instance_id": data["instance_id"].replace('/', '_'),
                    "response": response_content,
                    "messages": data["messages"],
                    "trial_idx": data["rollout_id"]
                }
                output_filename = f"{processed["instance_id"]}_{processed["trial_idx"]}_response.jsonl"
                with open(os.path.join(output_dir, output_filename), 'w') as f:
                    json.dump(processed, f, ensure_ascii=False)

def main():
    if os.path.exists(args.output_path) is False:
        os.makedirs(args.output_path)

    # 配置路径
    query_input = args.query_path
    basename = os.path.basename(query_input)
    query_output = f"{args.output_path}/{basename}"
    
    response_input_dir = args.result_path
    basename = os.path.basename(response_input_dir)
    response_output_dir = f"{args.output_path}/{basename}"
    
    # 处理query文件
    print(f"Processing query file from {query_input} to {query_output}")
    process_query_file(query_input, query_output)
    
    # 处理response文件
    print(f"Processing response files from {response_input_dir} to {response_output_dir}")
    process_response_files(response_input_dir, response_output_dir)
    
    print("Data processing completed successfully!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--query_path", type=str, default="data_20250910/deep2wide_20250911_postprocessed.jsonl")
    parser.add_argument("--result_path", type=str, default="result_20250910/websailor_gemini2.5")
    parser.add_argument("--output_path", type=str, default="../data")
    args = parser.parse_args()
    main()