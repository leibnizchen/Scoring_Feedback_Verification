import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import pandas as pd
import torch
import numpy as np
import re
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import cohen_kappa_score
import os
import transformers
import logging

# 设置 Transformers 日志级别为 ERROR（只显示错误信息）
transformers.logging.set_verbosity_error()

# 可选：同时禁用所有 Python 警告
logging.getLogger("transformers").setLevel(logging.ERROR)
# 初始化模型和分词器


from argparse import ArgumentParser

# Add params n_gpu
parser = ArgumentParser()
parser.add_argument('--model_path', default='output1/Llama-3.1-8B-Instruct_final_checkpoint')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--improve_answer_path', default='Qwen2.5-3B-Instruct/Llama-3.1-8B-Instruct_scoring_ua_results_improve_answer.json')
args = parser.parse_args()



tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 批量预测函数
def predict_batch(messages_list, model, tokenizer, batch_size=8, max_new_tokens=128):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    all_responses = []
    
    for i in range(0, len(messages_list), batch_size):
        batch_messages = messages_list[i:i+batch_size]
        
        texts = [tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) for messages in batch_messages]
        
        model_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            return_attention_mask=True
        ).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # 使用贪婪解码确保一致性
                temperature=0.7,  # 平衡创造性和一致性
                repetition_penalty=1.1  # 减少重复
            )
        
        responses = tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        all_responses.extend(responses)
    
    return all_responses

# 第一阶段：预测分数
def predict_scores(model, tokenizer, test_dataset, batch_size=8):
   
    messages_list = []
    valid_results = []  # 存储评分成功的样本

    
    # 第一阶段提示词：只要求输出分数
    base_instruction = "you are a teacher and you will receive a text containing questions, reference answers, and student answers. Please score the student answers and give scoring feedback"
    
    for example in test_dataset:

        input_value =  f"question:{example['question']}, reference answers:{example['reference_answers']}, student answers:{example['improved_answer']}"
        
        messages = [
            {"role": "system", "content": base_instruction},
            {"role": "user", "content": input_value}
        ]
        messages_list.append(messages)
    
    
    responses = predict_batch(messages_list, model, tokenizer, args.batch_size, max_new_tokens=128)
    
    # 解析分数并统计成功率
    success_count = 0
    improve_count = 0
    re_scoring_results = []
    error_value= 0.0
    for  response,example in zip(responses,test_dataset):
        # print(response)
        match = re.search(r"sorce:(\d+\.\d+)| (\d+\.\d+)", response)
        if match:
            value = match.group(1) if match.group(1) else match.group(2)
            score = float(value)
            # 确保分数在有效范围内
            if 0 <= score <= 1:
                success_count += 1
                diff_value = score - example['pred_score']
                print(score , example['pred_score'],diff_value)
                error_value += diff_value
            if 0.8<=score<=1:
                improve_count += 1

    # 计算评分成功率
    avg_error = error_value / success_count
    print("avg_error: ",avg_error)
    scoring_success_rate = success_count / len(messages_list) if len(messages_list) > 0 else 0.0
    print("nScoring Success Rate: ",scoring_success_rate)
    
    return  scoring_success_rate,improve_count,avg_error

# 主函数
def main():
    # 加载数据集
    test_dataset = []
    with open(args.improve_answer_path, "r") as f:
        results = json.load(f)
        for r in results:
            if int(r['true_score'])<0.8:
                test_dataset.append(r)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    

    scoring_success_rate,improve_count,avg_error = predict_scores(model, tokenizer, test_dataset, batch_size=16)
    

    print("impove rate:",improve_count/len(test_dataset))
    
    result_summary = {
        "dataset_len":len(test_dataset),
        "improve_count":improve_count,
        "impove_rate":improve_count/len(test_dataset),
        "avg_error":avg_error
    }
    torch.cuda.empty_cache()
    model_name = args.improve_answer_path.split("/")[0]
    results_improve_answer= args.improve_answer_path.split("/")[-1]
    rescore_model_name = args.model_path.split("/")[-1]
    results_improve_answer = results_improve_answer.split("scoring_")[-1]
    results_improve_answer = results_improve_answer.split("_results_improve_answer ")[0]
    with open(f"rescore/{rescore_model_name}_{model_name}_{results_improve_answer}_rescore.json", "w", encoding='utf-8') as f:
        json.dump(result_summary, f, indent=2)


if __name__ == "__main__":
    main()