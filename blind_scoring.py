import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
parser.add_argument('--model_path', default='/mnt/pfs/zitao_team/leibniz/Grading_Feedback/base_models/qwen2-5/Qwen2.5-72B-Instruct')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--improve_answer_path', default='Qwen2.5-3B-Instruct/Llama-3.1-8B-Instruct_scoring_uq_results_improve_answer.json')
args = parser.parse_args()



tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 批量预测函数
def predict_batch(messages_list, model, tokenizer, batch_size=8, max_new_tokens=128):
    model.eval()
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
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
        ).to(model.device)
        
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
    #base_instruction = "You are a professional Q&A judge assistant. Based on the user's question and two possible answers (A and B), choose the better answer. Finally, simply output A or B without any additional content."
    
    for example in test_dataset:

        # input_value =  f"Question: {example['question']}\nAnswer A: {example['improved_answer']}\nAnswer B: {example['student_answer']}."
        
        messages = [
            {"role": "system", "content": "You are a professional Q&A judge assistant. Evaluate the provided Question and the two candidate Answers (A and B) based on criteria such as relevance, accuracy, completeness, and clarity. Then, No additional text or explanation is needed, just outputting the better answer by choosing A or B"},

            {"role": "user", "content": "Question: What is the primary purpose of the TCP protocol in computer networking?\nAnswer_A: TCP (Transmission Control Protocol) is a core protocol of the Internet protocol suite. Its primary purpose is to provide reliable, ordered, and error-checked delivery of a stream of bytes between applications running on hosts communicating over an IP network. It manages sequencing, retransmission of lost packets, and flow control.\nAnswer_B: TCP is used for sending data on the internet. It makes sure data gets to the other side, unlike UDP which is faster but not reliable."},
            {"role": "assistant", "content": "choose A"},

            {"role": "user", "content": f"Question: {example['question']}\nAnswer_A: {example['improved_answer']}\nAnswer_B: {example['student_answer']}"}
        ]
        messages_list.append(messages)
    
    
    responses = predict_batch(messages_list, model, tokenizer, args.batch_size, max_new_tokens=256)
    
    # 解析分数并统计成功率
    success_count = 0
    A_count = 0
    re_scoring_results = []
    for  response,example in zip(responses,test_dataset):
        print(response)
        print('======')
        pattern = r'(?:preference|output|choose|assistantfinal|So).*?([A])'
        match = re.findall(pattern, response, re.IGNORECASE)
        if match:
            success_count += 1
            if match[0] == 'A': 
                A_count += 1
    
    # 计算评分成功率
    rate_a = A_count / success_count
    print("rate_a: ",rate_a)
    scoring_success_rate = success_count / len(messages_list) if len(messages_list) > 0 else 0.0
    print("nScoring Success Rate: ",scoring_success_rate)
    
    return  scoring_success_rate,A_count,rate_a

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
    )
    

    scoring_success_rate,A_count,rate_a = predict_scores(model, tokenizer, test_dataset, batch_size=16)
    

    print("A_count:",A_count/len(test_dataset))
    
    result_summary = {
        "dataset_len":len(test_dataset),
        "A_count":A_count,
        "A_rate":A_count/len(test_dataset),
        "rate_a":rate_a
    }
    torch.cuda.empty_cache()

    model_name = args.improve_answer_path.split("/")[0]
    results_improve_answer= args.improve_answer_path.split("/")[-1]

    results_improve_answer = results_improve_answer.split("_results_improve_answer ")[0]
    with open(f"blindscore/{model_name}_{results_improve_answer}_blind_score.json", "w", encoding='utf-8') as f:
        json.dump(result_summary, f, indent=2)


if __name__ == "__main__":
    main()