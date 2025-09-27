import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import json
import pandas as pd
import numpy as np
import re
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
from sklearn.metrics import mean_absolute_error

import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sklearn.metrics import cohen_kappa_score


from argparse import ArgumentParser

import transformers
import logging

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--model_path', default="base_models/Qwen2.5-1.5B-Instruct")
parser.add_argument('--feedback_path', default='output/Qwen2.5-14B-Instruct_scoring_ua_results.json')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--tensor_parallel_size', type=int, default=2)
parser.add_argument('--max_tokens', type=int, default=512)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True,legacy=False)
tokenizer.pad_token = tokenizer.eos_token

# 初始化 vLLM 模型
llm = LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    tensor_parallel_size=args.tensor_parallel_size,  # 根据 GPU 数量调整
    dtype="bfloat16",  # 使用 bfloat16 精度
    max_model_len=2048,  # 最大模型长度
    trust_remote_code=True
)


def predict_batch_vllm(prompts, sampling_params, batch_size=32):
    """使用 vLLM 批量生成响应"""
    all_responses = []
    
    # 分批处理提示
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # 使用 vLLM 生成响应
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # 提取生成的文本
        batch_responses = [output.outputs[0].text for output in outputs]
        all_responses.extend(batch_responses)
    
    return all_responses

# 单阶段：同时预测分数和生成反馈
def improve_answer_from_feedback(test_dataset, batch_size=32):
    prompts = []  # 存储所有提示
    results = []  # 存储所有结果
    responses_sample = []  # 存储所有生成的文本
    
    # 提示词：要求同时输出分数和反馈
    base_instruction = (
        "You are a student and you will receive a text containing the your answer, and feedback on your answer. Please improve your answer based on the feedback."
        "Output in the following format:\n"
        "Improved answer: [your improved answer]\n"
    )
    
    # 构建提示列表
    for example in test_dataset:
        input_value =  f"your answers:{example['student_answer']}, feedback:{example['pred_feedback']}"
        
        # 使用分词器构建聊天模板
        messages = [
            {"role": "system", "content": base_instruction},
            {"role": "user", "content": input_value}
        ]
        
        # 应用聊天模板生成完整提示
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
        
        # 初始化结果条目
        results.append({

            "question": example['question'],
            "reference_answers":example['reference answers'],
            "student_answer": example['student_answer'],
            "true_feedback": example['true_feedback'],
            "pred_feedback": example["pred_feedback"],
            "true_score": example['true_score'],
            "pred_score": example['pred_score'],
            "improved_answer": None
        })
    # 配置 vLLM 采样参数
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,  # 最大生成token数
        temperature=0,  # 温度参数
        top_p=0.95,       # top-p采样
        stop=tokenizer.eos_token,  # 停止token
        repetition_penalty=1.1,  # 重复惩罚
        skip_special_tokens=True  # 跳过特殊token
    )
    

    responses = predict_batch_vllm(prompts, sampling_params, batch_size)

    for i, response in enumerate(responses):
      
        improve_answer_match = re.search(r"Improved answer:\s*(.*?)(?:\n|$)", response,re.IGNORECASE)
       
        if improve_answer_match:
            improve_answer_match = improve_answer_match.group(1).strip()
            results[i]["improved_answer"] = improve_answer_match
        else:
            results[i]["improved_answer"] = response
    
    return results



# 主函数
def main():
    # 加载数据集
    test_dataset =[]
    with open(args.feedback_path, "r",encoding='utf-8') as f:
        dataset = json.load(f)
        for data in dataset:
            if float(data['true_score'])<0.8:
                test_dataset.append(data)
    
    improve_answer= improve_answer_from_feedback(
        test_dataset, batch_size=args.batch_size  # 使用更大的批处理大小
    )
    scoring_model = args.model_path.split("/")[-1]
    model_name = args.feedback_path.split("/")[-1]
    model_name = model_name.split(".json")[0]
    with open(f"{scoring_model}/{model_name}_improve_answer.json", "w",encoding='utf-8') as f:
        json.dump(improve_answer, f, indent=2)


if __name__ == "__main__":
    main()