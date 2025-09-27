import json
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq,AutoTokenizer
import os
import re
from rouge import Rouge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import json

import logging
import transformers
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--model_path', default="base_models/qwen2-5/Qwen2.5-14B-Instruct")
# parser.add_argument('--feedback_path', default='output/Llama-3.2-1B-Instruct_scoring_ua_results.json')
# parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--tensor_parallel_size', type=int, default=1)
# parser.add_argument('--max_tokens', type=int, default=32)
args = parser.parse_args(

)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token 
print("===========")
def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384 
    input_ids, attention_mask, labels = [], [], []
    inputs = f" question:{example['question']},reference answers:{example['reference_answer']},student answers:{example['provided_answer']}"
    instruction = tokenizer(
        f"<|im_start|>system\nYou are a teacher and you will receive a text containing questions, reference answers, and student answers. Please score the student answers and give scoring feedback <|im_end|>\n<|im_start|>user\n{inputs}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    score = round(float(example['score']),2)
    if score>1.0:
        score = 1.0
    output_text = f"sorce:{score},feedback:{example['answer_feedback']}" 
    response = tokenizer(output_text, add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   

def calculate_metrics(results):
    # 评分指标
    true_scores = [r["true_score"] for r in results]
    pred_scores = [r["pred_score"] for r in results]

    
    rmse = np.sqrt(mean_squared_error(true_scores, pred_scores))
    mae = mean_absolute_error(true_scores, pred_scores)
    pearson = np.corrcoef(true_scores, pred_scores)[0, 1]
    
    # 反馈指标
    rouge = Rouge()
    bleu_scores = []
    rouge_scores = {"rouge-1": [], "rouge-2": [], "rouge-l": []}
    
    for r in results:
        # BLEU
        ref = r["true_feedback"].split()
        hyp = r["pred_feedback"].split()
        bleu_scores.append(sentence_bleu([ref], hyp))
        
        # ROUGE
        scores = rouge.get_scores(r["pred_feedback"], r["true_feedback"])
        if scores:
            scores = scores[0]
            for key in rouge_scores:
                rouge_scores[key].append(scores[key]["f"])
    
    return {
        "scoring": {
            "rmse": rmse,
            "mae": mae,
            "pearson": pearson
        },
        "feedback": {
            "bleu": np.mean(bleu_scores) if bleu_scores else 0,
            "rouge-1": np.mean(rouge_scores["rouge-1"]) if rouge_scores["rouge-1"] else 0,
            "rouge-2": np.mean(rouge_scores["rouge-2"]) if rouge_scores["rouge-2"] else 0
        }
    }

DATASET_PATH = "Short-Answer-Feedback/saf_communication_networks_english"

dataset = load_dataset(DATASET_PATH)

train_dataset = dataset['train']
test_unseen_answers_dataset = dataset['test_unseen_answers']
test_unseen_questions_dataset = dataset['test_unseen_questions']

train_dataset = train_dataset.map(process_func, remove_columns=train_dataset.column_names)

model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

train_args = TrainingArguments(
    output_dir="./output/Qwen2",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
model_name = args.model_path.split('/')[-1]
trainer.model.save_pretrained(f"./output/{model_name}_final_checkpoint")
tokenizer.save_pretrained(f"./output/{model_name}_final_checkpoint")

# 新增批量预测函数
def batch_predict(model, tokenizer, messages_batch):
    """
    批量预测函数
    :param messages_batch: 列表的列表，每个内层列表是一个样本的消息字典
    """
    model.eval()
    device = model.device
    
    # 为每个样本应用聊天模板
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        for messages in messages_batch
    ]
    
    # 批量编码
    model_inputs = tokenizer(
        texts, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
        return_attention_mask=True
    ).to(device)
    
    # 生成预测
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    
    # 解码预测结果（跳过输入部分）
    responses = []
    input_lengths = model_inputs.input_ids.shape[1]
    
    for i, output_ids in enumerate(generated_ids):
        # 跳过输入部分
        output = output_ids[input_lengths:]
        response = tokenizer.decode(output, skip_special_tokens=True)
        responses.append(response)
    
    return responses

def prediction(model, tokenizer, test_dataset, batch_size,flag):
    results = []
    all_messages = []
    
    # 为每个样本准备消息
    for example in test_dataset:
        instruction = "You are a teacher and you will receive a text containing questions, reference answers, and student answers. Please score the student answers and give scoring feedback"
        input_value = f"question:{example['question']},reference answers:{example['reference_answer']},student answers:{example['provided_answer']}"
        
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_value}
        ]
        all_messages.append(messages)
    
    # 分批进行预测
    pattern = r"sorce:(\d+\.\d+),feedback:(.*)"
    
    for i in tqdm(range(0, len(all_messages), batch_size), desc="Predicting"):
        batch_messages = all_messages[i:i+batch_size]
        batch_responses = batch_predict(model, tokenizer, batch_messages)
        
        # 处理每个批次的响应
        for j, response in enumerate(batch_responses):
            idx = i + j
            if idx >= len(test_dataset):
                break
                
            example = test_dataset[idx]
            match = re.search(pattern, response)
            
            if match:
                score = match.group(1)
                pred_feedback = match.group(2)

                data = {}
                true_score = round(float(example['score']), 2)
                if true_score > 1.0:
                    true_score = 1.0

                data["true_score"] = true_score
                data["pred_score"] = float(score)
                data["true_feedback"] = example["answer_feedback"]
                data["pred_feedback"] = pred_feedback
                data['question'] = example['question']
                data["reference answers"]=example['reference_answer']
                data['student_answer'] = example['provided_answer']
                results.append(data)
    
    print(f'处理样本数: {len(results)}/{len(test_dataset)}')
    print('评分成功率：', len(results)/len(test_dataset))

    model_name = args.model_path.split('/')[-1]
    
    print("\nSaving results...")
    with open(f"./output/{model_name}_scoring_{flag}_results.json", "w",encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    metrics = calculate_metrics(results)

    with open(f"./output/{model_name}_scoring_{flag}_metrics.json", "w",encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

            
    # 打印结果
    print("\nEvaluation Metrics:")
    print("Scoring Metrics:")
    print(f"  RMSE: {metrics['scoring']['rmse']:.4f}")
    print(f"  MAE: {metrics['scoring']['mae']:.4f}")
    print(f"  Pearson Correlation: {metrics['scoring']['pearson']:.4f}")

    print("\nFeedback Metrics:")
    print(f"  BLEU: {metrics['feedback']['bleu']:.4f}")
    print(f"  ROUGE-1: {metrics['feedback']['rouge-1']:.4f}")
    print(f"  ROUGE-2: {metrics['feedback']['rouge-2']:.4f}")
    
    return metrics

print('=========test_unseen_answers_dataset=====')
prediction(model, tokenizer, test_unseen_answers_dataset,16,'ua')
print('=========test_unseen_questions_dataset=====')
prediction(model, tokenizer, test_unseen_questions_dataset,16,'uq')

