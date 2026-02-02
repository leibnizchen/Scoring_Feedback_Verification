# Scoring_Feedback_Verification
## Improving from Feedback: A Verifiable LLM Framework for Answer Scoring with Revision and Blind Evaluation

With the increasing application of artificial intelligence in educa-
tion, automated answer scoring and feedback systems have been
widely deployed in large-scale assessments and online learning plat-
forms, significantly improving evaluation efficiency and providing
personalized instructional support. However, most existing systems
remain confined to output scores and feedback. They lack an un-
derstanding of the critical process by which students learn through
feedback. This paper addresses this limitation by proposing an
integrated integrated scoring–feedback–revision–rescoring–blind-
scoring framework, which implements a comprehensive process
from initial scoring to learning improvement and objective verifi-
cation. The framework first receives the question, the student’s an-
swer, and the reference answers. A large language model fine-tuned
with LoRA generates initial scores and feedback. Subsequently,
Qwen2.5 models of varying sizes simulate students in three cate-
gories: poor, average, and excellent. Based on the feedback, multiple
revised answers are generated and re-scored by the same scoring
model to quantify the gains from the feedback. Independent blind
grading also objectively evaluates the revised answers without prior
knowledge to verify the credibility of the re-grading results. This
framework goes beyond producing a final score. It explicitly simu-
lates the feedback-driven learning process and evaluates whether
feedback leads to measurable answer improvement, laying the foun-
dation for building more reliable and adaptable intelligent tutoring
systems (ITS).

<div align="center">
<img src=".\image\F1.png" >
</div>

## Environment
conda create -n LLMScoring python=3.10

conda activate LLMScoring

pip install -r requirements.txt

## Datasets Download
SAFDataset: https://huggingface.co/datasets/Short-Answer-Feedback/saf_communication_networks_english


## Preparation
-Create a folder named output, which stores the training model weights, scoring and feedback results.

-Create three folders named Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, and Qwen2.5-7B-Instruct to store the revised answers.

-Create two folders named rescore and blindscore to output rescore and blind score respectively.

## Step 1: Lora Fine-tuning model, scoring and feedback
python lora_scoring.py --model_path base_models/Qwen2.5-14B-Instruct

## Step 2: Revising answer
python revising_answer.py --model_path base_models/Qwen2.5-7B-Instruct --feedback_path output/Qwen2.5-14B-Instruct_scoring_ua_results.json

## step 3: Re-scoring 
python rescoring.py --model_path output1/Qwen2.5-14B-Instruct_final_checkpoint  --improve_answer_path Qwen2.5-7B-Instruct//Qwen2.5-14B-Instruct_scoring_uq_results_improve_answer.json
    
## step 4: Blind scoring
python blind_scoring.py --model_path base_models/gpt-oss-120b  --improve_answer_path Qwen2.5-7B-Instruct//Qwen2.5-14B-Instruct_scoring_uq_results_improve_answer.json

## Scoring Results

<div align="center">
<img src=".\image\F2.png">
</div>

## Feedback Results

<div align="center">
<img src=".\image\F3.png">
</div>

## Re-scoring Results

<div align="center">
<img src=".\image\F4.png">
</div>

## Blind Scoring Results

<div align="center">
<img src=".\image\F5.png">
</div>

