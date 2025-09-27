# Scoring_Feedback_Verification
Improving from Feedback: An integrated LLM Framework for Answer Scoring and Feedback with Blind Scoring and Rescoring Verification

With the proliferation of large-scale online learning platforms and
Web-based assessment tools, AI-powered scoring systems have
become crucial for scaling education. However, most existing Webdeployed systems merely output scores and feedback, lack an understanding of the critical process by which students learn through
feedback. This paper addresses this limitation by proposing an integrated "scoring, feedback, revision, re-scoring, and blind scoring,"
which implements a comprehensive process from initial scoring to
learning improvement and objective verification. The framework
first receives the question, the student’s answer, and the reference
answers. A large language model fine-tuned with LoRA generates
initial scores and feedback. Subsequently, Qwen2.5 models of varying sizes simulate students in three categories: poor, average, and
excellent. Based on the feedback, multiple revised answers are generated and re-scored by the same scoring model to quantify the
gains from the feedback. Independent blind grading also objectively
evaluates the revised answers without prior knowledge to verify
the credibility of the re-grading results. This framework not only
provides a final evaluation score, but also simulates and evaluates
the effectiveness of the interactive process of "teaching feedbackstudent learning," laying the foundation for building more reliable
and adaptable online intelligent tutoring systems.

