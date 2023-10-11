# Exploring  Prompting Large Language Models as Explainable Metrics
This paper describes the IUST NLP Lab submission to the Prompting Large Language Models as Explainable Metrics Shared Task at the Eval4NLP 2023  Workshop on Evaluation \& Comparison of NLP Systems. We have proposed a zero-shot prompt-based strategy for explainable evaluation of the summarization task using Large Language Models (LLMs). The conducted experiments demonstrate the promising potential of LLMs as evaluation metrics in Natural Language Processing (NLP), particularly in the field of summarization. Both few-shot and zero-shot approaches are employed in these experiments. The performance of our best provided prompts achieved a Kendall correlation of 0.477 with human evaluations in the text summarization task on the test data.


# Run

## Install the required packages
```bash
pip install -r requirements.txt
```

## Run All Experiment
to run best performance prompt on test set:
```bash
python apply_prompt_evaluation.py
```