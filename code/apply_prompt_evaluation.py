from prompt_class import DirectAssessment
from model_dict import load_from_catalogue
from tqdm import tqdm
import pandas as pd
import torch
import csv


modelname = "psmathur/orca_mini_v3_7b"
model, tokenizer, u_prompt, a_prompt = load_from_catalogue(modelname)
BPG = None

files = {
    "sum":"../data/summarization/summarization_test_set.tsv"
    }

for key, file in files.items():
    df = pd.read_csv(file, sep="\t", quoting=csv.QUOTE_NONE)
    scores = []
    prompts = []
    explanations = []

    if key =="sum":
        mt = False
    else:
        mt = True

    cnt = 0
    for s, h in tqdm(df[["SRC","TGT"]].values.tolist(), desc=key + " progress: "):

        if BPG:
            del BPG
        BPG = DirectAssessment(model=model, tokenizer=tokenizer)

        _, score, prompt, explanation = BPG.prompt_model(
            gt=s,
            hyp=h,
            mt=mt,
            prompt_placeholder=u_prompt,
            response_placeholder=a_prompt,
            target_lang= "English" if key == "zh" else "German",
            source_lang= "Chinese" if key == "zh" else "English", 
            verbose=False
        )
        prompts.append(prompt)
        scores.append(score)
        explanations.append(explanation)
        cnt+=1

    df["eval_result"] = scores
    df["eval_result"].to_csv(f'../result/summarization.scores', header=False,index=False)
    df["prompt"] = prompts
    df["prompt"].to_csv(f'../result/summarization.prompts', header=False,index=False)
    df["explanations"] = explanations
    df["explanations"].to_csv(f'../result/summarization.explanations', header=False,index=False)
    
    


