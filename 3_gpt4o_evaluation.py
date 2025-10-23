from litellm import completion
from dotenv import load_dotenv
import os
import json

#Available Local Models(via Llama)
#llama_model="ollama/llama3.2"
#mistral_model = "ollama/mistral:7b"
#gemini_model = "gemini/gemini-2.5-flash" #Free Tier https://aistudio.google.com/
gpt_model = "gpt-4o"

# Required Env vars
load_dotenv(dotenv_path=os.path.join("env", "env"))  # Ensure .env with API Keys is loaded
llm_env_vars = ["GEMINI_API_KEY", "OPENAI_API_KEY"]

#Enable Paid LLMS
use_proprietary_llms = True
use_llama_llms = False


def ask_llm(prompt, model):
    messages = [{"role": "user", "content": prompt}]
    response = completion(model=model, messages=messages)
    return response['choices'][0]['message']['content']


def api_keys_exist(env_vars, debug=False):

    for env in env_vars:
        value = os.environ.get(env)
        if value is None:
            return False
    return True


# evaluation

# Load API key
load_dotenv(dotenv_path=os.path.join("env", "env"))

model = "gpt-4o" # for security

resumes_dir = "resume_texts"
pred_dir = "model_output"
results = []


# Ask Gpt_4o to judge faithfulness

def eval_faithfulness(resume_text, model_output):
    prompt = f"""
You are an impartial evaluator assessing the factual faithfulness of an extracted resume summary.

Compare the extracted JSON information (the model output) to the source resume text. 
Rate how accurate and truthful the extracted facts are, according to the resume. 
Give a score between 0 and 1, where:
- 1.0 = All extracted information is completely supported by the resume
- 0.0 = The information contradicts or fabricates facts from the resume

Resume Text:
{resume_text}

Extracted Information (JSON):
{json.dumps(model_output, indent=2)}

Now respond ONLY with a numeric score between 0 and 1.
    """

    response = completion(model=model, messages=[{"role": "user", "content": prompt}])
    score_str = response["choices"][0]["message"]["content"].strip()

    try:
        score = float(score_str)
    except ValueError:
        score = 0.0
    return min(max(score, 0.0), 1.0)

# Loop through model outputs

for file in os.listdir(pred_dir):
    if not file.endswith(".json"):
        continue

    resume_id = file.replace("model_output_", "").replace(".json", "")
    resume_path = os.path.join(resumes_dir, f"resume_{resume_id}.txt")
    pred_path = os.path.join(pred_dir, file)

    # Read files
    with open(resume_path, "r", encoding="utf-8") as f:
        resume_text = f.read()
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_json = json.load(f)

    # Get Gpt's faithfulness score
    score = eval_faithfulness(resume_text, pred_json)
    results.append((resume_id, score))
    #print(f"{resume_id}: {score:.3f}")


# Summary

if results:
    avg = sum(s for _, s in results) / len(results)
    print("\nG-Eval Faithfulness Summary")
    for resume_id, s in results:
        print(f"  {resume_id}: {s:.3f}")
    print(f"\nAverage Faithfulness Score: {avg:.3f}")
else:
    print("No model outputs found.")

# # save as a file
# with open("faithfulness_summary.txt", "w", encoding="utf-8") as f:
#     f.write(f"G-Eval Faithfulness Summary\n\n")
#     for resume_id, s in results:
#         f.write(f"{resume_id}: {s:.3f}\n")
#     f.write(f"\nAverage Faithfulness Score: {avg:.3f}\n")

