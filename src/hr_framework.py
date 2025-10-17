from pathlib import Path
from litellm import completion
from dotenv import load_dotenv
from datetime import datetime
import os
import json

#Define Paths
resume_path = Path("../data/")
result_path = Path("../experiment_results")
use_cases_path = Path("../system_prompts")


#Available Local Models(via Llama)
llama_model="ollama/llama3.2"
mistral_model = "ollama/mistral:7b"
gemini_model = "gemini/gemini-2.5-flash" #Free Tier https://aistudio.google.com/
gpt_model = "gpt-4o"

# Required Env vars
load_dotenv() # Ensure .env with API Keys is loaded
llm_env_vars = ["GEMINI_API_KEY", "OPENAI_API_KEY"]

def ask_llm(prompt, model):
    messages = prompt
    response = completion(model=model, messages=messages)
    return response['choices'][0]['message']['content']

def api_keys_exist(env_vars, debug=False):
    for env in env_vars:
        value = os.environ.get(env)
        if value is None:
            return False
    return True


def get_txt_filenames(folder_path):
    folder = Path(folder_path)
    return [f.name for f in folder.glob("*.txt")]


def safe_parse_json(text):
    """
    Try to parse JSON text safely.
    Handles cases where the model adds code fences or extra text.
    """
    try:
        # Remove markdown code fences or labels like ```json
        text = text.strip().strip('`')
        if text.lower().startswith("json"):
            text = text[4:].strip()

        # Find JSON substring (if the model wrapped it in text)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            text = text[start:end]

        return json.loads(text)
    except Exception as e:
        return {"parse_error": str(e), "raw_text": text}


def load_use_case(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Join list of strings into a single text block
    if isinstance(data["content"], list):
        data["content"] = "\n".join(data["content"])

    return data


def build_prompt(use_case_json, resume_txt, job_description):
    system_message = use_case_json
    user_message = {
        "role": "user",
        "content": (
            "Resume:\n"
            f"{resume_txt}\n\n"
            "Job Description:\n"
            f"{job_description if job_description else 'No job description provided.'}"
        )
    }
    return[system_message, user_message]


# Setup Experiment
models_under_test = [llama_model,mistral_model, gpt_model, gemini_model]
use_case_path = use_cases_path /"use_case2.json"


all_files = get_txt_filenames(folder_path=resume_path)

#Start of Test
all_answers = []
for selected_model in models_under_test:
    for resume in all_files:
        file_path = resume_path / resume
        extracted_txt = file_path.read_text(encoding="utf-8")
        built_prompt = build_prompt(
            use_case_json= load_use_case(filepath=use_case_path),
            resume_txt=extracted_txt,
            job_description="HR Recruiter")
        answer = ask_llm(prompt=built_prompt, model=selected_model)
        parsed_answer = safe_parse_json(answer)
        all_answers.append({
            "model": selected_model,
            "resume": resume,
            "answer": parsed_answer
        })

#Save the results
output_file = result_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with output_file.open("w", encoding="utf-8") as f:
    json.dump(all_answers, f, ensure_ascii=False, indent=4)





