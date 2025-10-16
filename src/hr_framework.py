from pathlib import Path
from litellm import completion
from dotenv import load_dotenv
import os
import json


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


def build_decision_prompt_with_reason(resume_txt, job_description):
    """
    Prompt for LLM to rate candidate-job fit and provide a short justification.
    """
    system_message = {
        "role": "system",
        "content": (
            "You are an experienced hiring manager evaluating candidates. "
            "Your task is to assess how well a candidate's resume matches a given job description. "
            "Provide two outputs:\n\n"
            "1. A score from 1 to 5 indicating the match (see criteria below).\n"
            "2. A one-sentence explanation of your reasoning.\n\n"
            "Scoring criteria:\n"
            "1 = Very poor fit\n"
            "2 = Weak fit\n"
            "3 = Moderate fit\n"
            "4 = Strong fit\n"
            "5 = Excellent fit\n\n"
            
            "Respond strictly in JSON format like this:\n"
            '{"score": <1-5>, "reason": "<short reason>"}'
        )
    }
    user_message = {
        "role": "user",
        "content": (
            "Please evaluate the following candidate resume against the job description.\n\n"
            "Resume:\n"
            f"{resume_txt}\n\n"
            "Job Description:\n"
            f"{job_description if job_description else 'No job description provided.'}"
        )
    }
    return[system_message, user_message]

# Basic Settings
models_under_test = [llama_model]
resume_path = Path("../data_2/")
all_files = get_txt_filenames(folder_path=resume_path)

#Start of Test
all_answers = []
for selected_model in models_under_test:
    for resume in all_files:
        file_path = resume_path / resume
        extracted_txt = file_path.read_text(encoding="utf-8")
        built_prompt = build_decision_prompt_with_reason(
            resume_txt=extracted_txt,
            job_description="HR Recruiter")
        answer = ask_llm(prompt=built_prompt, model=selected_model)
        all_answers.append({
            "model": selected_model,
            "resume": resume,
            "raw_answer": answer
        })

print(all_answers)



