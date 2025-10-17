from pathlib import Path
from litellm import completion
from dotenv import load_dotenv
from datetime import datetime
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


def build_decision_prompt_with_reason(resume_txt, job_description):
    """
    Prompt for LLM to rate candidate-job fit and provide a short justification.
    """
    system_message = {
        "role": "system",
        "content": (
            "You are an experienced hiring manager evaluating how well a candidate's resume matches a given job description.\n"
            "Your entire response must be a single valid JSON object — nothing else.\n\n"
            "Output format (required):\n"
            '{"score": <integer from 1 to 5>, "reason": "<short one-sentence explanation>"}\n\n'
            "Rules:\n"
            "- Do NOT include any text before or after the JSON.\n"
            "- Do NOT include markdown formatting (no ```json or ``` blocks).\n"
            "- Do NOT include comments or explanations outside the JSON.\n"
            "- Do NOT apologize or restate instructions.\n"
            "- The output must be parseable JSON (no trailing commas, no extra text).\n\n"
            "Scoring criteria:\n"
            "1 = Very poor fit\n"
            "2 = Weak fit\n"
            "3 = Moderate fit\n"
            "4 = Strong fit\n"
            "5 = Excellent fit"
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
models_under_test = [gpt_model]
resume_path = Path("../data_2/")
result_path = Path("../experiment_results")
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
        parsed_answer = safe_parse_json(answer)
        all_answers.append({
            "model": selected_model,
            "resume": resume,
            "answer": parsed_answer
        })

output_file = result_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with output_file.open("w", encoding="utf-8") as f:
    json.dump(all_answers, f, ensure_ascii=False, indent=4)





