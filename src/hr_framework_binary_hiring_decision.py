from pathlib import Path
from litellm import completion
from dotenv import load_dotenv
from datetime import datetime
import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

#Define Paths
resume_path = Path("../data/")
jd_path = Path("../jobdescriptions/")
result_path = Path("../experiment_results")
use_cases_path = Path("../system_prompts")


#Available Local Models(via Llama)
llama_model="ollama/llama3.2"
mistral_model = "ollama/mistral:7b"
gemini_model = "gemini/gemini-2.5-flash" #Free Tier https://aistudio.google.com/
gpt_model = "gpt-4o"

# Required Env vars
load_dotenv() # Ensure .env with API Keys is loaded
llm_env_vars = ["OPENAI_API_KEY"]

def ask_llm(prompt, model):
    try:
        messages = prompt
        response = completion(model=model, messages=messages)
        return response['choices'][0]['message']['content']
    except Exception as e:
        return json.dumps({"hire": 0, "reason": f"Error: {str(e)}"})

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

def run_use_case(models_under_test, use_case_path , evaluate_job, files_under_test):
    all_answers = []
    for selected_model in models_under_test:
        for resume in files_under_test:
            file_path = resume_path / resume
            extracted_txt = file_path.read_text(encoding="utf-8")
            built_prompt = build_prompt(
                use_case_json=load_use_case(filepath=use_case_path),
                resume_txt=extracted_txt,
                job_description=evaluate_job)
            answer = ask_llm(prompt=built_prompt, model=selected_model)
            parsed_answer = safe_parse_json(answer)
            all_answers.append({
                "model": selected_model,
                "resume": resume,
                "answer": parsed_answer
            })
    return all_answers

def save_result_as_json(result, name):
    # Save the results
    output_file = result_path / f"{name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def test_repeatability(model_under_test,use_case_path, evaluate_job, files_under_test , repetitions):
    all_answers = []
    for i in range(0, repetitions):
        result = run_use_case(model_under_test,use_case_path, evaluate_job, files_under_test)
        all_answers.append(result)

    return all_answers

# Human Ratings
def load_human_ratings(filepath):
    df = pd.read_csv(filepath)
    # Make sure the column is numeric
    df["Ratings combined"] = pd.to_numeric(df["Ratings combined"], errors="raise")  # will throw if any non-numeric
    # Check for NaNs right after loading
    if df["Ratings combined"].isna().any():
        raise ValueError(
            f"NaN found in 'Ratings combined' column for IDs: {df[df['Ratings combined'].isna()]['ID'].tolist()}")
    # Convert to int
    df["Ratings combined"] = df["Ratings combined"].astype(int)
    return df

def filter_by_level(human_ratings_df, level):
    """
    Filters human ratings and corresponding resume files by job level.
    """
    # Filter human ratings
    df_level = human_ratings_df[human_ratings_df["JD Level"].str.lower() == level.lower()]

    # Keep only resumes that exist in the folder
    resume_files = [
        f"resume_{row['ID']}.txt"
        for _, row in df_level.iterrows()
        if (resume_path / f"resume_{row['ID']}.txt").exists()
    ]

    return df_level, resume_files


# evaluate results
def evaluate_llm_results(llm_results, human_ratings_df):
    y_true = []
    y_pred = []
    reasoning_texts = []

    for result in llm_results:
        resume_name = result["resume"]
        answer = result["answer"]

        pred = answer.get("hire", 0)
        reason = answer.get("reason", "")

        try:
            resume_id = int(resume_name.replace("resume_", "").replace(".txt", ""))
        except Exception:
            resume_id = 0

        human_row = human_ratings_df[human_ratings_df["ID"] == resume_id]
        print(f"Resume: {resume_name}, Resume ID: {resume_id}, Human row:\n{human_row}")

        true_val = int(human_row["Ratings combined"].values[0]) if not human_row.empty else 0

        y_pred.append(pred)
        y_true.append(true_val)
        reasoning_texts.append({
            "resume": resume_name,
            "reasoning": reason
        })

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    return {
        "accuracy": acc,
        "cohen_kappa": kappa,
        "reasoning_texts": reasoning_texts
    }

def test_single_api_call_by_level(level="senior"):
    """
    Runs a single API call using a resume and job description
    that match the specified level.
    """
    print(f"=== Running single API call test for level: {level} ===")

    # Load human ratings
    human_ratings_df = load_human_ratings("../human_rating/ratings_combined_clean.csv")

    # Filter human ratings and get available resumes
    human_ratings_level, resume_files = filter_by_level(human_ratings_df, level)
    if not resume_files:
        print(f"No resumes found for level: {level}")
        return

    # Pick the first resume
    resume_file = resume_files[0]
    resume_txt = (resume_path / resume_file).read_text(encoding="utf-8")
    print(f"Selected resume: {resume_file}")  # <-- log the resume

    # Pick a job description for the same level
    jd_files = [f for f in get_txt_filenames(jd_path) if level in f.lower()]
    if not jd_files:
        print(f"No job descriptions found for level: {level}")
        return

    job_file = jd_path / jd_files[0]
    job_txt = job_file.read_text(encoding="utf-8")
    print(f"Selected job description: {jd_files[0]}")  # <-- log the JD

    # Load system prompt
    use_case_json = load_use_case(use_cases_path / "use_case_2_binary.json")

    # Build prompt
    prompt = build_prompt(use_case_json, resume_txt, job_txt)

    # Make the API call
    raw_answer = ask_llm(prompt, gpt_model)
    parsed_answer = safe_parse_json(raw_answer)

    print("Raw answer:", raw_answer)
    print("Parsed answer:", parsed_answer)

    # Evaluate against human rating for this resume
    metrics = evaluate_llm_results([{"resume": resume_file, "answer": parsed_answer}], human_ratings_level)
    print("Evaluation metrics:", metrics)
    print(f"=== Single API call test completed for level: {level} ===\n")


def main():
    # === SINGLE API CALL TEST ===
    test_single_api_call_by_level(level="senior")

    # === BATCH PROCESSING ===
"""
    # Load human ratings
    human_ratings_df = load_human_ratings("../human_rating/ratings_combined_clean.csv")
    human_ratings_df["JD Level"] = human_ratings_df["JD Level"].str.lower()  # normalize

    # Define levels to process
    levels = ["senior", "mid", "entry"]

    models_under_test = [gpt_model]
    all_results = []

    for level in levels:
        print(f"=== Processing level: {level} ===")

        # Filter resumes and human ratings by level
        human_ratings_level, resumes_level = filter_by_level(human_ratings_df, level)
        if not resumes_level:
            print(f"No resumes found for level: {level}. Skipping.")
            continue

        # Filter job descriptions by level
        jd_files = [f for f in get_txt_filenames(jd_path) if level in f.lower()]
        if not jd_files:
            print(f"No job descriptions found for level: {level}. Skipping.")
            continue

        for jd_file in jd_files:
            job_txt = (jd_path / jd_file).read_text(encoding="utf-8")
            print(f"Evaluating resumes for job description: {jd_file} (level: {level})")
            
            for resume in resumes_level:
                print(f"Processing resume: {resume} for JD: {jd_file}")

            # Run LLM for all resumes of this level
            llm_results = run_use_case(
                models_under_test=models_under_test,
                use_case_path=use_cases_path / "use_case_2_binary.json",
                evaluate_job=job_txt,
                files_under_test=resumes_level
            )

            # Evaluate results against human ratings
            metrics = evaluate_llm_results(llm_results, human_ratings_level)

            # Save results per level + JD
            save_result_as_json(result=llm_results, name=f"llm_vs_human_{level}_{Path(jd_file).stem}")

            # Append to aggregated results
            all_results.append({
                "level": level,
                "job_description": Path(jd_file).stem,
                "metrics": metrics,
                "llm_results": llm_results
            })

    # Save aggregated results for all levels
    save_result_as_json(result=all_results, name="llm_vs_human_all_levels")
    print("=== Batch evaluation completed ===")
"""

if __name__ == "__main__":
    main()





# result_uc2= run_use_case(models_under_test=[gpt_model, gemini_model],
#                          use_case_path=use_cases_path / "use_case2.json",
#                          evaluate_job="HR Recruiter",
#                          files_under_test= get_txt_filenames(folder_path=resume_path)
#                          )
# save_result_as_json(result_uc2, name = "uc2")










