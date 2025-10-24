from pathlib import Path
from litellm import completion
from dotenv import load_dotenv
from datetime import datetime
import os
import json

# ----------------- Paths -----------------
resume_path = Path("../data_resumes_gendered/")  # folder with all male/female resumes
jd_path = Path("../jobdescriptions/")           # folder with job descriptions
result_path = Path("../experiment_results")
use_cases_path = Path("../system_prompts")

# ----------------- Models -----------------
llama_model = "ollama/llama3.2"
mistral_model = "ollama/mistral:7b"
gemini_model = "gemini/gemini-2.5-flash"
gpt_model = "gpt-4o"

# ----------------- Environment -----------------
load_dotenv()  # Ensure .env with API keys is loaded
llm_env_vars = ["OPENAI_API_KEY", "GEMINI_API_KEY"]

# ----------------- Helper Functions -----------------
def ask_llm(prompt, model):
    try:
        messages = prompt
        response = completion(model=model, messages=messages)
        return response['choices'][0]['message']['content']
    except Exception as e:
        return json.dumps({"hire": 0, "reason": f"Error: {str(e)}"})

def get_txt_filenames(folder_path):
    folder = Path(folder_path)
    return [f.name for f in folder.glob("*.txt")]

def safe_parse_json(text):
    try:
        text = text.strip().strip('`')
        if text.lower().startswith("json"):
            text = text[4:].strip()
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
    if isinstance(data.get("content"), list):
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
    return [system_message, user_message]

def run_use_case(models_under_test, use_case_path, evaluate_job, files_under_test):
    all_answers = []
    use_case_json = load_use_case(use_case_path)
    for model in models_under_test:
        for resume_file in files_under_test:
            file_path = resume_path / resume_file
            resume_txt = file_path.read_text(encoding="utf-8")
            prompt = build_prompt(use_case_json, resume_txt, evaluate_job)
            answer_raw = ask_llm(prompt, model)
            parsed_answer = safe_parse_json(answer_raw)
            all_answers.append({
                "model": model,
                "resume": resume_file,
                "answer": parsed_answer
            })
    return all_answers

def save_result_as_json(result, name):
    output_file = result_path / f"{name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

# ----------------- Fit & Flip Evaluation -----------------
def evaluate_fit_and_flip(llm_results):
    results_dict = {r["resume"]: r["answer"].get("hire", 0) for r in llm_results}

    flip_count = 0
    total_pairs = 0
    fit_male = []
    fit_female = []

    # group by ID from filenames
    ids = set(f.split("_")[1] for f in results_dict.keys())
    pairwise_results = []

    for id_ in ids:
        male_file = f"resume_{id_}_male.txt"
        female_file = f"resume_{id_}_female.txt"
        if male_file in results_dict and female_file in results_dict:
            male_decision = results_dict[male_file]
            female_decision = results_dict[female_file]

            fit_male.append(male_decision)
            fit_female.append(female_decision)

            flipped = male_decision != female_decision
            if flipped:
                flip_count += 1
            total_pairs += 1

            pairwise_results.append({
                "id": id_,
                "male_file": male_file,
                "female_file": female_file,
                "male_decision": male_decision,
                "female_decision": female_decision,
                "flipped": flipped
            })

    fit_rate_male = sum(fit_male) / len(fit_male) if fit_male else 0
    fit_rate_female = sum(fit_female) / len(fit_female) if fit_female else 0
    flip_rate = flip_count / total_pairs if total_pairs else 0

    return fit_rate_male, fit_rate_female, flip_rate, pairwise_results

# ----------------- Main -----------------
# ----------------- Main -----------------
def main():
    models_under_test = [gemini_model]  # choose model
    resume_files = get_txt_filenames(resume_path)
    jd_files = get_txt_filenames(jd_path)
    use_case_file = use_cases_path / "use_case_2_binary.json"

    all_results = []
    fit_results = []

    # Track totals for overall aggregation
    all_fit_male = []
    all_fit_female = []
    all_flips = 0
    all_pairs = 0

    # Run per job description
    for jd_file in jd_files:
        job_txt = (jd_path / jd_file).read_text(encoding="utf-8")
        llm_results = run_use_case(
            models_under_test=models_under_test,
            use_case_path=use_case_file,
            evaluate_job=job_txt,
            files_under_test=resume_files
        )

        fit_male, fit_female, flip_rate, pairwise_results = evaluate_fit_and_flip(llm_results)

        print(f"Job: {jd_file}")
        print(f"Fit Rate Male: {fit_male:.2f}")
        print(f"Fit Rate Female: {fit_female:.2f}")
        print(f"Flip Rate: {flip_rate:.2f}\n")

        # Add to per-job results
        fit_results.append({
            "job_description": jd_file,
            "fit_rate_male": fit_male,
            "fit_rate_female": fit_female,
            "flip_rate": flip_rate,
            "pairwise_results": pairwise_results
        })

        all_results.extend(llm_results)

        # Aggregate for overall metrics
        all_fit_male.extend([r["male_decision"] for r in pairwise_results])
        all_fit_female.extend([r["female_decision"] for r in pairwise_results])
        all_flips += sum(1 for r in pairwise_results if r["flipped"])
        all_pairs += len(pairwise_results)

    # Compute overall metrics
    overall_fit_rate_male = sum(all_fit_male) / len(all_fit_male) if all_fit_male else 0
    overall_fit_rate_female = sum(all_fit_female) / len(all_fit_female) if all_fit_female else 0
    overall_flip_rate = all_flips / all_pairs if all_pairs else 0

    print("=== Overall Metrics ===")
    print(f"Overall Fit Rate Male: {overall_fit_rate_male:.2f}")
    print(f"Overall Fit Rate Female: {overall_fit_rate_female:.2f}")
    print(f"Overall Flip Rate: {overall_flip_rate:.2f}\n")

    # Save overall results
    save_result_as_json({
        "overall_fit_flip": fit_results,
        "all_candidates": all_results,
        "overall_metrics": {
            "fit_rate_male": overall_fit_rate_male,
            "fit_rate_female": overall_fit_rate_female,
            "flip_rate": overall_flip_rate
        }
    }, name="llm_fit_flip_results")

if __name__ == "__main__":
    main()















