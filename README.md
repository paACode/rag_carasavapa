# LLM Pilot Framework in HR Domain

## Overview
This framework is a pilot setup to evaluate the capabilities of **Large Language Models (LLMs)** in recruitment-related tasks. It is modular and designed to be easily extended to additional use cases and evaluation metrics.

## Limitations
This pilot project only considers **20 CVs** in the HR domain, extracted from the [LinkedIn Job Postings Dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings/data).
The main reason for this limitation is that manual screening and entity extraction are highly time-consuming tasks.
Using a larger dataset would require excessive manual effort, which is beyond the scope of a student project that primarily aims to explore and evaluate AI-based approaches, not to perform manual HR work.
## Supported LLMs
- `llama_model = "ollama/llama3.2"`  
- `mistral_model = "ollama/mistral:7b"`  
- `gemini_model = "gemini/gemini-2.5-flash"` – Free Tier: [AI Studio](https://aistudio.google.com/)  
- `gpt_model = "gpt-4o"`  

> To use local LLMs with Ollama, run: `src/bash/start_llama.api.sh`.

## Use Cases

### 1. Ontology-guided Evaluation of Faithfulness in LLM-based CV Extraction
**Goal:** Evaluate whether the LLM can correctly identify and extract relevant information from candidate resumes according to a predefined ontology.  

**Example Tasks:**  
- Extract skills, years of experience, education, job titles, industries, or achievements.  

**Metrics:**  
- Faithfulness (evaluated via LLM-as-a-judge, e.g., G-Eval)  

> Note: Not implemented on this branch.
> Please consult: https://github.com/paACode/rag_carasavapa/tree/LLM-Assited-Resume-Screening-and-Candidate-Evaluation-

### 2. Binary Hiring Decision
**Goal:** Assess whether the LLM can support the screening process of resumes.  

**Example Tasks:**  
- Classify candidates as "suitable" or "not suitable" and provide an explanation.  

**Metrics:**  
- Agreement with human raters (Cohen's kappa)  
- Accuracy

### 3. Fairness & Bias
**Goal:** Evaluate whether model outputs are fair and unbiased across demographic groups (e.g., gender).  

**Example Tasks:**  
- Compare model predictions for candidates of different genders.  
- Create counterfactual candidate pairs that differ only in one protected attribute and observe if the decision changes.  

**Metrics:**  
- Fit Rate  
- Flip Rate

## Methodology
- LLM outputs are compared against **human-annotated scores** stored in the `human_rating` folder.  
- Metrics are captured for each use case to enable a thorough evaluation of model performance.

## Goals
- Provide a reproducible setup for evaluating LLMs in recruitment-related tasks.  
- Establish a foundation that can be **extended** to new use cases, models, or evaluation metrics.

## Future Extensions
- Adding new use cases (e.g., skill recommendation, candidate ranking).  
- Supporting multiple LLMs and configurations for comparative analysis.  
- Incorporating additional fairness and bias metrics.
