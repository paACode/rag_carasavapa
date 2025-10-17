# LLM Pilot Framework in HR Domain

## Overview
This framework is a pilot setup to evaluate the capabilities of **Large Language Models (LLMs)** in recruitment-related tasks. It is modular and designed to be easily extended to additional use cases and metrics.

## Limitations
This is a pilot project that only considers **20 CVs** in the HR category, extracted from [LinkedIn Job Postings Dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings/data).

## Supported LLMs
- `llama_model = "ollama/llama3.2"`  
- `mistral_model = "ollama/mistral:7b"`  
- `gemini_model = "gemini/gemini-2.5-flash"` – Free Tier: [AI Studio](https://aistudio.google.com/)  
- `gpt_model = "gpt-4o"`  

## Use Cases

1. **Entity and Attribute Extraction from CVs**  
   - **Objective:** Evaluate whether the LLM can correctly identify entities and their attributes in a CV.  
   - **Metric:** Accuracy compared to gold standard annotations manually created by real team members.

2. **Candidate Suitability Assessment**  
   - **Objective:** Determine if the LLM can correctly evaluate how well a candidate matches a predefined job based on relevance, skills, and experience, providing a score from 1 (very poor fit) to 5 (excellent fit) along with a one-sentence justification.  
   - **Metric1 "Accuracy" :** Accuracy of the LLM’s numeric scores against manually evaluated scores by real team members.
   - **Metric2 "Validity" :** Validate manually if the stated explanation of the LLM makes sense. 

3. **Bias Evaluation**  
   - **Objective:** Assess whether ingested information, such as gender, affects the LLM's outputs and decisions.  
   - **Metric:** Bias analysis comparing model results across different demographic attributes.

## Methodology
- The LLM outputs are compared against **gold standards** and **scores** prepared by human annotators.  
- Metrics are captured for each use case to enable a thorough evaluation of model performance.

## Goals
- Provide a reproducible setup for evaluating LLMs in recruitment-related tasks.  
- Establish a foundation that can be **extended** to additional use cases, models, or evaluation metrics in the future.

## Future Extensions
- Adding new use cases (e.g., skill recommendation, candidate ranking).  
- Supporting multiple LLMs and configurations for comparative analysis.  
- Incorporating additional fairness and bias metrics.bias metrics.