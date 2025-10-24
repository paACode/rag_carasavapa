# Experiment 1: Ontology-guided Evaluation of Faithfulness in LLM-based CV Extraction

## Core interest: Can the LLM correctly identify and extract relevant information from candidate resumes given a predefined ontology?

## Step 1: Set up an ontology tailored for CVs 
	- major positions in a CV form entities (e.g., skills, work experiences, education) 
	- each entity consists of a diverse set of attributes (e.g., skill name, years of experience) 
	- see Ontology.json for full ontology

## Step 2: Select and extract 20 test CVs from a corpus of over 2'000 CVs saved in a csv file
	- Extraction was based on the CV IDs
	- each CV was saved in a separate txt-file named with the ID; 
	- additionally 20 empty json files for the gold standard annotation were created 
	- see script prepare_gold.py

## Step 3: Gold standard annotations of the 20 resumes
	- semi-automated: ChatGPT plus manual check for correctness of entities and attributes
	- see folder annotations_manual

## Step 4: Extraction of entities and attributes by Gemini
	- ontology and resumes were handed over to the LLM
	- prompt: asked Gemini to extract all entities and attributes from the CVs according to the ontology and save the model output 
	- see script gemini_extraction_evaluation.py 
	- note: only 18 json files, since Gemini ran out of capacity (free version)

## Step 4: Evaluation of the model output 
	- metric: faithfulness on the answer-level (one score per document)
	- customise G-Eval metric with prompt to evaluate, if extracted information is completely supported by the resume (score 1.0) or if there is 		contradiction or fabricated information (score 0.0) 
	- For both entity extraction and evaluation gemini was used. Assumption: similar to a new chat in ChatGpt 5, the LLM wouldn't actively connect both 	calls. 
	- for results see Gemini_G-Eval Faithfulness Summary.txt

## Step 5: Gpt-4o Model_Evaluation
	- For this part the model_outputs from Gemini were evaluated by Gpt-4o. Again here only 18 resumes respectively 18 model outputs were evaluated.
	- Step 4 applies accordingly
	- see script gpt4o_evaluation_py and Gpt4o_G-Eval Faithfulness Summary.txt

## Step 6: Final evaluation
	- Next to the evaluations done by the LLM-Judge, we compared the manually annotated JSONS with the model output mainly on a structural level 
	- did it follow the structure? Did it recognize degree abbreviation such as B.A. (not explicitly listed in the Ontology) 
	- Moreover, we chose two outputs with the score of 0.98 and 0.65 and compared them against the annotation and model output.

## Step 7: Comparison of both models
	- Findings: Mean faithfulness score was higher for GPT (0.972) than Gemini (0.946) 
	- None of the models had outliers
	- Gemini had one low score of 0.65; inspection shows inconsistencies regarding attributes of the entity work experience (e.g., mixed up dates and job titles)

# Critical reflection
	- CVs are a good test case, since they are structured but variable documents, however 20 cases is a very small sample size not allowing for generalisation
	- CVs only come from one domain (HR) --> sampling bias?
	- comparison with annotated CVs was done only qualitatively --> quantitative measures like precision, recall, F1 or consistency would be good extensions of the experiment
---

