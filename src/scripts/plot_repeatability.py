import json
from pathlib import Path
import matplotlib.pyplot as plt

result_path = Path("../../experiment_results")

# Path to your JSON file
json_file = result_path / "repeatability_uc2.20251021_080549.json"

# Read data from JSON file
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize score lists
gpt4o_scores = []
gemini_scores = []

# Extract scores
for group in data:
    for entry in group:
        model = entry.get("model")
        score = entry.get("answer", {}).get("score")
        if model == "gpt-4o":
            gpt4o_scores.append(score)
        elif model.startswith("gemini"):
            gemini_scores.append(score)

# Print results
print("GPT-4o scores:", gpt4o_scores)
print("Gemini scores:", gemini_scores)

# Plot both models’ scores
plt.figure(figsize=(8, 5))
plt.plot(range(len(gpt4o_scores)), gpt4o_scores, marker='o', color='blue', label='GPT-4o')
plt.plot(range(len(gemini_scores)), gemini_scores, marker='s', color='orange', label='Gemini')

# Add titles and labels
plt.title("Model Score Comparison")
plt.xlabel("Index")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

