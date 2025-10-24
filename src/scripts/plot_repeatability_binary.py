import json
from pathlib import Path
import matplotlib.pyplot as plt

result_path = Path("../../experiment_results")

# Path to your JSON file
json_file = result_path / "repeatability_uc2.20251024_072159.json"

# Read data from JSON file
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize lists for True/False answers
gpt4o_hires = []
gemini_hires = []

# Extract hire decisions
for group in data:
    for entry in group:
        model = entry.get("model")
        hire_decision = entry.get("answer", {}).get("hire")
        if model == "gpt-4o":
            gpt4o_hires.append(hire_decision)
        elif model.startswith("gemini"):
            gemini_hires.append(hire_decision)

# Print results
print("GPT-4o hire decisions:", gpt4o_hires)
print("Gemini hire decisions:", gemini_hires)

# Convert booleans to integers for plotting (True=1, False=0)
gpt4o_numeric = [1 if h else 0 for h in gpt4o_hires]
gemini_numeric = [1 if h else 0 for h in gemini_hires]

# Plot both models’ True/False results
plt.figure(figsize=(8, 5))
plt.plot(range(len(gpt4o_numeric)), gpt4o_numeric, marker='o', color='blue', label='GPT-4o')
plt.plot(range(len(gemini_numeric)), gemini_numeric, marker='s', color='orange', label='Gemini')

# Add titles and labels
plt.title("Repeatability (Hire Decisions)")
plt.xlabel("Index")
plt.ylabel("Hire Decision (True=1, False=0)")
plt.yticks([0, 1], ['False', 'True'])
plt.legend()
plt.grid(True)
plt.show()