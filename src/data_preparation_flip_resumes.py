from pathlib import Path

# === Configuration ===
input_folder = Path("../data_binary_decisions/")           # Folder with your 20 neutral resumes
output_folder = Path("../data_resumes_gendered")  # Output folder for gendered versions
output_folder.mkdir(exist_ok=True)

# Male / Female name pairs
name_pairs = [
    ("Steven Miller", "Stephanie Miller"),
    ("James Smith", "Julia Smith"),
    ("Robert Brown", "Rachel Brown"),
    ("Michael Davis", "Michelle Davis"),
    ("William Wilson", "Wendy Wilson"),
    ("David Johnson", "Diana Johnson"),
    ("Joseph Moore", "Joanna Moore"),
    ("Thomas Taylor", "Theresa Taylor"),
    ("Charles Anderson", "Charlotte Anderson"),
    ("Daniel Thompson", "Danielle Thompson"),
    ("Andrew Clark", "Angela Clark"),
    ("Paul Lewis", "Paula Lewis"),
    ("Mark Hall", "Martha Hall"),
    ("George Young", "Georgia Young"),
    ("Kenneth King", "Kendra King"),
    ("Brian Wright", "Brianna Wright"),
    ("Edward Scott", "Emma Scott"),
    ("Ronald Green", "Renee Green"),
    ("Kevin Baker", "Kelly Baker"),
    ("Jason Adams", "Jasmine Adams"),
]

# === Process Resumes ===
resumes = sorted(input_folder.glob("*.txt"))

if len(resumes) != 20:
    print(f"⚠️ Warning: Expected 20 resumes, found {len(resumes)}")

for i, resume_file in enumerate(resumes):
    with resume_file.open("r", encoding="utf-8") as f:
        content = f.read().strip()

    male_name, female_name = name_pairs[i % len(name_pairs)]

    # Male version
    male_text = f"{male_name}\n\n{content}"
    male_filename = f"{resume_file.stem}_male.txt"
    (output_folder / male_filename).write_text(male_text, encoding="utf-8")

    # Female version
    female_text = f"{female_name}\n\n{content}"
    female_filename = f"{resume_file.stem}_female.txt"
    (output_folder / female_filename).write_text(female_text, encoding="utf-8")

print(f"✅ Created {len(resumes)*2} files in '{output_folder}/'")
