import os
import json
import re
import glob
import argparse
import pandas as pd
from tqdm import tqdm

def build_prompt(messages, expected=None):
    prompt = {"messages": messages}
    if expected:
        prompt["expected"] = expected
    return prompt

def format_messages(job_title, contract_type, location, country, experience_required, salary_val):
    system_message = (
        "Task: Salary Estimation\n"
        f"Schema: Given the job details, estimate the salary for the job.\n"
        "Regulations: Respond ONLY with the salary in the format 'X triệu'."
    )
    user_message = (
        f"Job Title: {job_title}\n"
        f"Contract Type: {contract_type}\n"
        f"Location: {location}\n"
        f"Country: {country}\n"
        f"Experience Required: {experience_required}"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": salary_val}
    ]

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    # Filter out rows with 'Thoả Thuận' salary
    df = df[df['salary_avg'].str.lower() != 'thoả thuận']
    processed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing"):
        job_title = str(row.get("job_title", "")).strip()
        contract_type = str(row.get("job_type", "")).strip()
        location = str(row.get("city", "")).strip()
        country = "việt nam"
        experience_required = str(row.get("experience", "")).strip()
        salary_val = re.sub(r"[_\-\+]", " ", row.get("salary_avg", "").strip().lower())
        if not all([job_title, contract_type, location, country, experience_required, salary_val]):
            continue
            
        messages = format_messages(job_title, contract_type, location, country, experience_required, salary_val)
        processed.append(build_prompt(messages))

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(processed)} prompts to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Input CSV file", required=True)
    parser.add_argument("--output_file", type=str, help="Output JSON file", required=True)
    args = parser.parse_args()

    process_csv(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
