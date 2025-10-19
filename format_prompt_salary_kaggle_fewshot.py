import os
import json
import re
import glob
import argparse
import pandas as pd
from tqdm import tqdm

def load_csv_file(path):
    """Load CSV expecting 'text' and 'salary' columns."""
    data = []
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        salary = str(row.get("salary", "")).strip()
        if text and salary:
            data.append({"text": text, "salary": salary})
    print(f"Loaded {len(data)} rows from {os.path.basename(path)}")
    return data

def build_prompt(messages, expected=None):
    prompt = {"messages": messages}
    if expected:
        prompt["expected"] = expected
    return prompt

def format_messages(job_title, contract_type, location, country, experience_required, salary_val, fewshot_examples):
    system_message = (
        "Task: Salary Estimation\n"
        f"Schema: Given the job details, estimate the salary for the job.\n"
        "Regulations: Respond ONLY with the salary in the format 'X triệu'."
    )
    
    messages = [{"role": "system", "content": system_message}]
    
    # Add few-shot examples first
    for fs in fewshot_examples:
        messages.append({"role": "user", "content": fs['text']})
        messages.append({"role": "assistant", "content": fs['salary']})
    
    # Add the actual example
    user_message = (
        f"Job Title: {job_title}\n"
        f"Contract Type: {contract_type}\n"
        f"Location: {location}\n"
        f"Country: {country}\n"
        f"Experience Required: {experience_required}"
    )
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": salary_val})
    
    return messages

def process_csv(input_file, fewshot_file, output_file):
    df = pd.read_csv(input_file)
    # Filter out rows with 'Thoả Thuận' salary
    df = df[df['salary_avg'].str.lower() != 'thoả thuận']
    processed = []
    
    # Load few-shot examples
    fewshot_examples = load_csv_file(fewshot_file)
    print(f"Using {len(fewshot_examples)} few-shot examples")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing"):
        job_title = str(row.get("job_title", "")).strip()
        contract_type = str(row.get("job_type", "")).strip()
        location = str(row.get("city", "")).strip()
        country = "việt nam"
        experience_required = str(row.get("experience", "")).strip()
        salary_val = re.sub(r"[_\-\+]", " ", row.get("salary_avg", "").strip().lower())
        if not all([job_title, contract_type, location, country, experience_required, salary_val]):
            continue
            
        messages = format_messages(job_title, contract_type, location, country, experience_required, salary_val, fewshot_examples)
        processed.append(build_prompt(messages))

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(processed)} prompts to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Input CSV file", required=True)
    parser.add_argument("--fewshot_file", type=str, help="Few-shot examples CSV file", required=True)
    parser.add_argument("--output_file", type=str, help="Output JSON file", required=True)
    args = parser.parse_args()

    process_csv(args.input_file, args.fewshot_file, args.output_file)

if __name__ == "__main__":
    main()
