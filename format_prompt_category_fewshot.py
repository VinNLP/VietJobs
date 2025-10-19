import os
import json
import csv
import argparse
from tqdm import tqdm
import pandas as pd

def extract_all_categories_from_csv(input_file, normalize=True):
    df = pd.read_csv(input_file)
    all_categories = sorted(df['category'].dropna().unique().tolist())
    if normalize:
        all_categories = [c.lower() for c in all_categories]
    return all_categories

def load_csv_file(path):
    """Load CSV expecting 'text' (or 'description') and 'category' columns, with BOM-safe decoding."""
    data = []
    # Use utf-8-sig to handle potential BOM in CSV header
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Support both 'text' and 'description' as the source field
            text = (row.get("text") or row.get("description") or "").strip()
            category = (row.get("category") or "").strip().lower()
            if text and category:
                data.append({"text": text, "category": category})
    print(f"Loaded {len(data)} rows from {os.path.basename(path)}")
    return data

def prepare_fewshot_prompt(example, fewshot_examples, all_categories):
    formatted_categories = ", ".join(f'"{c}"' for c in all_categories)
    system_msg = (
        "Task: Job Category Classification\n"
        f"Schema: Given the job description, choose the single most appropriate category from the list: [{formatted_categories}].\n"
        "Regulations: Respond ONLY with the exact category name (in lowercase)."
    )

    messages = [{"role": "system", "content": system_msg}]

    # Add few-shot examples first
    for fs in fewshot_examples:
        messages.append({"role": "user", "content": fs['text']})
        messages.append({"role": "assistant", "content": fs['category']})

    # Add the actual example last
    messages.append({"role": "user", "content": f"The job description is: {example['text']}"})
    messages.append({"role": "assistant", "content": example['category']})

    return {"messages": messages}

def generate_prompts(dataset, fewshot_examples, all_categories):
    prompts = []
    for ex in tqdm(dataset, desc="Generating few-shot prompts"):
        prompts.append(prepare_fewshot_prompt(ex, fewshot_examples, all_categories))
    return prompts

def save_json(data, output_file):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} prompts to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Main dataset CSV file")
    parser.add_argument("--fewshot_file", type=str, required=True, help="Few-shot examples CSV file") 
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()

    dataset = load_csv_file(args.input_file)
    fewshot_examples = load_csv_file(args.fewshot_file)
    all_categories = extract_all_categories_from_csv(args.input_file)

    print(f"Found {len(all_categories)} unique job categories")
    print(f"Using {len(fewshot_examples)} few-shot examples")

    prompts = generate_prompts(dataset, fewshot_examples, all_categories)
    save_json(prompts, args.output_file)
    print("Few-shot data preparation complete!")

if __name__ == "__main__":
    main()
