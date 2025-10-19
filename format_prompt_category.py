import os
import json
import re
import glob
import argparse
import pandas as pd
from tqdm import tqdm

def extract_all_categories_from_csv(input_file, normalize=True):
    df = pd.read_csv(input_file)
    all_intents = sorted(df['category'].unique().tolist())
    if normalize:
        all_intents = [i.lower() for i in all_intents]
    return all_intents

def build_prompt(messages, expected=None):
    prompt = {"messages": messages}
    if expected:
        prompt["expected"] = expected
    return prompt

def format_messages(sentence, category, all_categories):
    formatted_categories = ", ".join(f'"{i}"' for i in all_categories)

    system_message = (
        "Task: Job Category Classification\n"
        f"Schema: Given the job description, choose the single most appropriate category from the list: [{formatted_categories}].\n"
        "Regulations: Respond ONLY with the exact category name (in lowercase)."
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"The job description is: {sentence}"},
        {"role": "assistant", "content": category}
    ]

def process_csv(input_file, output_file, all_categories=None):
    if all_categories is None:
        all_categories = extract_all_categories_from_csv(input_file)

    df = pd.read_csv(input_file)
    processed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing"):
        sentence_field = "description"
        sentence = str(row.get(sentence_field, "")).strip()
        category = row.get("category", "").strip().lower()
        if not sentence or not category:
            continue
        messages = format_messages(sentence, category, all_categories)
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

    all_categories = extract_all_categories_from_csv(args.input_file)
    process_csv(args.input_file, args.output_file, all_categories)

if __name__ == "__main__":
    main()
