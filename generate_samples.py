import os
import json
import random
import csv

def extract_text_and_label_from_messages(messages):
    user_text = None
    assistant_text = None
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            user_text = content
        elif role == "assistant":
            assistant_text = content
    return user_text, assistant_text


def sample_category(dev_json_path, out_csv_path, expected_unique_count=16):
    with open(dev_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group samples by category label (assistant content)
    label_to_samples = {}
    for item in data:
        messages = item.get("messages", [])
        text, label = extract_text_and_label_from_messages(messages)
        if not text or not label:
            continue
        label_to_samples.setdefault(label, []).append((text, label))

    # Choose 1 random per label, up to expected_unique_count labels
    chosen_rows = []
    labels = list(label_to_samples.keys())
    # Keep original order but limit to expected_unique_count distinct labels if more
    for label in labels[:expected_unique_count]:
        samples = label_to_samples.get(label, [])
        if not samples:
            continue
        chosen = random.choice(samples)
        chosen_rows.append(chosen)

    # Write CSV
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "category"])  # header
        for text, label in chosen_rows:
            writer.writerow([text, label])


def sample_salary(dev_json_path, out_csv_path, k=3):
    with open(dev_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        messages = item.get("messages", [])
        text, label = extract_text_and_label_from_messages(messages)
        if not text or not label:
            continue
        rows.append((text, label))

    if len(rows) == 0:
        chosen = []
    elif len(rows) <= k:
        chosen = rows
    else:
        chosen = random.sample(rows, k)

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "salary"])  # header
        for text, label in chosen:
            writer.writerow([text, label])


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root, "experiments", "data")

    category_dev = os.path.join(data_dir, "job_classification", "dev.json")
    salary_dev = os.path.join(data_dir, "salary_estimation", "dev.json")

    out_category_csv = os.path.join(data_dir, "samples_category.csv")
    out_salary_csv = os.path.join(data_dir, "samples_salary.csv")

    sample_category(category_dev, out_category_csv, expected_unique_count=2)
    # sample_salary(salary_dev, out_salary_csv, k=3)


if __name__ == "__main__":
    main()


