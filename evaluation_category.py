import os
import json
import torch
import argparse
import pandas as pd
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from peft import PeftModel
from huggingface_hub import login
from HUGGINGFACE_API import API_TOKEN

login(API_TOKEN)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--model", type=str, required=True, help="Base model name")
    parser.add_argument("--adapter", type=str, required=False, help="Path to adapter")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Name of output JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    return parser.parse_args()

def extract_all_categories_from_csv(input_file, normalize=True):
    df = pd.read_csv(input_file)
    all_categories = sorted(df['category'].unique().tolist())
    if normalize:
        # Don't normalize the categories since we need exact match
        all_categories = [i.lower() for i in all_categories]
    return all_categories

def parse_category(gen_text, all_categories):
    gen_text = gen_text.strip().lower()
    # Exact match only
    if gen_text in all_categories:
        return gen_text
    return "Other"

def prepare_chat_prompt(example, tokenizer=None):
    msgs = example.get("messages", [])
    if not msgs or msgs[-1]["role"] != "assistant":
        return {"prompt": None, "ground_truth_category": None}
    ground_truth = msgs[-1]["content"].strip().lower()
    prompt_msgs = msgs[:-1]
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True
        )
        return {"prompt": prompt_text, "ground_truth_category": ground_truth}
    except Exception as e:
        print(f"Tokenizer error: {e}")
        return {"prompt": None, "ground_truth_category": None}

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Starting evaluation...")
    print(f"📊 Model: {args.model}")
    print(f"🔧 Adapter: {args.adapter if args.adapter else 'None (base model)'}")
    print(f"📁 Test file: {args.test_file}")
    print(f"💾 Output: {args.output_dir}/{args.output_file}")
    print(f"⚙️  Batch size: {args.batch_size}")
    print("=" * 50)

    all_categories = extract_all_categories_from_csv('data/test.csv')
    print(f"📋 Loaded {len(all_categories)} categories")

    # Load base model
    print("🔄 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print("✅ Base model loaded successfully")

    # Apply adapter if provided
    if args.adapter and os.path.exists(args.adapter):
        print(f"🔄 Loading adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        print("✅ Adapter loaded successfully")
    else:
        print("ℹ️  No adapter provided, using base model")

    model.eval()

    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded successfully")

    # Load dataset
    print("🔄 Loading and processing dataset...")
    raw_ds = load_dataset("json", data_files=args.test_file)["train"]
    print(f"📊 Raw dataset size: {len(raw_ds)} samples")
    
    proc = raw_ds.map(lambda ex: prepare_chat_prompt(ex, tokenizer), remove_columns=raw_ds.column_names)
    proc = proc.filter(lambda x: x["prompt"] and x["ground_truth_category"])
    print(f"📊 Processed dataset size: {len(proc)} samples")
    print(f"🔄 Starting inference with {len(proc)} samples...")

    preds, gts, out_records = [], [], []
    device = model.device
    with torch.no_grad():
        for i in tqdm(range(0, len(proc), args.batch_size), desc="Processing batches"):
            batch = proc[i : i + args.batch_size]
            prompts = batch["prompt"]
            labels = batch["ground_truth_category"]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_length,
            )
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

            seq_len = inputs["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(gen[:, seq_len:], skip_special_tokens=True)

            for prmpt, lbl, text in zip(prompts, labels, decoded):
                pred = parse_category(text, all_categories)
                preds.append(pred)
                gts.append(lbl)
                out_records.append({
                    "prompt": prmpt,
                    "ground_truth": lbl,
                    "generated_text": text,
                    "predicted_category": pred,
                    "correct": pred == lbl,
                })

    # Compute metrics
    print("🔄 Computing metrics...")
    accuracy = accuracy_score(gts, preds)
    macro_f1 = f1_score(gts, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(gts, preds, average='weighted', zero_division=0)
    class_report = classification_report(gts, preds, zero_division=0, digits=4, output_dict=True)

    unique_labels = sorted(list(set(gts + preds)))
    conf_matrix = confusion_matrix(gts, preds, labels=unique_labels)
    
    print("📊 Results Summary:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Macro F1: {macro_f1:.4f}")
    print(f"   Weighted F1: {weighted_f1:.4f}")
    print(f"   Total samples: {len(preds)}")
    print(f"   Correct predictions: {sum(1 for p, g in zip(preds, gts) if p == g)}")

    results = {
        "model_name": args.model,
        "adapter": args.adapter,
        "metrics": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "class_report": class_report
        },
        "confusion_matrix": {
            "labels": unique_labels,
            "matrix": conf_matrix.tolist()
        },
        "predictions": out_records
    }

    out_path = os.path.join(args.output_dir, args.output_file)
    print(f"💾 Saving results to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("✅ Evaluation completed successfully!")
    print(f"📁 Results saved to: {out_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()