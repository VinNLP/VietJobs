import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

def parse_salary(gen_text):
    try:
        # Split text by whitespace and find first number
        words = gen_text.lower().strip().split()
        for word in words:
            try:
                return int(word)
            except ValueError:
                continue
        return None
    except:
        return None

def prepare_chat_prompt(example, tokenizer=None):
    msgs = example.get("messages", [])
    if not msgs or msgs[-1]["role"] != "assistant":
        return {"prompt": None, "ground_truth_salary": None}
    ground_truth = msgs[-1]["content"].strip().lower()
    prompt_msgs = msgs[:-1]
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True
        )
        return {"prompt": prompt_text, "ground_truth_salary": ground_truth}
    except Exception as e:
        print(f"Tokenizer error: {e}")
        return {"prompt": None, "ground_truth_salary": None}

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
    proc = proc.filter(lambda x: x["prompt"] and x["ground_truth_salary"])
    print(f"📊 Processed dataset size: {len(proc)} samples")
    print(f"🔄 Starting inference with {len(proc)} samples...")

    preds, gts, out_records = [], [], []
    device = model.device
    with torch.no_grad():
        for i in tqdm(range(0, len(proc), args.batch_size), desc="Processing batches"):
            batch = proc[i : i + args.batch_size]
            prompts = batch["prompt"]
            labels = batch["ground_truth_salary"]

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
                pred = parse_salary(text)
                lbl = parse_salary(lbl)
                if pred is not None:
                    preds.append(pred)
                    gts.append(lbl)
                    out_records.append({
                        "prompt": prmpt,
                        "ground_truth_salary": lbl,
                        "generated_text": text,
                        "predicted_salary": pred
                    })

    # Convert to numpy arrays for calculations
    y_true = np.array(gts)
    y_pred = np.array(preds)

    # Compute metrics
    print("🔄 Computing metrics...")
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate R-squared
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print("📊 Results Summary:")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R-squared: {r2:.4f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   Total samples: {len(preds)}")

    results = {
        "model_name": args.model,
        "adapter": args.adapter,
        "metrics": {
            "mape": float(mape),
            "r_squared": float(r2),
            "mae": float(mae),
            "rmse": float(rmse)
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