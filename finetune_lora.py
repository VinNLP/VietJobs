#!/usr/bin/env python
import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login
from transformers import EarlyStoppingCallback
from HUGGINGFACE_API import API_TOKEN

login(API_TOKEN) 

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLMs with LoRA")
    parser.add_argument("--model_name", type=str, required=True, help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data file")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to evaluation data file")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for private models")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA r dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.2, help="LoRA dropout value")
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", 
                        help="Comma-separated list of target modules for LoRA")
    
    # Training parameters
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--eval_save_steps", type=int, default=200, help="Steps between evaluations and saves")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed configuration file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Login to Hugging Face if token provided
    if args.hf_token:
        login(args.hf_token)
    
    # Check GPU availability
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    # --- Model Loading ---
    print(f"Loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2",
    )
    
    # --- PEFT Setup ---
    target_modules = args.target_modules.split(',')
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none"
    )
    
    print("Applying LoRA PEFT adapter...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # --- Tokenizer ---
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # For Flash Attention compatibility
    
    # --- Data Loading and Processing ---
    print(f"Loading datasets: {args.train_file}, {args.dev_file}")
    train_dataset = load_dataset("json", data_files=args.train_file)['train']
    eval_dataset = load_dataset("json", data_files=args.dev_file)['train']
    
    def tokenize_function(examples):
        if not isinstance(examples.get("messages"), list):
            return {"input_ids": [], "attention_mask": []}

        texts = []
        for messages in examples["messages"]:
            if isinstance(messages, list) and all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in messages):
                try:
                    formatted_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    texts.append(formatted_text)
                except Exception as e:
                    pass
            else:
                pass

        if not texts:
            return {"input_ids": [], "attention_mask": []}

        tokenized = tokenizer(
            texts,
            padding=False,  # Padding will be handled by DataCollator
            truncation=True,
            max_length=args.max_seq_length,
        )
        return tokenized
    
    num_proc = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1
    
    print("Tokenizing training data...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=train_dataset.column_names
    )
    
    print("Tokenizing evaluation data...")
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=eval_dataset.column_names
    )
    
    # --- Data Collator ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # --- Training Arguments ---
    print("Defining Training Arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.micro_batch_size,
        eval_strategy="steps",
        eval_steps=args.eval_save_steps,  
        logging_steps=20,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        warmup_ratio=0.03,
        bf16=True,
        fp16=False,
        tf32=True,
        save_strategy="steps",
        save_steps=args.eval_save_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="tensorboard",
        dataloader_num_workers=8,
        gradient_checkpointing=False,  
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
    )
    
    if training_args.gradient_checkpointing:
        print("Enabling gradient checkpointing on the PEFT model...")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    # --- Trainer Initialization ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # --- Start Training ---
    print("Starting training...")
    trainer.train()
    
    # --- Save Final Model ---
    print("Saving final best PEFT adapter...")
    final_peft_path = os.path.join(args.output_dir, "final_best_adapter")
    model.save_pretrained(final_peft_path)
    tokenizer.save_pretrained(final_peft_path)
    print(f"Final model state saved to {final_peft_path}. Best model checkpoint is in {args.output_dir}/checkpoint-<best_step>.")
    
    print("Training complete.")

if __name__ == "__main__":
    main()