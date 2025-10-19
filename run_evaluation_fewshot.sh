# ======================================
# CONFIGURATION
# ======================================

# List of models (add as many as you want)
models=(
    "Qwen/Qwen2.5-7B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "aisingapore/Llama-SEA-LION-v3-8B-IT"
    # "mistralai/Ministral-8B-Instruct-2410"
    # "sail/Sailor2-8B-Chat"
    # "SeaLLMs/SeaLLMs-v3-7B-Chat"
)

# Corresponding short model names (used for folder and output naming)
model_names=(
    "Qwen2.5-7B-Instruct"
    # "Llama-3.1-8B-Instruct"
    # "Llama-SEA-LION-v3-8B-IT"
    # "Ministral-8B-Instruct-2410"
    # "Sailor2-8B-Chat"
    # "SeaLLMs-v3-7B-Chat"
)

# Task
category_task="job_classification"
salary_task="salary_estimation"

# Test files
declare -A category_test_files=(
  ["default_fewshot"]="data/${category_task}/test_fewshot.json"
)

declare -A salary_test_files=(
  ["default_fewshot"]="data/${salary_task}/test_fewshot.json"
  ["kaggle_fewshot"]="data/${salary_task}/test_kaggle_fewshot.json"
  ["merged_fewshot"]="data/${salary_task}/test_merged_fewshot.json"
)

# Output directory
output_dir="results"

# ======================================
# MAIN LOOP
# ======================================

for i in "${!models[@]}"; do
  model_path="${models[$i]}"
  model_name="${model_names[$i]}"

  echo "======================================"
  echo "🔹 Evaluating model: ${model_name}"
  echo "======================================"

  # Run category tests
  echo "Running category classification tests..."
  for tag in "${!category_test_files[@]}"; do
    test_file="${category_test_files[$tag]}"
    output_file="${model_name}_category_test-${tag}.json"

    echo "➡️  Running: ${model_name} (test: ${tag})"
    python evaluation_category.py \
      --model "$model_path" \
      --test_file "$test_file" \
      --output_file "$output_file" \
      --output_dir "$output_dir"
  done

  echo "✅ Finished category classifications for ${model_name}"
  echo

  # Run salary tests  
  echo "Running salary estimation tests..."
  for tag in "${!salary_test_files[@]}"; do
    test_file="${salary_test_files[$tag]}"
    output_file="${model_name}_salary_test-${tag}.json"

    echo "➡️  Running: ${model_name} (test: ${tag})"
    python evaluation_salary.py \
      --model "$model_path" \
      --test_file "$test_file" \
      --output_file "$output_file" \
      --output_dir "$output_dir"
  done

  echo "✅ Finished salary estimations for ${model_name}"
  echo
done

echo "🎯 All evaluations completed!"
