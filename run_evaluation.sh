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

# Base directory for checkpoints
base_dir="../checkpoints/data"

# Task
salary_task="salary_estimation"

# Adapter types — leave blank for no adapter
# When adapter_type="", the script will skip the --adapter argument
salary_adapter_types=("" "train_kaggle" "train_merged")

# Test files
declare -A salary_tests=(
  ["default"]="data/${salary_task}/test.json"
  ["kaggle"]="data/${salary_task}/test_kaggle.json"
  ["merged"]="data/${salary_task}/test_merged.json"
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

  for adapter_type in "${salary_adapter_types[@]}"; do
    # Build adapter path only if adapter_type is not blank
    if [ -n "$adapter_type" ]; then
      adapter_path="${base_dir}/${salary_task}/${model_name}_${adapter_type}/final_best_adapter"
      adapter_arg="--adapter $adapter_path"
      adapter_label="${adapter_type}"
    else
      adapter_arg=""
      adapter_label="no-adapter"
    fi

    for tag in "${!salary_tests[@]}"; do
      test_file="${salary_tests[$tag]}"

      # Build output tag and file name
      if [ "$tag" == "default" ]; then
        output_tag="salary_estimation"
      else
        output_tag="${tag}_salary_estimation"
      fi

      output_file="${model_name}_${adapter_label}_test-${output_tag}.json"

      echo "➡️  Running: ${model_name} (${adapter_label}, test: ${tag})"
      python evaluation_salary.py \
        --model "$model_path" \
        $adapter_arg \
        --test_file "$test_file" \
        --output_file "$output_file" \
        --output_dir "$output_dir"
    done
  done

  echo "✅ Finished salary estimations for ${model_name}"
  echo
done

echo "🎯 All salary estimation evaluations completed!"
