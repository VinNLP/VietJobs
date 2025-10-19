python format_prompt_category_fewshot.py --fewshot_file data/samples_category.csv --input_file data/test.csv --output_file data/job_classification/test_fewshot.json
python format_prompt_salary_fewshot.py --fewshot_file data/samples_salary.csv --input_file data/test.csv --output_file data/salary_estimation/test_fewshot.json
python format_prompt_salary_kaggle_fewshot.py --fewshot_file data/samples_salary.csv --input_file data/test_kaggle.csv --output_file data/salary_estimation/test_kaggle_fewshot.json

python format_prompt_category.py --input_file data/train.csv --output_file data/job_classification/train.json
python format_prompt_salary.py --input_file data/train.csv --output_file data/salary_estimation/train.json
python format_prompt_salary_kaggle.py --input_file data/train_kaggle.csv --output_file data/salary_estimation/train_kaggle.json

python format_prompt_category.py --input_file data/dev.csv --output_file data/job_classification/dev.json
python format_prompt_salary.py --input_file data/dev.csv --output_file data/salary_estimation/dev.json
python format_prompt_salary_kaggle.py --input_file data/dev_kaggle.csv --output_file data/salary_estimation/dev_kaggle.json

python format_prompt_category.py --input_file data/test.csv --output_file data/job_classification/test.json
python format_prompt_salary.py --input_file data/test.csv --output_file data/salary_estimation/test.json
python format_prompt_salary_kaggle.py --input_file data/test_kaggle.csv --output_file data/salary_estimation/test_kaggle.json