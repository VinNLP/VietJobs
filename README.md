# VietJobs: A Vietnamese Job Advertisement Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/VietJobs%20Data-🤗%20HuggingFace%20Dataset-blue)](https://huggingface.co/datasets/dinhieufam/VietJobs)
[![GitHub](https://img.shields.io/badge/⭐%20Star-VietJobs-green)](https://github.com/dinhieufam/vietjobs)

---

⭐ **If you find this project helpful, please consider giving it a [star on GitHub](https://github.com/dinhieufam/vietjobs)!**

---

<p align="center">
  <a href="https://www.linkedin.com/in/dinhieufam/" target="_blank"><strong>Hieu Pham-Dinh</strong></a>,
  <a href="https://github.com/whistle-hikhi" target="_blank"><strong>Hung Nguyen Huy</strong></a>,
  <a href="https://elhaj.uk/" target="_blank"><strong>Mo El-Haj</strong></a>,
</p>

<p align="center">
  <em>College of Engineering and Computer Science, VinUniversity</em><br>
</p>

<details>
  <summary><strong>📚 Table of Contents</strong> (click to expand)</summary>

- [🔍 What is VietJobs?](#-what-is-vietjobs)
- [📊 Dataset Statistics](#-dataset-statistics)
- [🗂 Repo Layout](#-repo-layout)
- [I. Quickstart](#i-quickstart)
  - [1. 📥 Clone & Setup](#1--clone--setup)
  - [2. 📦 Install Dependencies](#2--install-dependencies)
  - [3. Run Format, Fine-tune & Evaluation](#3-run-format-fine-tune--evaluation)
- [📄 License](#-license)

</details>

## 📣 News

- **[Feb 2026]** Released experiment scripts and dataset
- **[Feb 2026]** The paper has been accepted at LREC 2026

---

## 🔍 What is VietJobs?

VietJobs is a comprehensive dataset designed for **Vietnamese Natural Language Processing** research, particularly focused on job advertisement analysis. The dataset supports multiple downstream tasks:

- **Job Classification** — Categorizing job postings into industry sectors
- **Salary Estimation** — Predicting salary ranges from job descriptions
- **Information Extraction** — Structured extraction of job attributes

### Key Features

- 🌏 **Comprehensive Coverage**: All 34 Vietnamese provinces and municipalities
- 📊 **Large Scale**: 48,092 job postings with 15+ million words
- 🏷️ **Multi-task Ready**: Pre-processed for classification and regression tasks
- 🔧 **Research Ready**: Includes training, evaluation, and fine-tuning scripts
- 🤖 **LLM Compatible**: Formatted for modern language model training

### Data Distribution

- **Job Categories**: Technology, Finance, Healthcare, Education, Manufacturing, etc.
- **Experience Levels**: Entry-level to Senior (0–10+ years)
- **Contract Types**: Full-time, Part-time, Internship, Freelance
- **Salary Ranges**: 1–500 million VND

---

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Job Postings | 48,092 |
| Total Words | 15,000,000+ |
| Geographic Coverage | 34 provinces/municipalities |
| Average Posting Length | ~321 words |
| Job Categories | 16 categories |
| Time Period | July – October 2025 |

---

## 🗂 Repo Layout

- `data/` — dataset samples and directory conventions (full dataset on HuggingFace)
- `run_format_prompt.sh` — format prompts and prepare data for training
- `run_finetune_lora.sh` — fine-tune the model using LoRA
- `run_evaluation.sh` — run evaluation pipelines
- `requirements.txt` — Python dependencies

---

## I. Quickstart

### 1. 📥 Clone & Setup

```bash
git clone https://github.com/dinhieufam/vietjobs.git
cd VietJobs
```

### 2. 📦 Install Dependencies

**Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Dataset**: The dataset is available on HuggingFace and you can download it via:

```bash
huggingface-cli download dinhieufam/VietJobs
```

### 3. Run Format, Fine-tune & Evaluation

**Step 1 — Format prompts and prepare data**

```bash
bash run_format_prompt.sh
```

**Step 2 — Fine-tune the model using LoRA**

```bash
bash run_finetune_lora.sh
```

**Step 3 — Run evaluation**

```bash
bash run_evaluation.sh
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Made with ❤️ for the Vietnamese NLP community</strong>
</p>
