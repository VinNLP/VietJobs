# VietJobs: Vietnamese Job Advertisement Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The first large-scale, publicly available corpus of Vietnamese job advertisements, comprising **48,092 postings** and over **15 million words** collected from all 34 provinces and municipalities across Vietnam.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset Statistics](#dataset-statistics)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [License](#license)
- [Contributing](#contributing)

## 🎯 Overview

VietJobs is a comprehensive dataset designed for Vietnamese Natural Language Processing research, particularly focused on job advertisement analysis. The dataset supports multiple downstream tasks including:

- **Job Classification**: Categorizing job postings into industry sectors
- **Salary Estimation**: Predicting salary ranges from job descriptions
- **Information Extraction**: Structured extraction of job attributes

### Key Features

- 🌏 **Comprehensive Coverage**: All 34 Vietnamese provinces and municipalities
- 📊 **Large Scale**: 48,092 job postings with 15+ million words
- 🏷️ **Multi-task Ready**: Pre-processed for classification and regression tasks
- 🔧 **Research Ready**: Includes training, evaluation, and fine-tuning scripts
- 🤖 **LLM Compatible**: Formatted for modern language model training

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Job Postings | 48,092 |
| Total Words | 15,000,000+ |
| Geographic Coverage | 34 provinces/municipalities |
| Average Posting Length | ~321 words |
| Job Categories | 16 categories |
| Time Period | July - October 2025 |

### Data Distribution

- **Job Categories**: Technology, Finance, Healthcare, Education, Manufacturing, etc.
- **Experience Levels**: Entry-level to Senior (0-10+ years)
- **Contract Types**: Full-time, Part-time, Internship, Freelance
- **Salary Ranges**: 1-500 million VND

## 🚀 Installation

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/VietJobs.git
   cd VietJobs
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset from HuggingFace**
   ```bash
   # The dataset will be automatically downloaded when running the scripts
   # or you can manually download using:
   # huggingface-cli download your-username/vietjobs-dataset
   ```

## 🚀 Usage Examples

To get started with VietJobs, follow these steps:

1. **Format the prompts and prepare data**
   ```bash
   bash run_format_prompt.sh
   ```

2. **Fine-tune the model using LoRA**
   ```bash
   bash run_finetune_lora.sh
   ```

3. **Run evaluation**
   ```bash
   bash run_evaluation.sh
   ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:

- Report bugs
- Suggest new features
- Submit pull requests
- Join our community discussions

---

**Made with ❤️ for the Vietnamese NLP community**




