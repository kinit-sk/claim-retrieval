# A Generative-AI-Driven Claim Retrieval System Capable of Detecting and Retrieving Claims from Social Media Platforms in Multiple Languages
This repository contains the official codebase for our paper: "A Generative-AI-Driven Claim Retrieval System Capable of Detecting and Retrieving Claims from Social Media Platforms in Multiple Languages"

The system is designed to support fact-checkers by retrieving relevant previously fact-checked claims for a given input, filtering out irrelevant content, and generating human-readable summaries and explanations using large language models (LLMs). This tool aims to streamline the fact-checking workflow and reduce the verification effort across multiple languages.


<!-- Important files:
 - `Prepare multiclaim.ipynb`


Files that could be found on [Zenodo]():
- `afp-sum.csv`
- `sample2.csv`
- `sample100.csv`
- `multiclaim/sampled_posts.csv`
- `fact_checks_metadata.csv` -->

## Abstract

Online disinformation poses a global challenge, placing significant demands on fact-checkers who must verify claims efficiently to prevent the spread of false information. A major issue in this process is the redundant verification of already fact-checked claims, which increases workload and delays responses to newly emerging claims. This research introduces an approach that retrieves previously fact-checked claims, evaluates their relevance to a given input, and provides supplementary information to support fact-checkers. Our method employs large language models (LLMs) to filter irrelevant fact-checks and generate concise summaries and explanations, enabling fact-checkers to faster assess whether a claim has been verified before. In addition, we evaluate our approach through both automatic and human assessments, where humans interact with the developed tool to review its effectiveness. Our results demonstrate that LLMs are able to filter out many irrelevant fact-checks and, therefore, reduce effort and streamline the fact-checking process.

## 📦 Reproducibility & Setup

### 🔧 Installation

We recommend using Python 3.11.5 for compatibility.

1. Clone this repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Install FAISS with GPU support for efficient similarity search:

```bash
mkdir faiss
cd faiss
wget https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
python -m pip install faiss_gpu-1.7.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### 📁 Data Preparation

`Prepare multiclaim.ipynb`

### 🔍 Running Experiments

#### Simple Retrieval Experiments

These experiments test the baseline retrieval performance using English-only and multilingual text embeddings models (TEMs).

```bash
python -m scripts.language_experiments
```

#### Filtered Retrieval Experiments

These experiments apply natural language instructions to retrieve the relevant fact-checks.

```bash
python -m scripts.filtered_experiments
python -m scripts.original_experiments
python -m scripts.filtered_evaluation
```

#### Summarization

To expriment with summarization of [AFP](https://factcheck.afp.com/) fact-checking articles with several LLMs:

Set your API keys for OpenAI and Anthropic in `scripts/summarization_experiments.py`, especially the following ones:
- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- OPENAI_AZURE_ENDPOINT

Then run:

```bash
python -m scripts.summarization_experiments
```

#### Overall Pipeline

This script ties together retrieval, filtering, and summarization into a single pipeline:

```bash
python -m scripts.pipeline
```

To experiments with various LLM, it is necessary to update the `model_name` field in the following config file: `configs/pipeline.yaml`.

You can evaluate the full pipeline using the included Jupyter notebook:
- `Pipeline evaluation.ipynb`

#### Retrieval Classification Baseline

For comparison, we provide a retrieval classification baseline.

First run code in `Retrieval baseline.ipynb` to prepare data for the baseline experiments.

Then, run the code fo retreival baseline using the following command:

```bash
python -m scripts.retrieval_classification
```

#### Veracity Classification Baseline

For comparison, we provide a simple veracity classification baseline:


```bash
python -m scripts.veracity_baseline
```

## Paper citing

If you use the code or information from this repository, please cite our paper, which is available on arXiv.

```bibtex
```

