# Vulnerability Classification with DistilBERT, CodeBERT, and Gemma-2B

This repository evaluates three Transformer models—**DistilBERT**, **CodeBERT**, and **Gemma-2B**—for **binary vulnerability classification** on code functions, under two data-balancing hypotheses:

- **H1 — Equal Distribution by Subclass:** each category (API, AU, PU, AE) is internally balanced (all vulnerable + equal # of non-vulnerable).
- **H2 — Balance by Smallest Subclass:** every category is downsampled to the smallest vulnerable subclass size; perfectly balanced and uniform.

---


# 📂 Download SySeVR Data (one step)

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/SySeVR/SySeVR.git

## Data

Place your raw CSV under `data/raw/` (default expected columns):

- `Function` – source code string
- `Label` – binary (1 = vulnerable, 0 = non-vulnerable)
- `Type of Vulnerability` – one of `{API, AU, PU, AE}` (names can be longer; we do string match)

You can override column names via CLI flags.

---

## Reproducible Splits

```bash
# Hypothesis 1: per-category balance (all vulnerable + equal non-vulnerable)
python scripts/prepare_data.py \
  --input data/raw/merged_allkind.csv \
  --strategy h1 \
  --outdir data/prepared/h1

# Hypothesis 2: uniform low-resource balance by smallest vulnerable subclass
python scripts/prepare_data.py \
  --input data/raw/merged_allkind.csv \
  --strategy h2 \
  --outdir data/prepared/h2

## Training

# Example: train all 3 models on H1
python scripts/train.py \
  --dataset_dir data/prepared/h1 \
  --models microsoft/codebert-base,distilbert-base-uncased,google/gemma-2-2b \
  --epochs 5 --batch_size 16 --max_len 128


# Example: train all 3 models on H2
python scripts/train.py \
  --dataset_dir data/prepared/h2 \
  --models microsoft/codebert-base,distilbert-base-uncased,google/gemma-2-2b \
  --epochs 5 --batch_size 16 --max_len 128



# Requirements

```text
transformers>=4.41.0
datasets>=2.20.0
accelerate>=0.30.0
scikit-learn>=1.3.0
torch>=2.2.0
pandas>=2.0.0
numpy>=1.24.0
psutil>=5.9.0
tqdm>=4.66.0

