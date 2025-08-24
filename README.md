# Vulnerability Classification with DistilBERT, CodeBERT, and Gemma-2B

Evaluate three Transformer models—**DistilBERT**, **CodeBERT**, and **Gemma-2B**—for **binary vulnerability classification** on code functions under two balancing strategies.

---

## Hypotheses

- **H1 — Equal Distribution by Subclass:**  
  For each category (API, AU, PU, AE), include **all vulnerable** samples and sample an **equal number of non-vulnerable** samples from the same category. Preserves category signal while removing class imbalance.

- **H2 — Balance by Smallest Subclass:**  
  Downsample each category to the size of the **smallest vulnerable subclass** and match with an equal number of non-vulnerable samples. Creates a perfectly uniform, low-resource dataset.

---

## 📂 Data

Place your raw CSV at `data/raw/merged_allkind.csv` with the following (default) columns:

- `Function` — source code string  
- `Label` — 1 = vulnerable, 0 = non-vulnerable  
- `Type of Vulnerability` — one of `{API, AU, PU, AE}` (names can be longer; string match)

> You can override column names via CLI flags in `scripts/prepare_data.py`.

### (Optional) Fetch SySeVR repository
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/SySeVR/SySeVR.git
cd SySeVR && git sparse-checkout set --no-cone .
cd ..
```

# Install all required Python packages
```bash
pip install transformers>=4.41.0 \
             datasets>=2.20.0 \
             accelerate>=0.30.0 \
             scikit-learn>=1.3.0 \
             torch>=2.2.0 \
             pandas>=2.0.0 \
             numpy>=1.24.0 \
             psutil>=5.9.0 \
             tqdm>=4.66.0

```

### Prepare Data

## H1 (Per-category balance)
```bash
python scripts/prepare_data.py \
  --input data/raw/merged_allkind.csv \
  --strategy h1 \
  --outdir data/prepared/h1
```

## H2 (Uniform low-resource balance)
```bash
python scripts/prepare_data.py \
  --input data/raw/merged_allkind.csv \
  --strategy h2 \
  --outdir data/prepared/h2
```

### 🚀 Training
```bash
Optional GPU Selection

Linux/macOS:

export CUDA_VISIBLE_DEVICES=0


Windows (PowerShell):

$env:CUDA_VISIBLE_DEVICES="0"

```



### Train all 3 models on H1
```bash
python scripts/train.py \
  --dataset_dir data/prepared/h1 \
  --models microsoft/codebert-base,distilbert-base-uncased,google/gemma-2-2b \
  --epochs 5 --batch_size 16 --max_len 128
```

Train all 3 models on H2
```bash
python scripts/train.py \
  --dataset_dir data/prepared/h2 \
  --models microsoft/codebert-base,distilbert-base-uncased,google/gemma-2-2b \
  --epochs 5 --batch_size 16 --max_len 128
```

