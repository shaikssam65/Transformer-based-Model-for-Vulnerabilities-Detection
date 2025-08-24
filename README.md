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
