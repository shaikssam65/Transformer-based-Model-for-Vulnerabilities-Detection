#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def normalize_columns(df, code_col, label_col, type_col):
    # Standardize column names
    rename_map = {
        code_col: "code",
        label_col: "label",
        type_col: "vtype",
    }
    df = df.rename(columns=rename_map)
    # Minimal cleaning
    df["label"] = df["label"].astype(int)
    df["vtype"] = df["vtype"].astype(str).str.strip()
    return df[["code", "label", "vtype"]].dropna()

def hypothesis1(df, seed=42):
    """
    For each vtype:
      - take ALL vulnerable (label==1)
      - sample equal number of non-vulnerable (label==0) from the SAME vtype
    """
    out = []
    vtypes = sorted(df["vtype"].unique())
    for vt in vtypes:
        pos = df[(df.vtype == vt) & (df.label == 1)]
        if len(pos) == 0:
            continue
        neg = df[(df.vtype == vt) & (df.label == 0)]
        n = len(pos)
        if len(neg) < n:
            # if insufficient negatives in that vtype, sample with replacement
            neg_s = neg.sample(n=n, replace=True, random_state=seed)
        else:
            neg_s = neg.sample(n=n, random_state=seed)
        out.append(pd.concat([pos, neg_s], ignore_index=True))
    balanced = pd.concat(out, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced

def hypothesis2(df, seed=42):
    """
    For each vtype:
      - find min vulnerable count across vtypes
      - sample min_count vulnerable and min_count non-vulnerable within that vtype
    """
    vtypes = sorted(df["vtype"].unique())
    vuln_counts = df[df.label == 1]["vtype"].value_counts()
    if vuln_counts.empty:
        raise ValueError("No vulnerable samples (label==1) found.")
    min_count = int(vuln_counts.min())
    out = []
    for vt in vtypes:
        pos = df[(df.vtype == vt) & (df.label == 1)]
        if len(pos) == 0:
            continue
        pos_s = pos.sample(n=min_count, replace=(len(pos) < min_count), random_state=seed)

        neg = df[(df.vtype == vt) & (df.label == 0)]
        neg_s = neg.sample(n=min_count, replace=(len(neg) < min_count), random_state=seed)

        out.append(pd.concat([pos_s, neg_s], ignore_index=True))
    balanced = pd.concat(out, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced

def split_and_save(df, outdir, seed=42, test_size=0.15, val_size=0.15):
    os.makedirs(outdir, exist_ok=True)
    # Stratify by (label, vtype) to keep both proportions
    strat = df["label"].astype(str) + "_" + df["vtype"].astype(str)

    train_dev, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strat
    )
    strat_dev = train_dev["label"].astype(str) + "_" + train_dev["vtype"].astype(str)

    # compute val split relative to train_dev
    val_ratio = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_dev, test_size=val_ratio, random_state=seed, stratify=strat_dev
    )

    for name, split in [("train", train), ("val", val), ("test", test)]:
        split.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)

    # Also keep a single file (balanced.csv) for reference
    df.to_csv(os.path.join(outdir, "balanced.csv"), index=False)

    # Basic stats
    def stats(d):
        return d.groupby(["vtype", "label"]).size().unstack(fill_value=0)

    (stats(train)).to_csv(os.path.join(outdir, "_stats_train_by_vtype.csv"))
    (stats(val)).to_csv(os.path.join(outdir, "_stats_val_by_vtype.csv"))
    (stats(test)).to_csv(os.path.join(outdir, "_stats_test_by_vtype.csv"))

    print(f"[OK] Saved splits to: {outdir}")
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}  total={len(df)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw CSV (e.g., merged_allkind.csv)")
    ap.add_argument("--strategy", choices=["h1", "h2"], required=True, help="Balancing strategy")
    ap.add_argument("--outdir", required=True, help="Output directory (e.g., data/prepared/h1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--code_col", default="Function")
    ap.add_argument("--label_col", default="Label")
    ap.add_argument("--type_col", default="Type of Vulnerability")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = normalize_columns(df, args.code_col, args.label_col, args.type_col)

    if args.strategy == "h1":
        balanced = hypothesis1(df, seed=args.seed)
    else:
        balanced = hypothesis2(df, seed=args.seed)

    split_and_save(balanced, args.outdir, seed=args.seed,
                   test_size=args.test_size, val_size=args.val_size)

if __name__ == "__main__":
    main()
