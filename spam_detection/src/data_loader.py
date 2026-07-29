"""
data_loader.py - Load and split spam datasets.

Supports two formats automatically:
  1. processed_data_clean.csv  → columns: full_text, label (0/1)
  2. Legacy spam.csv           → columns: Message, Category (ham/spam)
  3. unified_dataset.csv       → columns: text, label (0/1)

Class imbalance is handled via class weights.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src import config


def load_dataset(path: str = config.DATA_PATH) -> pd.DataFrame:
    """Load CSV, auto-detect format, encode labels to 0/1."""
    print(f"Loading dataset from: {path}")
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, encoding=config.ENCODING)

    print(f"Detected columns: {list(df.columns)}")

    # ── Format 1: processed_data_clean.csv (full_text + label) ───────────────
    if "full_text" in df.columns and "label" in df.columns:
        print("Format detected: processed_data_clean.csv")
        df = df[["full_text", "label"]].dropna()
        df = df.rename(columns={"full_text": "text"})
        df["label"] = df["label"].astype(int)

    # ── Format 2: unified_dataset.csv (text + label) ─────────────────────────
    elif "text" in df.columns and "label" in df.columns:
        print("Format detected: unified_dataset.csv")
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)

    # ── Format 3: legacy spam.csv (Message + Category) ───────────────────────
    elif config.TEXT_COL in df.columns and config.LABEL_COL in df.columns:
        print("Format detected: legacy spam.csv")
        df = df[[config.TEXT_COL, config.LABEL_COL]].dropna()
        df = df.rename(columns={config.TEXT_COL: "text", config.LABEL_COL: "label"})

        # Convert "ham"/"spam" strings to 0/1
        sample = str(df["label"].iloc[0])
        if not sample.lstrip("-").isdigit():
            unique = df["label"].astype(str).unique().tolist()
            spam_str = next((v for v in unique if "spam" in v.lower()), None)
            ham_str  = next((v for v in unique if "ham"  in v.lower()), None)
            if spam_str and ham_str:
                df["label"] = df["label"].astype(str).map({ham_str: 0, spam_str: 1})
            else:
                mapping = {v: i for i, v in enumerate(sorted(unique))}
                df["label"] = df["label"].astype(str).map(mapping)
                print(f"Label mapping applied: {mapping}")

        df["label"] = df["label"].astype(int)

    else:
        raise ValueError(
            f"Unrecognised dataset format.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Expected one of:\n"
            f"  - 'full_text' + 'label'  (processed_data_clean.csv)\n"
            f"  - 'text'     + 'label'  (unified_dataset.csv)\n"
            f"  - '{config.TEXT_COL}' + '{config.LABEL_COL}'  (spam.csv)"
        )

    n_spam = int(df["label"].sum())
    n_ham  = int((df["label"] == 0).sum())
    ratio  = n_ham / n_spam if n_spam else float("inf")

    print(f"\nDataset loaded: {len(df):,} samples")
    print(f"  Ham  (0): {n_ham:,} ({n_ham/len(df)*100:.1f}%)")
    print(f"  Spam (1): {n_spam:,} ({n_spam/len(df)*100:.1f}%)")
    print(f"  Imbalance ratio: {ratio:.1f}:1 (ham:spam)")

    if ratio > 3 or ratio < 0.33:
        print("  Class imbalance detected — class weights will be applied during training.")

    return df


def compute_weights(labels) -> dict:
    """Compute class weights to counter imbalance."""
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    cw = {0: float(weights[0]), 1: float(weights[1])}
    print(f"Class weights: Ham={cw[0]:.3f}, Spam={cw[1]:.3f}")
    return cw


def split_dataset(df: pd.DataFrame):
    """Stratified train / val / test split."""
    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED, stratify=df["label"]
    )
    val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_ratio,
        random_state=config.RANDOM_SEED, stratify=train_val["label"]
    )

    print(f"\nData splits:")
    print(f"  Train : {len(train):,}  (spam: {train['label'].sum()})")
    print(f"  Val   : {len(val):,}   (spam: {val['label'].sum()})")
    print(f"  Test  : {len(test):,}   (spam: {test['label'].sum()})")
    return train, val, test


if __name__ == "__main__":
    df = load_dataset()
    train, val, test = split_dataset(df)
    cw = compute_weights(train["label"])