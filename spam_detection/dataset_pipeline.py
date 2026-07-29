"""
dataset_pipeline.py - Production multi-dataset loader.

Combines 4 datasets into one unified training corpus:
  1. SMS Spam Collection  (your existing spam.csv)
  2. Enron Email Dataset  (30,000+ corporate emails)
  3. SpamAssassin Public  (6,000 emails)
  4. TREC 2007            (75,000 emails)

Each dataset is normalized to: {"text": "...", "label": 0|1, "source": "..."}

Usage:
    python dataset_pipeline.py --download        # download all datasets
    python dataset_pipeline.py --merge           # merge into unified CSV
    python dataset_pipeline.py --download --merge --stats  # full pipeline + stats
"""

import os
import re
import sys
import json
import email
import tarfile
import zipfile
import hashlib
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR     = Path("data/raw")
UNIFIED_CSV = Path("data/processed_data_clean.csv")
STATS_FILE  = Path("data/dataset_stats.json")

RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Download helper ───────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download with progress bar. Returns True if downloaded, False if skipped."""
    if dest.exists():
        print(f"  ✓ Already downloaded: {dest.name}")
        return False
    print(f"  ↓ Downloading {desc}...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {desc}: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ── Dataset 1: SMS Spam Collection (already have it) ─────────────────────────

def load_sms_spam(path: str = "data/spam.csv") -> pd.DataFrame:
    print("\n[1/4] Loading SMS Spam Collection...")
    df = pd.read_csv(path)
    df = df.rename(columns={"Message": "text", "Category": "label_str"})
    df["label"] = df["label_str"].map({"ham": 0, "spam": 1})
    df = df[["text", "label"]].dropna()
    df["source"] = "sms_spam_collection"
    print(f"  Loaded: {len(df):,} messages (spam: {df.label.sum():,})")
    return df

def load_processed_email(path: str = "data/processed_data_clean.csv") -> pd.DataFrame:
    """Load the cleaned Enron/TREC email dataset."""
    print("\n[2/4] Loading processed email dataset...")
    df = pd.read_csv(path, usecols=["label", "full_text"])
    df = df.rename(columns={"full_text": "text"})
    df = df.dropna(subset=["text"])
    df["label"] = df["label"].astype(int)
    df["source"] = "enron_trec_clean"
    print(f"  Loaded: {len(df):,} emails (spam: {df.label.sum():,})")
    return df


# ── Dataset 2: Enron Email Dataset ───────────────────────────────────────────

ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"

def download_enron():
    dest = RAW_DIR / "enron_mail.tar.gz"
    download_file(ENRON_URL, dest, "Enron Email Dataset (~423MB)")
    return dest

def parse_email_body(raw: str) -> str:
    """Extract plain text body from raw email string."""
    try:
        msg = email.message_from_string(raw)
        parts = []
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        parts.append(payload.decode("utf-8", errors="replace"))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                parts.append(payload.decode("utf-8", errors="replace"))
        body = " ".join(parts)
        # Get subject too
        subject = msg.get("Subject", "")
        return f"{subject} {body}".strip()
    except Exception:
        return raw[:2000]

def load_enron(max_per_class: int = 15000) -> pd.DataFrame:
    """
    Load Enron dataset.
    Folder structure after extraction:
      maildir/<user>/spam/   → spam emails
      maildir/<user>/ham/    → ham emails
    Falls back to synthetic generation if not downloaded.
    """
    print("\n[2/4] Loading Enron Email Dataset...")
    tar_path = RAW_DIR / "enron_mail.tar.gz"

    if not tar_path.exists():
        print("  ⚠ Enron tar not found. Run with --download first.")
        print("  Using Enron-lite (preprocessed CSV fallback)...")
        return _load_enron_lite()

    records = []
    print("  Extracting emails (this may take a few minutes)...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            spam_count = ham_count = 0
            for member in tqdm(members, desc="  Parsing"):
                path_parts = Path(member.name).parts
                # Enron label detection: folders named "spam" or "ham" or "inbox"
                label = None
                for part in path_parts:
                    if part.lower() in ("spam", "junk"):
                        label = 1
                        break
                    elif part.lower() in ("ham", "inbox", "sent"):
                        label = 0
                        break
                if label is None:
                    continue
                if label == 1 and spam_count >= max_per_class:
                    continue
                if label == 0 and ham_count >= max_per_class:
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue
                raw = f.read().decode("utf-8", errors="replace")
                body = parse_email_body(raw)
                if len(body.strip()) < 5:
                    continue

                records.append({"text": body[:3000], "label": label, "source": "enron"})
                if label == 1:
                    spam_count += 1
                else:
                    ham_count += 1

    except Exception as e:
        print(f"  ✗ Extraction error: {e}")
        return _load_enron_lite()

    df = pd.DataFrame(records)
    print(f"  Loaded: {len(df):,} emails (spam: {df.label.sum():,})")
    return df


def _load_enron_lite() -> pd.DataFrame:
    """
    Fallback: load preprocessed Enron CSV if available, or return empty.
    Place 'enron_lite.csv' in data/raw/ if you have a preprocessed version.
    (columns: text, label)
    """
    lite_path = RAW_DIR / "enron_lite.csv"
    if lite_path.exists():
        df = pd.read_csv(lite_path)
        df["source"] = "enron_lite"
        df = df[["text", "label", "source"]].dropna()
        df["label"] = df["label"].astype(int)
        print(f"  Loaded Enron-lite: {len(df):,} emails")
        return df
    print("  No Enron data available. Skipping.")
    return pd.DataFrame(columns=["text", "label", "source"])


# ── Dataset 3: SpamAssassin ───────────────────────────────────────────────────

SPAMASSASSIN_URLS = {
    "easy_ham"  : "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "hard_ham"  : "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
    "spam"      : "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
    "spam_2"    : "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
}

def download_spamassassin():
    for name, url in SPAMASSASSIN_URLS.items():
        dest = RAW_DIR / f"spamassassin_{name}.tar.bz2"
        download_file(url, dest, f"SpamAssassin {name}")

def load_spamassassin() -> pd.DataFrame:
    print("\n[3/4] Loading SpamAssassin Dataset...")
    records = []

    for name, _ in SPAMASSASSIN_URLS.items():
        path = RAW_DIR / f"spamassassin_{name}.tar.bz2"
        if not path.exists():
            print(f"  ⚠ {path.name} not found — skipping.")
            continue
        label = 1 if "spam" in name else 0
        try:
            with tarfile.open(path, "r:bz2") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    raw = f.read().decode("utf-8", errors="replace")
                    body = parse_email_body(raw)
                    if len(body.strip()) > 5:
                        records.append({
                            "text"  : body[:3000],
                            "label" : label,
                            "source": "spamassassin"
                        })
        except Exception as e:
            print(f"  ✗ Error reading {name}: {e}")

    if not records:
        print("  No SpamAssassin data loaded. Run with --download first.")
        return pd.DataFrame(columns=["text", "label", "source"])

    df = pd.DataFrame(records)
    print(f"  Loaded: {len(df):,} emails (spam: {df.label.sum():,})")
    return df


# ── Dataset 4: TREC 2007 ──────────────────────────────────────────────────────

def load_trec(path: Optional[str] = None) -> pd.DataFrame:
    """
    TREC 2007 requires manual download (registration).
    Place trec07p/ folder in data/raw/ after downloading from:
    https://plg.uwaterloo.ca/~gvcormac/treccorpus07/

    Alternatively place a preprocessed 'trec_2007.csv' in data/raw/
    """
    print("\n[4/4] Loading TREC 2007 Dataset...")

    # Check for preprocessed CSV first
    csv_path = RAW_DIR / "trec_2007.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["source"] = "trec_2007"
        df = df[["text", "label", "source"]].dropna()
        df["label"] = df["label"].astype(int)
        print(f"  Loaded TREC 2007: {len(df):,} emails")
        return df

    # Check for raw TREC folder
    trec_dir = RAW_DIR / "trec07p"
    if not trec_dir.exists():
        print("  ⚠ TREC 2007 not found.")
        print("    Download from: https://plg.uwaterloo.ca/~gvcormac/treccorpus07/")
        print("    Extract to: data/raw/trec07p/")
        return pd.DataFrame(columns=["text", "label", "source"])

    records = []
    index_file = trec_dir / "full" / "index"
    if not index_file.exists():
        print(f"  ✗ Index file not found at {index_file}")
        return pd.DataFrame(columns=["text", "label", "source"])

    with open(index_file) as f:
        for line in tqdm(f, desc="  Parsing TREC"):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            label_str, rel_path = parts[0], parts[1]
            label = 1 if label_str.lower() == "spam" else 0
            full_path = trec_dir / rel_path.lstrip("../")
            if not full_path.exists():
                continue
            try:
                raw = full_path.read_text(errors="replace")
                body = parse_email_body(raw)
                if len(body.strip()) > 5:
                    records.append({"text": body[:3000], "label": label, "source": "trec_2007"})
            except Exception:
                continue

    if not records:
        return pd.DataFrame(columns=["text", "label", "source"])

    df = pd.DataFrame(records)
    print(f"  Loaded TREC 2007: {len(df):,} emails (spam: {df.label.sum():,})")
    return df


# ── Merge & deduplicate ───────────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove near-duplicate texts using hash fingerprinting."""
    original = len(df)
    # Normalize whitespace for hashing
    df["_hash"] = df["text"].str.lower().str.split().str.join(" ").str[:500].apply(
        lambda x: hashlib.md5(x.encode()).hexdigest()
    )
    df = df.drop_duplicates(subset="_hash").drop(columns="_hash")
    removed = original - len(df)
    if removed:
        print(f"  Removed {removed:,} duplicates")
    return df


def clean_for_merge(text: str, max_len: int = 2000) -> str:
    """Light cleaning for merged dataset."""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def merge_datasets(
    sms_path: str = "data/spam.csv",
    output_path: str = str(UNIFIED_CSV),
    max_enron: int = 15000,
) -> pd.DataFrame:
    """Load all datasets, merge, deduplicate, and save."""
    print("\n" + "="*60)
    print("  DATASET MERGE PIPELINE")
    print("="*60)

    dfs = []

    # 1. SMS Spam
    try:
        dfs.append(load_sms_spam(sms_path))
    except Exception as e:
        print(f"  ✗ SMS Spam error: {e}")

    # 2. Enron
    enron_df = load_enron(max_per_class=max_enron)
    if len(enron_df):
        dfs.append(enron_df)

    # 3. SpamAssassin
    sa_df = load_spamassassin()
    if len(sa_df):
        dfs.append(sa_df)

    # 4. TREC
    trec_df = load_trec()
    if len(trec_df):
        dfs.append(trec_df)

    if not dfs:
        raise RuntimeError("No datasets loaded!")

    # Merge
    unified = pd.concat(dfs, ignore_index=True)
    unified["text"] = unified["text"].apply(clean_for_merge)
    unified["label"] = unified["label"].astype(int)

    # Deduplicate
    print(f"\nBefore dedup: {len(unified):,} total samples")
    unified = deduplicate(unified)
    print(f"After  dedup: {len(unified):,} total samples")

    # Stats
    stats = {
        "total"   : len(unified),
        "ham"     : int((unified.label == 0).sum()),
        "spam"    : int(unified.label.sum()),
        "sources" : unified.groupby("source").agg(
            count=("label", "count"),
            spam=("label", "sum")
        ).to_dict()
    }

    # Print summary
    print(f"\n{'─'*50}")
    print(f"  UNIFIED DATASET SUMMARY")
    print(f"{'─'*50}")
    print(f"  Total  : {stats['total']:,}")
    print(f"  Ham    : {stats['ham']:,} ({stats['ham']/stats['total']*100:.1f}%)")
    print(f"  Spam   : {stats['spam']:,} ({stats['spam']/stats['total']*100:.1f}%)")
    print(f"\n  By source:")
    for src, row in unified.groupby("source").agg(
        count=("label","count"), spam=("label","sum")
    ).iterrows():
        print(f"    {src:<25} {row['count']:>6,} msgs  ({row['spam']:,} spam)")
    print(f"{'─'*50}")

    # Save
    unified.to_csv(output_path, index=False)
    print(f"\n  Saved → {output_path}")

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats  → {STATS_FILE}")

    return unified


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-dataset pipeline for spam detection")
    parser.add_argument("--download", action="store_true", help="Download all available datasets")
    parser.add_argument("--merge",    action="store_true", help="Merge all datasets into unified CSV")
    parser.add_argument("--stats",    action="store_true", help="Print dataset statistics")
    parser.add_argument("--max-enron", type=int, default=15000, help="Max Enron emails per class")
    args = parser.parse_args()

    if args.download:
        print("\n=== Downloading Datasets ===")
        print("Downloading SpamAssassin (Enron & TREC require manual steps)...")
        download_spamassassin()
        print("\nFor Enron (~423MB):")
        print(f"  wget '{ENRON_URL}' -O data/raw/enron_mail.tar.gz")
        print("\nFor TREC 2007 (requires registration):")
        print("  https://plg.uwaterloo.ca/~gvcormac/treccorpus07/")

    if args.merge:
        df = merge_datasets(max_enron=args.max_enron)

    if args.stats:
        if UNIFIED_CSV.exists():
            df = pd.read_csv(UNIFIED_CSV)
            print(f"\nUnified dataset: {len(df):,} samples")
            print(df.groupby(["source","label"]).size().unstack(fill_value=0))
        else:
            print("No unified dataset found. Run with --merge first.")

    if not any([args.download, args.merge, args.stats]):
        parser.print_help()
