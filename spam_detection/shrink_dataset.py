"""
shrink_dataset.py — reduce the Cyberastryx training corpus on disk.

The big win: MAX_SEQ_LEN is 256, so the trainer already throws away
everything past ~256 tokens. Storing full email bodies (with quoted
reply chains, signatures, and disclaimers) wastes most of the file.

Usage:
    python shrink_dataset.py data/processed_data_clean.csv

Writes:
    data/processed_data_clean.parquet    <- use this going forward
"""

import sys
import os
import pandas as pd

TEXT_COL = "full_text"
LABEL_COL = "label"
MAX_SEQ_LEN = 256          # must match your trainer config
CHARS_PER_TOKEN = 6        # generous upper bound for English
KEEP_CHARS = MAX_SEQ_LEN * CHARS_PER_TOKEN   # ~1536 chars


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main(src):
    if not os.path.exists(src):
        sys.exit(f"Not found: {src}")

    before = os.path.getsize(src)
    print(f"Reading {src} ({human(before)})")

    df = pd.read_csv(src)
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")

    # 1. Drop every column the trainer doesn't read
    keep = [c for c in (TEXT_COL, LABEL_COL) if c in df.columns]
    if TEXT_COL not in keep:
        sys.exit(f"Missing text column '{TEXT_COL}'")
    dropped = [c for c in df.columns if c not in keep]
    df = df[keep]
    if dropped:
        print(f"  dropped columns: {dropped}")

    # 2. Truncate to what the model actually reads
    orig_chars = df[TEXT_COL].str.len().sum()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.slice(0, KEEP_CHARS)
    new_chars = df[TEXT_COL].str.len().sum()
    print(f"  truncated to {KEEP_CHARS} chars: "
          f"{human(orig_chars)} -> {human(new_chars)} of text")

    # 3. Remove exact duplicates created by truncation
    n0 = len(df)
    df = df.drop_duplicates(subset=[TEXT_COL])
    if len(df) < n0:
        print(f"  removed {n0 - len(df):,} duplicate rows after truncation")

    # 4. Narrow the label dtype
    if LABEL_COL in df.columns:
        df[LABEL_COL] = df[LABEL_COL].astype("int8")

    # 5. Write columnar + compressed
    dst = os.path.splitext(src)[0] + ".parquet"
    try:
        df.to_parquet(dst, compression="zstd", index=False)
    except Exception:
        # zstd needs a recent pyarrow; snappy is always available
        df.to_parquet(dst, compression="snappy", index=False)
        print("  (zstd unavailable, used snappy)")

    after = os.path.getsize(dst)
    print(f"\nWrote {dst} ({human(after)})")
    print(f"Reduction: {before / after:.1f}x smaller, "
          f"{len(df):,} rows retained")
    print("\nUpdate your trainer config:")
    print(f'    DATA_PATH = "{dst.replace(os.sep, "/")}"')
    print("    # and swap pd.read_csv(...) for pd.read_parquet(...)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python shrink_dataset.py <path-to-csv>")
    main(sys.argv[1])