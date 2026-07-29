"""
config.py - Central configuration for all hyperparameters and paths.
Edit this file to match your dataset and hardware.
"""

import os

# ─────────────────────────────────────────────
# DATASET SETTINGS
# ─────────────────────────────────────────────
DATA_PATH = "data/processed_data_clean.parquet"          # path to your CSV file
TEXT_COL      = "full_text"               # column name for email/SMS text
LABEL_COL     = "label"                   # column name for labels ("ham" / "spam")
ENCODING      = "utf-8"                   # CSV file encoding

# Class imbalance: dataset is ~58% spam / ~42% ham
# We compute class weights to penalize misclassifying spam more heavily
USE_CLASS_WEIGHTS = True

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
MAX_VOCAB_SIZE = 10_000    # max tokens in vocabulary (baseline model)
MAX_SEQ_LEN = 256          # emails avg ~1300 chars, 256 tokens captures most content
TEST_SIZE      = 0.15      # fraction for test set
VAL_SIZE       = 0.10      # fraction for validation set
RANDOM_SEED    = 42

# ─────────────────────────────────────────────
# STAGE 1 — BiLSTM BASELINE
# ─────────────────────────────────────────────
BASELINE = {
    "embedding_dim"  : 128,
    "lstm_units"     : 64,
    "dropout_rate"   : 0.3,
    "dense_units"    : 64,
    "batch_size"     : 64,
    "epochs"         : 15,
    "learning_rate"  : 1e-3,
    "patience"       : 3,       # early stopping patience
}

# ─────────────────────────────────────────────
# STAGE 2 — BERT FINE-TUNING
# ─────────────────────────────────────────────
TRANSFORMER = {
    "model_name"     : "bert-base-uncased",   # HuggingFace model id
    "max_length"     : 128,                   # BERT token limit
    "batch_size"     : 16,
    "epochs"         : 4,
    "learning_rate"  : 2e-5,
    "warmup_ratio"   : 0.1,
    "weight_decay"   : 0.01,
    "patience"       : 2,
}

# ─────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────
BASELINE_MODEL_PATH     = "outputs/models/baseline_bilstm"
TRANSFORMER_MODEL_PATH  = "outputs/models/bert_spam"
TOKENIZER_PATH          = "outputs/models/tokenizer.json"
REPORTS_DIR             = "outputs/reports"

# ─────────────────────────────────────────────
# GPU SETTINGS
# ─────────────────────────────────────────────
# Set to -1 to force CPU, or "0","1"... to pick a specific GPU
GPU_DEVICE = "0"
MIXED_PRECISION = True   # fp16 training (speeds up GPU training)
