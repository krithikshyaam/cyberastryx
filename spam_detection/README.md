# Email Spam Detection - LLM Pipeline

A two-stage email spam detection system built with TensorFlow/Keras.

## Project Structure

```
spam_detection/
├── README.md
├── requirements.txt
├── data/
│   └── your_dataset.csv        ← Place your dataset here
├── src/
│   ├── config.py               ← All hyperparameters & paths
│   ├── data_loader.py          ← Load & preprocess your dataset
│   ├── preprocessor.py         ← Text cleaning & tokenization
│   ├── models/
│   │   ├── baseline_model.py   ← Stage 1: BiLSTM baseline
│   │   └── transformer_model.py← Stage 2: Fine-tuned BERT
│   ├── trainer.py              ← Training loop with callbacks
│   ├── evaluator.py            ← Metrics, confusion matrix, report
│   └── predict.py              ← Inference on new emails
├── notebooks/
│   └── exploration.ipynb       ← EDA notebook
└── outputs/
    ├── models/                 ← Saved model weights
    └── reports/                ← Evaluation reports & plots
```

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset
Place your CSV in `data/`. Expected format:
```
text,label
"You won a free prize!",1
"Meeting at 3pm tomorrow",0
```
Update `src/config.py` with your column names if different.

### 3. Run Stage 1 — Baseline (BiLSTM)
```bash
python src/trainer.py --model baseline
```

### 4. Run Stage 2 — Transformer (BERT fine-tuning)
```bash
python src/trainer.py --model transformer
```

### 5. Predict on new emails
```bash
python src/predict.py --model baseline --text "Congratulations! You've won $1000!"
python src/predict.py --model transformer --text "Hi, can we schedule a call tomorrow?"
```

## Dataset Format

Your CSV must have at minimum:
- A **text column** (email body/subject) — default: `text`
- A **label column** (0 = ham, 1 = spam) — default: `label`

Update `config.py` → `TEXT_COL` and `LABEL_COL` to match your column names.

## Model Stages

| Stage | Model | Accuracy (typical) | Training Time |
|-------|-------|--------------------|---------------|
| 1 | BiLSTM + Embeddings | ~97% | ~5 min |
| 2 | BERT fine-tuned | ~99%+ | ~30 min |

## GPU Usage
Both models auto-detect GPU. BERT fine-tuning strongly benefits from GPU.
