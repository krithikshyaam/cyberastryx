"""
baseline_model.py - Stage 1: BiLSTM spam classifier.

Architecture:
  Embedding → SpatialDropout → BiLSTM → GlobalMaxPool → Dense → Output

Fast to train, strong baseline (~97% accuracy on clean datasets).
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from src import config


def build_bilstm_model(vocab_size: int) -> tf.keras.Model:
    """
    Build and compile the BiLSTM spam classifier.

    Args:
        vocab_size: Size of the vocabulary (from SpamTokenizer.vocab_size)

    Returns:
        Compiled Keras model
    """
    cfg = config.BASELINE

    inputs = layers.Input(shape=(config.MAX_SEQ_LEN,), name="token_ids")

    # Embedding layer
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=cfg["embedding_dim"],
        name="embedding"
    )(inputs)
    x = layers.SpatialDropout1D(cfg["dropout_rate"])(x)

    # Bidirectional LSTM — captures context from both directions
    x = layers.Bidirectional(
        layers.LSTM(cfg["lstm_units"], return_sequences=True),
        name="bi_lstm"
    )(x)

    # Global max pooling extracts the most salient features
    x = layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    # Classifier head
    x = layers.Dense(cfg["dense_units"], activation="relu", name="dense")(x)
    x = layers.Dropout(cfg["dropout_rate"])(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs, outputs, name="BiLSTM_SpamDetector")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    model.summary()
    return model


def get_baseline_callbacks(model_path: str = config.BASELINE_MODEL_PATH):
    """Return training callbacks: early stopping + model checkpoint."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=config.BASELINE["patience"],
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path + "/best_model.keras",
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="outputs/logs/baseline",
            histogram_freq=1
        ),
    ]


def load_baseline_model(model_path: str = config.BASELINE_MODEL_PATH):
    """Load a saved baseline model."""
    path = model_path + "/best_model.keras"
    model = tf.keras.models.load_model(path)
    print(f"Baseline model loaded ← {path}")
    return model
