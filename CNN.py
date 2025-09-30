import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# === Config ===
CSV = "rf_gini_selected.csv"
LABEL_COL = None 
EPOCHS = 60
BATCH_SIZE = 512
RANDOM_STATE = 42
USE_CLASS_WEIGHTS = True
SCALE_MINMAX = True
MODEL_PATH = "cnn_model.keras"
METRICS_FILE = "cnn_metrics.json"
REPORT_CSV = "cnn_classification_report.csv"
CM_CSV = "cnn_confusion_matrix.csv"
REPORT_TXT = "cnn_classification_report.txt"
CM_PNG = "cnn_confusion_matrix.png"

# CNN (1D)
CONV_BLOCKS = [
    # filters, kernel_size, pool_size, dropout
    (64, 5, 2, 0.1),
    (128, 5, 2, 0.1),
]
DENSE = [256, 128]
DROPOUT_DENSE = 0.3
L2 = 0.0

SMOOTH_CLASS_WEIGHTS = True 
LABEL_SMOOTHING = 0.05
INITIAL_LR = 3e-4
MONITOR_METRIC = "val_macro_f1"


def detect_label(cols: List[str]) -> Optional[str]:
    for cand in ("family", "label", "target", "y", "class"):
        if cand in cols:
            return cand
    return None


def load_dataset(csv_path: Path, label_col: Optional[str]) -> Tuple[pd.DataFrame, pd.Series, str]:
    cols = list(pd.read_csv(csv_path, nrows=0).columns)
    if label_col is None:
        label_col = detect_label(cols)
        if label_col is None:
            raise SystemExit("Không xác định được cột nhãn. Chỉ định LABEL_COL trong file hoặc sửa mã.")

    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise SystemExit(f"Không tìm thấy cột nhãn '{label_col}'.")

    y = df[label_col].astype(str)
    X = df.drop(columns=[label_col])
    return X, y, label_col


def build_cnn(input_len: int, n_classes: int, use_onehot: bool) -> keras.Model:
    reg = keras.regularizers.l2(L2) if L2 and L2 > 0 else None
    inputs = keras.Input(shape=(input_len, 1), dtype=tf.float32)
    x = inputs
    for (filters, k, pool, drop) in CONV_BLOCKS:
        x = layers.Conv1D(filters, k, padding="same", activation="relu", kernel_regularizer=reg)(x)
        if pool and pool > 1:
            x = layers.MaxPool1D(pool)(x)
        if drop and drop > 0:
            x = layers.Dropout(drop)(x)
    x = layers.Flatten()(x)
    for units in DENSE:
        x = layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
        if DROPOUT_DENSE and DROPOUT_DENSE > 0:
            x = layers.Dropout(DROPOUT_DENSE)(x)

    if n_classes == 2:
        outputs = layers.Dense(1)(x)
        loss = keras.losses.BinaryCrossentropy(from_logits=True,
                                               label_smoothing=LABEL_SMOOTHING if LABEL_SMOOTHING else 0.0)
    else:
        outputs = layers.Dense(n_classes)(x)
        if use_onehot:
            loss = keras.losses.CategoricalCrossentropy(from_logits=True,
                                                        label_smoothing=LABEL_SMOOTHING if LABEL_SMOOTHING else 0.0)
        else:
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR, clipnorm=1.0),
        loss=loss,
        metrics=["accuracy"]
    )
    return model

class MacroF1Callback(keras.callbacks.Callback):

    def __init__(self, X_val, y_val, n_classes, verbose: int = 0):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.n_classes = n_classes
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Dự đoán full validation
        logits = self.model.predict(self.X_val, verbose=0)
        if self.n_classes == 2:
            probs = 1 / (1 + np.exp(-logits.ravel()))
            y_pred = (probs >= 0.5).astype(int)
        else:
            y_pred = np.argmax(logits, axis=1)
        macro_f1 = f1_score(self.y_val, y_pred, average="macro", zero_division=0)
        logs["val_macro_f1"] = macro_f1  # đảm bảo xuất hiện cho EarlyStopping & ReduceLROnPlateau
        if self.verbose:
            print(f"\n[MacroF1Callback] val_macro_f1={macro_f1:.4f}")

def main():
    print("[CNN] Bắt đầu train CNN 1D trên dữ liệu IG.")

    # GPU memory growth
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    csv_path = Path(CSV)
    if not csv_path.exists():
        raise SystemExit(f"Không tìm thấy file: {csv_path}")

    X_df, y_series, label_col = load_dataset(csv_path, LABEL_COL)

    le = LabelEncoder()
    y_int = le.fit_transform(y_series)
    n_classes = len(le.classes_)

    vc = pd.Series(y_int).value_counts().sort_index()
    dist = (vc / vc.sum()).round(4)
    majority_baseline = float(dist.max())
    print(f"[CNN] Số lớp: {n_classes} | Phân bố: {dist.to_dict()} | Majority baseline: {majority_baseline:.4f}")

    X = X_df.to_numpy(dtype=np.float32)
    n_features = X.shape[1]

    # Split 70/20/10
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_int, test_size=0.10, stratify=y_int, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=2 / 9, stratify=y_trainval, random_state=RANDOM_STATE
    )

    if SCALE_MINMAX:
        scaler = MinMaxScaler()
        scaler.fit(X_trainval)
        X_trainval_scaled = scaler.transform(X_trainval)
        X_test = scaler.transform(X_test)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval_scaled, y_trainval, test_size=2 / 9, stratify=y_trainval, random_state=RANDOM_STATE
        )
    else:
        X_trainval_scaled = X_trainval

    X_train = X_train.reshape((-1, n_features, 1))
    X_val = X_val.reshape((-1, n_features, 1))
    X_test = X_test.reshape((-1, n_features, 1))

    use_onehot = (n_classes > 2 and LABEL_SMOOTHING and LABEL_SMOOTHING > 0)
    model = build_cnn(n_features, n_classes, use_onehot)

    macro_cb = MacroF1Callback(X_val, y_val, n_classes, verbose=0)

    callbacks = [
        macro_cb,
        keras.callbacks.ReduceLROnPlateau(monitor=MONITOR_METRIC, mode="max", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor=MONITOR_METRIC, mode="max", patience=6, restore_best_weights=True, verbose=1),
    ]

    if use_onehot:
        y_train_target = to_categorical(y_train, num_classes=n_classes).astype(np.float32)
        y_val_target = to_categorical(y_val, num_classes=n_classes).astype(np.float32)
    else:
        y_train_target = y_train if n_classes > 2 else y_train.astype(np.float32)
        y_val_target = y_val if n_classes > 2 else y_val.astype(np.float32)

    class_weight = None
    if USE_CLASS_WEIGHTS:
        classes = np.arange(n_classes)
        raw_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        if SMOOTH_CLASS_WEIGHTS:
            smooth_w = np.power(raw_w, 0.5)
            smooth_w = smooth_w / smooth_w.mean()
            weights = smooth_w
        else:
            weights = raw_w
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
        print(f"[CNN] class_weight (smoothed={SMOOTH_CLASS_WEIGHTS}): {class_weight}")

    history = model.fit(
        X_train,
        y_train_target,
        validation_data=(X_val, y_val_target),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    # Evaluate
    if n_classes == 2:
        logits = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
        y_pred_prob = 1 / (1 + np.exp(-logits))
        y_pred = (y_pred_prob >= 0.5).astype(int)
    else:
        logits = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        y_pred_prob = tf.nn.softmax(logits, axis=1).numpy()
        y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 (macro): {f1:.4f}")

    target_names = [str(c) for c in le.classes_]
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred, target_names=target_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    print("\nEvaluation Report:\n" + report_str)

    model.save(MODEL_PATH)

    metrics = {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "n_classes": n_classes,
        "label_col": label_col,
        "classes": target_names,
        "history_keys": list(history.history.keys()),
        "n_features": n_features,
        "conv_blocks": CONV_BLOCKS,
        "dense_layers": DENSE,
    }
    Path(METRICS_FILE).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    pd.DataFrame(report_dict).to_csv(REPORT_CSV, index=True)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(CM_CSV)
    Path(REPORT_TXT).write_text("Evaluation Report:\n" + report_str + "\n", encoding="utf-8")

    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    w = max(8, int(0.9 * len(target_names)) + 6)
    h = max(6, int(0.9 * len(target_names)) + 4)
    plt.figure(figsize=(w, h))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="magma", cbar=True)
    plt.title("Confusion Matrix (CNN)")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
