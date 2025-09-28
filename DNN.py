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

CSV = "rf_gini_selected.csv"  
LABEL_COL = None  
HIDDEN = [512, 256, 128]
DROPOUT = 0.3
L2 = 0.0
EPOCHS = 50
BATCH_SIZE = 512
RANDOM_STATE = 42
MODEL_PATH = "dnn_model.keras"  
METRICS_FILE = "dnn_metrics.json"
REPORT_CSV = "dnn_classification_report.csv"
CM_CSV = "dnn_confusion_matrix.csv"
USE_CLASS_WEIGHTS = True  # Set False to disable balancing
SCALE_MINMAX = True  # Scale features to [0,1] using train-only fit
REPORT_TXT = "dnn_classification_report.txt"
CM_PNG = "dnn_confusion_matrix.png"


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
			raise SystemExit("Không xác định được cột nhãn. Dùng --label-col để chỉ định.")

	df = pd.read_csv(csv_path)
	if label_col not in df.columns:
		raise SystemExit(f"Không tìm thấy cột nhãn '{label_col}'.")

	y = df[label_col].astype(str)
	X = df.drop(columns=[label_col])
	return X, y, label_col


def build_dnn(input_dim: int, n_classes: int, hidden: List[int], dropout: float, l2: float) -> keras.Model:
	reg = keras.regularizers.l2(l2) if l2 and l2 > 0 else None
	inputs = keras.Input(shape=(input_dim,), dtype=tf.float32)
	x = inputs
	for units in hidden:
		x = layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
		if dropout and dropout > 0:
			x = layers.Dropout(dropout)(x)

	if n_classes == 2:
		outputs = layers.Dense(1)(x)  
		loss = keras.losses.BinaryCrossentropy(from_logits=True)
		metrics = ["accuracy"]
	else:
		outputs = layers.Dense(n_classes)(x)  
		loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		metrics = ["accuracy"]

	model = keras.Model(inputs, outputs)
	model.compile(optimizer=keras.optimizers.Adam(clipnorm=1.0), loss=loss, metrics=metrics)
	return model


def main():
	print("[DNN] Bắt đầu train với cấu hình mặc định. Sửa phần config đầu file để thay đổi.")

	# Optional: set memory growth on GPUs if present
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
	print(f"[DNN] Số lớp: {n_classes} | Phân bố lớp (tỉ lệ): {dist.to_dict()} | Baseline đoán lớp lớn nhất: {majority_baseline:.4f}")


	X = X_df.to_numpy(dtype=np.float32)

	
	X_trainval, X_test, y_trainval, y_test = train_test_split(
		X, y_int, test_size=0.10, stratify=y_int, random_state=RANDOM_STATE
	)

	X_train, X_val, y_train, y_val = train_test_split(
		X_trainval,
		y_trainval,
		test_size=2 / 9,
		stratify=y_trainval,
		random_state=RANDOM_STATE,
	)

	
	if SCALE_MINMAX:
		scaler = MinMaxScaler()
		X_trainval_tmp, X_test_tmp, y_trainval_tmp, y_test_tmp = X_trainval, X_test, y_trainval, y_test
		scaler.fit(X_trainval_tmp)
		X_trainval = scaler.transform(X_trainval_tmp)
		X_test = scaler.transform(X_test_tmp)
	
		X_train, X_val, y_train, y_val = train_test_split(
			X_trainval,
			y_trainval,
			test_size=2 / 9,
			stratify=y_trainval,
			random_state=RANDOM_STATE,
		)

	model = build_dnn(input_dim=X.shape[1], n_classes=n_classes, hidden=HIDDEN, dropout=DROPOUT, l2=L2)

	callbacks = [
		keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
		keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
	]

	y_train_target = y_train if n_classes > 2 else y_train.astype(np.float32)
	y_val_target = y_val if n_classes > 2 else y_val.astype(np.float32)

	class_weight = None
	if USE_CLASS_WEIGHTS:
		classes = np.arange(n_classes)
		weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
		class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
		print(f"[DNN] class_weight: {class_weight}")

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

	# Evaluate on test
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

	# Classification report and confusion matrix (label names)
	target_names = [str(c) for c in le.classes_]
	report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
	report_str = classification_report(y_test, y_pred, target_names=target_names, digits=4, zero_division=0)
	cm = confusion_matrix(y_test, y_pred)
	print("\nEvaluation Report:\n" + report_str)

	# Save model (Keras v3): use a file path with extension
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
	}
	Path(METRICS_FILE).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

	# Save report and confusion matrix as CSV/Text/Image
	pd.DataFrame(report_dict).to_csv(REPORT_CSV, index=True)
	pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(CM_CSV)
	Path(REPORT_TXT).write_text("Evaluation Report:\n" + report_str + "\n", encoding="utf-8")

	# Confusion matrix heatmap
	cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
	w = max(8, int(0.9 * len(target_names)) + 6)
	h = max(6, int(0.9 * len(target_names)) + 4)
	plt.figure(figsize=(w, h))
	sns.heatmap(cm_df, annot=True, fmt="d", cmap="magma", cbar=True)
	plt.title("Confusion Matrix")
	plt.ylabel("True label")
	plt.xlabel("Predicted label")
	plt.tight_layout()
	plt.savefig(CM_PNG, dpi=200)
	plt.close()


if __name__ == "__main__":
	main()

