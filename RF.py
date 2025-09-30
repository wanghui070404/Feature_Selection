import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

CSV = "ig_selected.csv"         
LABEL_COL = None                      
TEST_SIZE = 0.10                      # Tỉ lệ test tổng thể
VAL_RATIO_WITHIN_NONTEST = 2/9        # 70/20/10 (train/val/test)
RANDOM_STATE = 42
SCALE_MINMAX = True                   
CLASS_WEIGHT = "balanced"            
N_ESTIMATORS = 800
MAX_DEPTH = None
MAX_FEATURES = "sqrt"                
MIN_SAMPLES_LEAF = 1
N_JOBS = -1
MODEL_PATH = "rf_model.joblib"      
METRICS_FILE = "rf_metrics.json"
FEATURE_IMPORTANCE_CSV = "rf_feature_importances.csv"
REPORT_CSV = "rf_classification_report.csv"
REPORT_TXT = "rf_classification_report.txt"
CM_CSV = "rf_confusion_matrix.csv"
CM_PNG = "rf_confusion_matrix.png"
TOP_N_IMPORTANCE_PLOT = 40           
SAVE_PROBAS = True                    
PROBAS_CSV = "rf_test_probabilities.csv"
SEED_REPRO = True                    


try:
    import joblib
except ImportError:  
    joblib = None


def detect_label(cols: List[str]) -> Optional[str]:
    for cand in ("family", "label", "target", "y", "class"):
        if cand in cols:
            return cand
    return None


def load_dataset(csv_path: Path, label_col: Optional[str]):
    cols = list(pd.read_csv(csv_path, nrows=0).columns)
    if label_col is None:
        label_col = detect_label(cols)
        if label_col is None:
            raise SystemExit("Không xác định được cột nhãn. Chỉ định LABEL_COL.")
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise SystemExit(f"Không tìm thấy cột nhãn '{label_col}' trong file.")
    y = df[label_col].astype(str)
    X = df.drop(columns=[label_col])
    return X, y, label_col


def main():
    print("[RF] Bắt đầu train RandomForest.")
    csv_path = Path(CSV)
    if not csv_path.exists():
        raise SystemExit(f"Không tìm thấy file: {csv_path}")

    if SEED_REPRO:
        np.random.seed(RANDOM_STATE)

    X_df, y_series, label_col = load_dataset(csv_path, LABEL_COL)

    le = LabelEncoder()
    y_int = le.fit_transform(y_series)
    n_classes = len(le.classes_)

    vc = pd.Series(y_int).value_counts().sort_index()
    dist = (vc / vc.sum()).round(4)
    print(f"[RF] Số lớp: {n_classes} | Phân bố: {dist.to_dict()}")

    X = X_df.to_numpy(dtype=np.float32)  

    # Split test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_int, test_size=TEST_SIZE, stratify=y_int, random_state=RANDOM_STATE
    )
    # Split train/val from remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_RATIO_WITHIN_NONTEST, stratify=y_trainval, random_state=RANDOM_STATE
    )

    if SCALE_MINMAX:
        scaler = MinMaxScaler()
        scaler.fit(X_train)  # fit chỉ trên train
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    # Class weight
    cw = CLASS_WEIGHT if CLASS_WEIGHT in ("balanced", "balanced_subsample") else None

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        criterion="gini",
        max_depth=MAX_DEPTH,
        max_features=MAX_FEATURES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=cw,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=0,
    )

    rf.fit(X_train, y_train)

    # Validation performance (useful for early manual tuning)
    y_val_pred = rf.predict(X_val)
    val_f1_macro = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
    print(f"[RF] Macro F1 (val): {val_f1_macro:.4f}")

    # Test predictions
    y_test_pred = rf.predict(X_test)
    if n_classes == 2:
        y_test_prob = rf.predict_proba(X_test)[:, 1]
    else:
        y_test_prob = rf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_test_pred, average="macro", zero_division=0)
    f1m = f1_score(y_test, y_test_pred, average="macro", zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 (macro): {f1m:.4f}")

    target_names = [str(c) for c in le.classes_]
    report_dict = classification_report(y_test, y_test_pred, target_names=target_names, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_test_pred, target_names=target_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nEvaluation Report:\n" + report_str)

    # Save metrics
    metrics = {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1m,
        "n_classes": n_classes,
        "classes": target_names,
        "label_col": label_col,
        "n_features": X.shape[1],
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "class_weight": cw,
        "val_macro_f1": val_f1_macro,
    }
    Path(METRICS_FILE).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    pd.DataFrame(report_dict).to_csv(REPORT_CSV, index=True)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(CM_CSV)
    Path(REPORT_TXT).write_text("Evaluation Report:\n" + report_str + "\n", encoding="utf-8")

    # Feature importances
    importances = rf.feature_importances_.astype(float)
    imp_df = pd.DataFrame({"feature": X_df.columns, "importance": importances}).sort_values("importance", ascending=False)
    imp_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    if TOP_N_IMPORTANCE_PLOT and TOP_N_IMPORTANCE_PLOT > 0:
        top_plot = imp_df.head(TOP_N_IMPORTANCE_PLOT).iloc[::-1]  # reverse for horizontal bar
        plt.figure(figsize=(8, max(4, 0.25 * len(top_plot))))
        plt.barh(top_plot["feature"], top_plot["importance"], color="teal")
        plt.xlabel("Gini Importance")
        plt.title(f"Top {len(top_plot)} Feature Importances (RF)")
        plt.tight_layout()
        plt.savefig("rf_feature_importances_top.png", dpi=180)
        plt.close()

    # Confusion matrix heatmap
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    w = max(8, int(0.9 * len(target_names)) + 6)
    h = max(6, int(0.9 * len(target_names)) + 4)
    plt.figure(figsize=(w, h))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="magma", cbar=True)
    plt.title("Confusion Matrix (RF)")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=200)
    plt.close()

    # Probabilities (optional)
    if SAVE_PROBAS:
        if n_classes == 2:
            prob_df = pd.DataFrame({"prob_positive": y_test_prob})
        else:
            prob_df = pd.DataFrame(y_test_prob, columns=[f"prob_{c}" for c in target_names])
        prob_df.to_csv(PROBAS_CSV, index=False)

    # Save model
    if joblib is not None:
        try:
            import joblib as jb
            jb.dump({
                "model": rf,
                "label_encoder": le,
                "scaler": scaler,
                "config": {k: v for k, v in globals().items() if k.isupper() and k not in ("__name__",)}
            }, MODEL_PATH)
        except Exception as e:
            print(f"[RF] Không thể lưu model bằng joblib: {e}")

    print("[RF] Hoàn tất. Model + báo cáo đã lưu.")


if __name__ == "__main__":
    main()
