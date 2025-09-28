import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_columns(csv_path: Path) -> List[str]:
	return list(pd.read_csv(csv_path, nrows=0).columns)


def get_feature_columns(csv_path: Path, label_col: str, exclude_cols: List[str]) -> List[str]:
	cols = get_columns(csv_path)
	features = [c for c in cols if c not in set([label_col] + exclude_cols)]
	if not features:
		raise ValueError("Không tìm thấy cột feature nào sau khi loại trừ.")
	return features


def read_training_frame(
	csv_path: Path,
	label_col: str,
	feature_cols: List[str],
	n_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.Series]:
	"""Read a training frame for RF. Use compact dtypes to save memory.

	If n_rows is provided, only the first n_rows are loaded to avoid OOM.
	"""
	usecols = [label_col] + feature_cols
	dtype_map = {c: "uint8" for c in feature_cols}
	# Read limited rows if requested
	df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype_map, nrows=n_rows)
	y = df[label_col].astype(str)  # RF handles string labels fine
	X = df.drop(columns=[label_col])
	return X, y


def export_selected_dataset(csv_path: Path, label_col: str, select_cols: List[str], out_path: Path, chunksize: int) -> None:
	header_written = False
	usecols = [label_col] + select_cols
	for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
		mode = "w" if not header_written else "a"
		chunk.to_csv(out_path, mode=mode, index=False, header=not header_written)
		header_written = True


def run_rf_gini(
	csv_path: Path,
	label_col: str,
	exclude_cols: List[str],
	n_rows: Optional[int],
	n_estimators: int,
	max_depth: Optional[int],
	max_features: Optional[str],
	min_samples_leaf: int,
	class_weight: Optional[str],
	random_state: int,
	out_scores: Path,
	select_topk: int,
	out_selected: Optional[Path],
	chunksize: int,
) -> pd.DataFrame:
	feature_cols = get_feature_columns(csv_path, label_col=label_col, exclude_cols=exclude_cols)

	# Load a training slice to fit RF (avoid OOM with n_rows)
	X, y = read_training_frame(csv_path, label_col=label_col, feature_cols=feature_cols, n_rows=n_rows)

	rf = RandomForestClassifier(
		n_estimators=n_estimators,
		criterion="gini",
		max_depth=max_depth,
		max_features=max_features,
		min_samples_leaf=min_samples_leaf,
		class_weight=class_weight,
		n_jobs=-1,
		random_state=random_state,
	)

	rf.fit(X, y)
	importances = rf.feature_importances_.astype(float)

	df_scores = pd.DataFrame({"feature": feature_cols, "gini_importance": importances})
	df_scores = df_scores.sort_values("gini_importance", ascending=False)
	df_scores.to_csv(out_scores, index=False)

	if select_topk and select_topk > 0 and out_selected is not None:
		top_cols = df_scores.head(select_topk)["feature"].tolist()
		export_selected_dataset(csv_path, label_col, top_cols, out_selected, chunksize)

	return df_scores


def parse_args() -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Feature selection bằng Random Forest (Gini importance).")
	ap.add_argument("--csv", default="drebin_benign_malware.csv", help="Đường dẫn CSV nguồn.")
	ap.add_argument("--label-col", default="family", help="Tên cột nhãn.")
	ap.add_argument("--exclude-cols", default="apk_name", help="Các cột loại trừ, phân tách bằng dấu phẩy.")
	ap.add_argument("--n-rows", type=int, default=None, help="Chỉ đọc N hàng đầu để huấn luyện RF (tránh OOM).")
	ap.add_argument("--n-estimators", type=int, default=300, help="Số cây RF.")
	ap.add_argument("--max-depth", type=int, default=None, help="Độ sâu tối đa của cây.")
	ap.add_argument("--max-features", default="sqrt", help="Số đặc trưng thử tại mỗi split: sqrt|log2|None hoặc số/chuỗi scikit.")
	ap.add_argument("--min-samples-leaf", type=int, default=1, help="Số mẫu tối thiểu tại lá.")
	ap.add_argument("--class-weight", default="balanced_subsample", help="class_weight cho RF: balanced|balanced_subsample|None")
	ap.add_argument("--random-state", type=int, default=0, help="Seed.")
	ap.add_argument("--out-scores", default="rf_gini_scores.csv", help="CSV kết quả quan trọng Gini.")
	ap.add_argument("--select-topk", type=int, default=0, help="Nếu >0, xuất dataset chỉ gồm top-K feature + nhãn.")
	ap.add_argument("--out-selected", default="rf_gini_selected.csv", help="CSV dữ liệu sau khi chọn top-K feature.")
	ap.add_argument("--chunksize", type=int, default=200_000, help="Chunk size khi export dataset đã chọn.")
	return ap.parse_args()


def main() -> None:
	args = parse_args()
	csv_path = Path(args.csv)
	if not csv_path.exists():
		raise SystemExit(f"Không tìm thấy file: {csv_path}")

	exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
	out_scores = Path(args.out_scores)
	out_selected = Path(args.out_selected) if args.select_topk and args.select_topk > 0 else None

	df = run_rf_gini(
		csv_path=csv_path,
		label_col=args.label_col,
		exclude_cols=exclude_cols,
		n_rows=args.n_rows,
		n_estimators=args.n_estimators,
		max_depth=args.max_depth,
		max_features=(None if args.max_features.lower() == "none" else args.max_features),
		min_samples_leaf=args.min_samples_leaf,
		class_weight=(None if args.class_weight.lower() == "none" else args.class_weight),
		random_state=args.random_state,
		out_scores=out_scores,
		select_topk=args.select_topk,
		out_selected=out_selected,
		chunksize=args.chunksize,
	)

	print(f"Đã tính RF Gini cho {len(df)} features. Lưu vào: {out_scores.resolve()}")
	if out_selected is not None:
		print(f"Đã xuất dataset với top-{args.select_topk} features + nhãn: {out_selected.resolve()}")


if __name__ == "__main__":
	main()

