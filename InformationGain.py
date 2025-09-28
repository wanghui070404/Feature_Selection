import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


def get_columns(csv_path: Path) -> List[str]:
	"""Read only the header to get all columns."""
	cols = list(pd.read_csv(csv_path, nrows=0).columns)
	if not cols:
		raise ValueError("CSV không có cột nào.")
	return cols


def get_feature_columns(
	csv_path: Path,
	label_col: str,
	exclude_cols: List[str],
) -> List[str]:
	cols = get_columns(csv_path)
	features = [c for c in cols if c not in set([label_col] + exclude_cols)]
	if not features:
		raise ValueError("Không tìm thấy cột feature nào sau khi loại trừ.")
	return features


def pass1_label_counts(
	csv_path: Path, label_col: str, chunksize: int
) -> Tuple[Dict[str, int], int]:
	"""Count label frequency in a streaming manner.

	Returns: (label_counts, total_rows)
	"""
	counts: Dict[str, int] = {}
	total = 0
	for chunk in pd.read_csv(csv_path, usecols=[label_col], chunksize=chunksize):
		vals = chunk[label_col].astype(str)
		vc = vals.value_counts()
		for k, v in vc.items():
			counts[k] = counts.get(k, 0) + int(v)
		total += int(len(vals))
	if total == 0:
		raise ValueError("CSV rỗng, không có bản ghi nào.")
	return counts, total


def pass2_feature_class_counts(
	csv_path: Path,
	label_col: str,
	feature_cols: List[str],
	class_to_idx: Dict[str, int],
	chunksize: int,
) -> np.ndarray:
	"""Accumulate counts of X=1 by class for each feature.

	Returns: counts1 of shape (n_features, n_classes)
	"""
	n_features = len(feature_cols)
	n_classes = len(class_to_idx)
	counts1 = np.zeros((n_features, n_classes), dtype=np.int64)

	# Build dtype mapping to reduce memory
	dtype_map = {c: "uint8" for c in feature_cols}
	dtype_map[label_col] = "category"
	usecols = [label_col] + feature_cols

	for chunk in pd.read_csv(csv_path, usecols=usecols, dtype=dtype_map, chunksize=chunksize):
		# Ensure label to string keys matching mapping
		chunk[label_col] = chunk[label_col].astype(str)

		# Group by label and sum features (fast and memory-friendly)
		grp = chunk.groupby(label_col)[feature_cols].sum()
		if grp.empty:
			continue
		# Align each group's row to counts1 by class index
		for label_value, row in grp.iterrows():
			cls_idx = class_to_idx.get(str(label_value))
			if cls_idx is None:
				continue
			counts1[:, cls_idx] += row.to_numpy(dtype=np.int64)

	return counts1


def entropy_from_counts(counts: np.ndarray) -> float:
	"""Compute entropy H from class counts."""
	total = counts.sum()
	if total == 0:
		return 0.0
	p = counts / total
	# Avoid log2(0) by masking zeros
	p = p[p > 0]
	if p.size == 0:
		return 0.0
	return float(-(p * np.log2(p)).sum())


def information_gain_per_feature(
	class_counts: np.ndarray,  # shape (n_classes,)
	counts1: np.ndarray,  # shape (n_features, n_classes)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Compute IG for each feature given overall class counts and X=1 counts per class.

	Returns: (ig, n1, n0)
	"""
	n_features, n_classes = counts1.shape
	total = class_counts.sum()
	base_entropy = entropy_from_counts(class_counts)

	# Prepare arrays
	ig = np.zeros(n_features, dtype=np.float64)
	n1 = counts1.sum(axis=1)  # total X=1
	n0 = total - n1

	# Precompute class counts for X=0: class_counts - counts1[j]
	for j in range(n_features):
		c1 = counts1[j]  # counts for X=1 by class
		c0 = class_counts - c1
		h1 = entropy_from_counts(c1)
		h0 = entropy_from_counts(c0)
		p1 = n1[j] / total if total > 0 else 0.0
		p0 = 1.0 - p1
		cond_entropy = p1 * h1 + p0 * h0
		ig[j] = base_entropy - cond_entropy

	return ig, n1, n0


def export_selected_dataset(
	csv_path: Path,
	label_col: str,
	select_cols: List[str],
	out_path: Path,
	chunksize: int,
) -> None:
	"""Stream and write only selected columns + label to a new CSV."""
	header_written = False
	usecols = [label_col] + select_cols
	for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
		mode = "w" if not header_written else "a"
		chunk.to_csv(out_path, mode=mode, index=False, header=not header_written)
		header_written = True


def run_information_gain(
	csv_path: Path,
	label_col: str,
	exclude_cols: List[str],
	chunksize: int,
	out_scores: Path,
	topk: int | None,
	out_selected: Path | None,
) -> pd.DataFrame:
	# Discover features and compute class counts (pass 1)
	feature_cols = get_feature_columns(csv_path, label_col=label_col, exclude_cols=exclude_cols)
	label_counts_dict, total = pass1_label_counts(csv_path, label_col=label_col, chunksize=chunksize)

	labels = sorted(label_counts_dict.keys())
	class_to_idx = {lab: i for i, lab in enumerate(labels)}
	class_counts = np.array([label_counts_dict[lab] for lab in labels], dtype=np.int64)

	# Pass 2: counts of X=1 by class per feature
	counts1 = pass2_feature_class_counts(
		csv_path=csv_path,
		label_col=label_col,
		feature_cols=feature_cols,
		class_to_idx=class_to_idx,
		chunksize=chunksize,
	)

	# Compute IG
	ig, n1, n0 = information_gain_per_feature(class_counts, counts1)

	# Build and save result
	df = pd.DataFrame(
		{
			"feature": feature_cols,
			"information_gain": ig,
			"count_ones": n1,
			"count_zeros": n0,
		}
	).sort_values("information_gain", ascending=False)

	df.to_csv(out_scores, index=False)

	# Optional export of selected dataset
	if topk is not None and topk > 0 and out_selected is not None:
		select_cols = df.head(topk)["feature"].tolist()
		export_selected_dataset(
			csv_path=csv_path,
			label_col=label_col,
			select_cols=select_cols,
			out_path=out_selected,
			chunksize=chunksize,
		)

	return df


def parse_args() -> argparse.Namespace:
	ap = argparse.ArgumentParser(
		description="Tính Information Gain cho từng feature (dạng nhị phân) so với cột nhãn, đọc theo chunks để tiết kiệm RAM."
	)
	ap.add_argument("--csv", default="drebin_benign_malware.csv", help="Đường dẫn CSV nguồn.")
	ap.add_argument(
		"--label-col",
		default="family",
		help="Tên cột nhãn (ví dụ: family hoặc label).",
	)
	ap.add_argument(
		"--exclude-cols",
		default="apk_name",
		help="Các cột loại trừ, phân tách bằng dấu phẩy (mặc định: apk_name).",
	)
	ap.add_argument("--chunksize", type=int, default=200_000, help="Kích thước chunk khi đọc CSV.")
	ap.add_argument("--out-scores", default="ig_scores.csv", help="CSV kết quả IG cho từng feature.")
	ap.add_argument(
		"--select-topk",
		type=int,
		default=0,
		help="Nếu >0, sẽ xuất dataset chỉ gồm top-K feature và cột nhãn.",
	)
	ap.add_argument(
		"--out-selected",
		default="ig_selected.csv",
		help="CSV dữ liệu sau khi chọn top-K feature (khi --select-topk>0).",
	)
	return ap.parse_args()


def main() -> None:
	args = parse_args()
	csv_path = Path(args.csv)
	if not csv_path.exists():
		raise SystemExit(f"Không tìm thấy file: {csv_path}")

	exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
	out_scores = Path(args.out_scores)
	out_selected = Path(args.out_selected) if args.select_topk and args.select_topk > 0 else None

	df = run_information_gain(
		csv_path=csv_path,
		label_col=args.label_col,
		exclude_cols=exclude_cols,
		chunksize=args.chunksize,
		out_scores=out_scores,
		topk=args.select_topk,
		out_selected=out_selected,
	)

	# Print a short summary to console
	print(f"Đã tính IG cho {len(df)} features. Lưu vào: {out_scores.resolve()}")
	if out_selected is not None:
		print(
			f"Đã xuất dataset với top-{args.select_topk} features + nhãn: {out_selected.resolve()}"
		)


if __name__ == "__main__":
	main()

