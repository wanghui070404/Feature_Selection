import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def get_columns(csv_path: Path) -> List[str]:
	return list(pd.read_csv(csv_path, nrows=0).columns)


def get_feature_columns(csv_path: Path, label_col: str, exclude_cols: List[str]) -> List[str]:
	cols = get_columns(csv_path)
	features = [c for c in cols if c not in set([label_col] + exclude_cols)]
	if not features:
		raise ValueError("Không tìm thấy cột feature nào sau khi loại trừ.")
	return features


def pass1_label_counts(csv_path: Path, label_col: str, chunksize: int) -> Tuple[Dict[str, int], int]:
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
	n_features = len(feature_cols)
	n_classes = len(class_to_idx)
	counts1 = np.zeros((n_features, n_classes), dtype=np.int64)

	dtype_map = {c: "uint8" for c in feature_cols}
	dtype_map[label_col] = "category"
	usecols = [label_col] + feature_cols

	for chunk in pd.read_csv(csv_path, usecols=usecols, dtype=dtype_map, chunksize=chunksize):
		chunk[label_col] = chunk[label_col].astype(str)
		grp = chunk.groupby(label_col)[feature_cols].sum()
		if grp.empty:
			continue
		for label_value, row in grp.iterrows():
			cls_idx = class_to_idx.get(str(label_value))
			if cls_idx is None:
				continue
			counts1[:, cls_idx] += row.to_numpy(dtype=np.int64)

	return counts1


def chi_square_from_counts(class_counts: np.ndarray, counts1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Compute Chi-square statistic per feature for multi-class label.

	For each feature j, build a 2 x C contingency table:
	- Row 0: X=1 counts per class -> counts1[j]
	- Row 1: X=0 counts per class -> class_counts - counts1[j]

	chi2 = sum_{rows,classes} (observed - expected)^2 / expected

	Returns: (chi2, n1, n0)
	"""
	n_features, n_classes = counts1.shape
	total = class_counts.sum()

	n1 = counts1.sum(axis=1)  # totals for X=1
	n0 = total - n1           # totals for X=0

	chi2 = np.zeros(n_features, dtype=np.float64)

	# Expected counts for each feature j are derived from row totals and column totals
	col_totals = class_counts.astype(np.float64)
	for j in range(n_features):
		o1 = counts1[j].astype(np.float64)
		o0 = col_totals - o1

		# Row totals
		r1 = float(n1[j])
		r0 = float(n0[j])

		# Avoid division by zero: if a row total is zero, its contribution is zero
		contrib = 0.0
		if r1 > 0:
			e1 = r1 * col_totals / total
			# mask expected > 0
			m1 = e1 > 0
			diff1 = o1[m1] - e1[m1]
			contrib += np.sum((diff1 * diff1) / e1[m1])
		if r0 > 0:
			e0 = r0 * col_totals / total
			m0 = e0 > 0
			diff0 = o0[m0] - e0[m0]
			contrib += np.sum((diff0 * diff0) / e0[m0])

		chi2[j] = contrib

	return chi2, n1, n0


def export_selected_dataset(csv_path: Path, label_col: str, select_cols: List[str], out_path: Path, chunksize: int) -> None:
	header_written = False
	usecols = [label_col] + select_cols
	for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
		mode = "w" if not header_written else "a"
		chunk.to_csv(out_path, mode=mode, index=False, header=not header_written)
		header_written = True


def run_chi_square(
	csv_path: Path,
	label_col: str,
	exclude_cols: List[str],
	chunksize: int,
	out_scores: Path,
	topk: int | None,
	out_selected: Path | None,
) -> pd.DataFrame:
	feature_cols = get_feature_columns(csv_path, label_col=label_col, exclude_cols=exclude_cols)
	label_counts_dict, total = pass1_label_counts(csv_path, label_col=label_col, chunksize=chunksize)

	labels = sorted(label_counts_dict.keys())
	class_to_idx = {lab: i for i, lab in enumerate(labels)}
	class_counts = np.array([label_counts_dict[lab] for lab in labels], dtype=np.int64)

	counts1 = pass2_feature_class_counts(
		csv_path=csv_path,
		label_col=label_col,
		feature_cols=feature_cols,
		class_to_idx=class_to_idx,
		chunksize=chunksize,
	)

	chi2, n1, n0 = chi_square_from_counts(class_counts, counts1)

	df = pd.DataFrame(
		{
			"feature": feature_cols,
			"chi_square": chi2,
			"count_ones": n1,
			"count_zeros": n0,
		}
	).sort_values("chi_square", ascending=False)

	df.to_csv(out_scores, index=False)

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
	ap = argparse.ArgumentParser(description="Feature selection theo Chi-square (đọc CSV theo chunks để tiết kiệm RAM).")
	ap.add_argument("--csv", default="drebin_benign_malware.csv", help="Đường dẫn CSV nguồn.")
	ap.add_argument("--label-col", default="family", help="Tên cột nhãn (ví dụ: family hoặc label).")
	ap.add_argument("--exclude-cols", default="apk_name", help="Các cột loại trừ, phân tách bằng dấu phẩy.")
	ap.add_argument("--chunksize", type=int, default=200_000, help="Kích thước chunk khi đọc CSV.")
	ap.add_argument("--out-scores", default="chi2_scores.csv", help="CSV kết quả Chi-square cho từng feature.")
	ap.add_argument("--select-topk", type=int, default=0, help="Nếu >0, xuất dataset chỉ gồm top-K feature + nhãn.")
	ap.add_argument("--out-selected", default="chi2_selected.csv", help="CSV dữ liệu sau khi chọn top-K feature.")
	return ap.parse_args()


def main() -> None:
	args = parse_args()
	csv_path = Path(args.csv)
	if not csv_path.exists():
		raise SystemExit(f"Không tìm thấy file: {csv_path}")

	exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
	out_scores = Path(args.out_scores)
	out_selected = Path(args.out_selected) if args.select_topk and args.select_topk > 0 else None

	df = run_chi_square(
		csv_path=csv_path,
		label_col=args.label_col,
		exclude_cols=exclude_cols,
		chunksize=args.chunksize,
		out_scores=out_scores,
		topk=args.select_topk,
		out_selected=out_selected,
	)

	print(f"Đã tính Chi-square cho {len(df)} features. Lưu vào: {out_scores.resolve()}")
	if out_selected is not None:
		print(f"Đã xuất dataset với top-{args.select_topk} features + nhãn: {out_selected.resolve()}")


if __name__ == "__main__":
	main()

