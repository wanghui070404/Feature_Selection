# Feature_Selection
## 1. Information Gain
File: `InformationGain.py`
```cmd
python InformationGain.py --csv drebin_benign_malware.csv --label-col family --exclude-cols apk_name --select-topk 200 --out-selected ig_selected.csv --out-scores ig_scores.csv
```

## 2. Chi-Square
File: `Chi-Square.py`
```cmd
python Chi-Square.py --csv drebin_benign_malware.csv --label-col family --exclude-cols apk_name --select-topk 200 --out-selected chi_selected.csv --out-scores chi_scores.csv
```

## 3. Random Forest Gini Importance
File: `RF Gini.py`
```cmd
python "RF Gini.py" --csv drebin_benign_malware.csv --label-col family --exclude-cols apk_name --n-rows 300000 --select-topk 200 --out-selected rf_gini_selected.csv --out-scores rf_gini_scores.csv
```
