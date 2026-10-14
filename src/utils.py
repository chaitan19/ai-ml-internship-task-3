"""Data access, schema checks, and split reproducibility for Task 3."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from src.hr_validation import TARGET, validate_hr_schema

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "employee_attrition_clean.csv"
MANIFEST_PATH = ROOT / "data" / "processed" / "train_test_split_manifest.json"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor_task2.pkl"

def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Employee dataset is missing: {path}. See README dataset recovery section.")
    data = pd.read_csv(path)
    validate_hr_schema(data)
    if data["Attrition"].isna().any():
        raise ValueError("Target Attrition contains missing labels.")
    return data

def load_task2_split(data: pd.DataFrame, manifest_path: Path = MANIFEST_PATH):
    """Recover Task 2's fixed stratified index split and verify it against source data."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Task 2 split manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_idx, test_idx = manifest["train_indices"], manifest["test_indices"]
    if len(train_idx) != manifest["train_rows"] or len(test_idx) != manifest["test_rows"]:
        raise ValueError("Split manifest row counts do not match the stored indices.")
    if set(train_idx).intersection(test_idx) or set(train_idx + test_idx) != set(range(len(data))):
        raise ValueError("Split indices overlap or do not cover this dataset exactly.")
    train, test = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()
    expected = manifest["train_target_counts"]
    actual = {str(k): int(v) for k, v in train["Attrition"].value_counts().items()}
    if expected != actual:
        raise ValueError("Task 2 split manifest target counts disagree with dataset row order.")
    return train, test, manifest

def load_preprocessor_blueprint(path: Path = PREPROCESSOR_PATH) -> Pipeline:
    """Load Task 2's fitted transform and return a cloneable, unfitted blueprint."""
    import joblib
    if not path.is_file():
        raise FileNotFoundError(f"Task 2 preprocessor artifact not found: {path}")
    pipeline = joblib.load(path)
    if not isinstance(pipeline, Pipeline) or "feature_engineering" not in pipeline.named_steps:
        raise TypeError("Task 2 preprocessor is incompatible with this project.")
    return clone(pipeline)
