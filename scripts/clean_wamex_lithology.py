from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}

COLUMN_ALIASES: dict[str, list[str]] = {
    "hole_id": ["hole_id", "drillhole", "bhid", "holeid", "hole", "hole_no", "borehole_id"],
    "depth_from": ["from", "depth_from", "from_m", "from_depth", "interval_from"],
    "depth_to": ["to", "depth_to", "to_m", "to_depth", "interval_to"],
    "lithology": ["lithology", "rock_type", "description", "lith", "lith_desc", "geology"],
    "rqd": ["rqd", "rock_quality_designation"],
    "recovery": ["recovery", "core_recovery", "recov"],
    "weathering": ["weathering", "weather", "weathering_grade"],
    "strength": ["strength", "rock_strength", "strength_class"],
    "easting": ["easting", "east", "mga_e", "x"],
    "northing": ["northing", "north", "mga_n", "y"],
    "rl": ["rl", "elevation", "reduced_level", "z"],
    "fault_distance": ["fault_distance", "distance_to_fault", "structure_distance", "distance_to_structure"],
    "target_label": ["target", "label", "class", "hazard_label"],
}

TEXT_PATTERNS = {
    "felsic": ["granite", "granodiorite", "felsic", "quartz", "porphyry"],
    "mafic": ["basalt", "dolerite", "gabbro", "mafic", "diorite"],
    "ultramafic": ["ultramafic", "komatiite", "serpentinite", "peridotite"],
    "sedimentary": ["sandstone", "siltstone", "shale", "sediment", "conglomerate"],
    "shear_zone": ["shear", "mylonite", "foliation", "fault gouge"],
}

STIFFNESS_MAP = {
    "felsic": "stiff",
    "mafic": "stiff",
    "ultramafic": "variable",
    "sedimentary": "moderate",
    "shear_zone": "weak",
    "unknown": "unknown",
}

AS1726_RQD_BINS = [
    (0, 25, "very_poor"),
    (25, 50, "poor"),
    (50, 75, "fair"),
    (75, 90, "good"),
    (90, 101, "excellent"),
]

PROXY_LABEL_RULES = {
    "Seismic Instability": lambda df: (df["structural_risk_high"] == 1) & (df["rqd_numeric"].fillna(100) < 35),
    "Rockfall/Overbreak": lambda df: (df["structural_risk_high"] == 1) & (df["rqd_numeric"].fillna(100) < 50),
    "Normal": lambda df: pd.Series(True, index=df.index),
}


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _scan_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
        return [input_path]

    files = [path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No CSV/XLS/XLSX files found in {input_path}")
    return sorted(files)


def _read_any_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _auto_map_columns(df: pd.DataFrame) -> dict[str, str]:
    normalised = {_normalise_name(col): col for col in df.columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalised:
                mapping[canonical] = normalised[alias]
                break
    return mapping


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _infer_lithology_group(text: str) -> str:
    for label, patterns in TEXT_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            return label
    return "unknown"


def _infer_stiffness_class(group: str, strength: str) -> str:
    if "very strong" in strength or "extremely strong" in strength:
        return "stiff"
    if "weak" in strength:
        return "weak"
    return STIFFNESS_MAP.get(group, "unknown")


def _rqd_category(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    for low, high, label in AS1726_RQD_BINS:
        if low <= value < high:
            return label
    return "unknown"


def _safe_numeric(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    cleaned = (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9\.\-]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _prepare_base_table(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    col_map = _auto_map_columns(df)
    out = pd.DataFrame(index=df.index)
    out["source_file"] = source_name

    for canonical in COLUMN_ALIASES:
        if canonical in col_map:
            out[canonical] = df[col_map[canonical]]

    out["hole_id"] = out.get("hole_id", pd.Series(index=df.index, dtype="object")).fillna("UNKNOWN_HOLE")
    out["depth_from"] = _safe_numeric(out.get("depth_from", pd.Series(index=df.index, dtype="object")))
    out["depth_to"] = _safe_numeric(out.get("depth_to", pd.Series(index=df.index, dtype="object")))
    out["rqd_numeric"] = _safe_numeric(out.get("rqd", pd.Series(index=df.index, dtype="object")))
    out["recovery_numeric"] = _safe_numeric(out.get("recovery", pd.Series(index=df.index, dtype="object")))
    out["easting"] = _safe_numeric(out.get("easting", pd.Series(index=df.index, dtype="object")))
    out["northing"] = _safe_numeric(out.get("northing", pd.Series(index=df.index, dtype="object")))
    out["rl"] = _safe_numeric(out.get("rl", pd.Series(index=df.index, dtype="object")))
    out["fault_distance"] = _safe_numeric(out.get("fault_distance", pd.Series(index=df.index, dtype="object")))

    out["lithology_text_raw"] = out.get("lithology", pd.Series(index=df.index, dtype="object")).map(_clean_text)
    out["weathering_text"] = out.get("weathering", pd.Series(index=df.index, dtype="object")).map(_clean_text)
    out["strength_text"] = out.get("strength", pd.Series(index=df.index, dtype="object")).map(_clean_text)

    out["interval_thickness_m"] = (out["depth_to"] - out["depth_from"]).clip(lower=0)
    out["midpoint_depth_m"] = out["depth_from"] + out["interval_thickness_m"] / 2.0
    out["lithology_group"] = out["lithology_text_raw"].map(_infer_lithology_group)
    out["stiffness_class"] = [
        _infer_stiffness_class(group, strength)
        for group, strength in zip(out["lithology_group"], out["strength_text"], strict=False)
    ]
    out["rqd_category"] = out["rqd_numeric"].map(_rqd_category)
    out["structural_risk_high"] = ((out["fault_distance"].fillna(999999) < 50).astype(int))

    for column in [
        "depth_from",
        "depth_to",
        "interval_thickness_m",
        "midpoint_depth_m",
        "rqd_numeric",
        "recovery_numeric",
        "easting",
        "northing",
        "rl",
        "fault_distance",
    ]:
        out[f"{column}_missing"] = out[column].isna().astype(int)

    return out


def _generate_proxy_labels(df: pd.DataFrame) -> pd.DataFrame:
    has_true_target = "target_label" in df.columns and df["target_label"].notna().any()
    if has_true_target:
        df["target_label"] = df["target_label"].astype(str)
        df["target_label_source"] = "observed"
        return df

    df["target_label"] = "Normal"
    for label, rule in PROXY_LABEL_RULES.items():
        mask = rule(df)
        df.loc[mask, "target_label"] = label
    df["target_label_source"] = "synthetic_proxy"
    return df


def _build_preprocessor(df: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    ignore_cols = {"target_label", "target_label_source", "source_file", "lithology_text_raw"}
    feature_df = df.drop(columns=[col for col in ignore_cols if col in df.columns])
    feature_df = feature_df.dropna(axis=1, how="all")

    numeric_cols = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in feature_df.columns if col not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("ordinal_target", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    return pipeline, numeric_cols, categorical_cols


def _feature_dictionary(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], synthetic_target: bool) -> dict[str, Any]:
    dictionary = {
        "numeric_features": {},
        "categorical_features": {},
        "target": {
            "name": "target_label",
            "synthetic_proxy_used": synthetic_target,
            "classes": sorted(df["target_label"].astype(str).unique().tolist()),
        },
    }
    for col in numeric_cols:
        dictionary["numeric_features"][col] = {
            "dtype": str(df[col].dtype),
            "description": "Numeric feature for XGBoost training",
        }
    for col in categorical_cols:
        dictionary["categorical_features"][col] = {
            "dtype": str(df[col].dtype),
            "description": "Categorical feature cleaned and encoded in preprocessing pipeline",
            "sample_values": sorted(df[col].astype(str).dropna().unique().tolist())[:10],
        }
    return dictionary


def _write_report(
    output_dir: Path,
    files_used: list[Path],
    raw_row_count: int,
    clean_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> None:
    report = f"""# WAMEX Lithology Preprocessing Report

## Input summary
- files processed: {len(files_used)}
- raw rows loaded: {raw_row_count}
- cleaned rows exported: {len(clean_df)}

## Feature summary
- numeric features: {len(numeric_cols)}
- categorical features: {len(categorical_cols)}
- target source values: {", ".join(sorted(clean_df['target_label_source'].astype(str).unique()))}

## Notes
- Lithology text was standardized before grouping into broad geotechnical rock classes.
- RQD was encoded into AS 1726-style quality bins.
- Missing values were imputed in the saved preprocessing pipeline and tracked with indicator columns.
- Structural risk is set to `High` when fault or structure distance is available and less than 50 m.
- Proxy labels are clearly marked as `synthetic_proxy` when observed labels are not present.
- Output is suitable for tabular ML prototyping and XGBoost training, not mine-grade validation.
"""
    (output_dir / "preprocessing_report.md").write_text(report, encoding="utf-8")


def _optional_onnx_export(preprocessor: ColumnTransformer, sample_features: pd.DataFrame, output_dir: Path) -> str | None:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    except Exception:
        return None

    initial_types = []
    for col in sample_features.columns:
        if pd.api.types.is_numeric_dtype(sample_features[col]):
            initial_types.append((col, FloatTensorType([None, 1])))
        else:
            initial_types.append((col, StringTensorType([None, 1])))

    onnx_model = convert_sklearn(preprocessor, initial_types=initial_types)
    target_path = output_dir / "preprocess_pipeline.onnx"
    target_path.write_bytes(onnx_model.SerializeToString())
    return str(target_path)


def process_wamex_lithology(input_path: Path, output_dir: Path, export_onnx: bool = False) -> dict[str, Any]:
    files = _scan_input_files(input_path)
    frames = []
    raw_row_count = 0

    for file_path in files:
        try:
            frame = _read_any_table(file_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read {file_path}: {exc}") from exc
        raw_row_count += len(frame)
        frames.append(_prepare_base_table(frame, file_path.name))

    clean_df = pd.concat(frames, ignore_index=True)
    clean_df = _generate_proxy_labels(clean_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_dir / "cleaned_lithology_features.csv", index=False)

    pipeline, numeric_cols, categorical_cols = _build_preprocessor(clean_df)
    feature_df = clean_df.drop(columns=["target_label", "target_label_source", "source_file"], errors="ignore")
    feature_df = feature_df.dropna(axis=1, how="all")

    transformed = pipeline.named_steps["preprocess"].fit_transform(feature_df)
    transformed_columns = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    xgb_table = pd.DataFrame(transformed, columns=transformed_columns)
    xgb_table["target_label"] = clean_df["target_label"].astype(str).values
    xgb_table["target_label_source"] = clean_df["target_label_source"].astype(str).values
    xgb_table.to_csv(output_dir / "xgboost_training_table.csv", index=False)

    feature_dict = _feature_dictionary(
        clean_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        synthetic_target=(clean_df["target_label_source"] == "synthetic_proxy").any(),
    )
    (output_dir / "feature_dictionary.json").write_text(json.dumps(feature_dict, indent=2), encoding="utf-8")
    _write_report(output_dir, files, raw_row_count, clean_df, numeric_cols, categorical_cols)

    joblib.dump(pipeline.named_steps["preprocess"], output_dir / "preprocess_pipeline.joblib")

    onnx_path = None
    if export_onnx:
        onnx_path = _optional_onnx_export(pipeline.named_steps["preprocess"], feature_df.head(1), output_dir)

    return {
        "files_processed": len(files),
        "raw_rows": raw_row_count,
        "clean_rows": len(clean_df),
        "onnx_exported": onnx_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert messy WAMEX lithology CSV/XLSX files into XGBoost-ready numeric training tables."
    )
    parser.add_argument("--input", required=True, help="Input file or directory containing CSV/XLS/XLSX files.")
    parser.add_argument("--output", required=True, help="Output directory for processed artifacts.")
    parser.add_argument("--export-onnx", action="store_true", help="Try exporting the preprocessing graph to ONNX.")
    args = parser.parse_args()

    result = process_wamex_lithology(Path(args.input), Path(args.output), export_onnx=args.export_onnx)
    print(json.dumps(result, indent=2))
    print("\nUsage example:\npython clean_wamex_lithology.py --input data/raw_wamex --output data/processed")


if __name__ == "__main__":
    main()
