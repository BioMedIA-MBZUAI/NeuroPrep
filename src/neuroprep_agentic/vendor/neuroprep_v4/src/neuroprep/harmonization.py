from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HarmonizationSpec:
    participant_column: str
    split_column: str
    batch_column: str
    protected_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    train_label: str = "train"
    allow_repeated_participants: bool = False


def harmonize_table(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    report_path: Path,
    spec: HarmonizationSpec,
) -> dict[str, Any]:
    try:
        import pandas as pd
        from neuroHarmonize import harmonizationApply, harmonizationLearn
    except ImportError as exc:
        raise RuntimeError(
            "Install NeuroPrep's harmonization extra: pip install 'neuroprep[harmonization]'"
        ) from exc

    frame = _read_table(input_path, pd)
    details = validate_harmonization_frame(frame, spec)
    train_mask = frame[spec.split_column].astype(str) == spec.train_label
    train = frame.loc[train_mask]

    batch_labels = sorted(train[spec.batch_column].astype(str).unique())
    batch_codes = {label: index for index, label in enumerate(batch_labels)}
    all_batches = set(frame[spec.batch_column].astype(str))
    unseen = sorted(all_batches - set(batch_codes))
    if unseen:
        raise ValueError(
            "Validation/test data contain batches absent from training: " + ", ".join(unseen)
        )

    categorical_levels = _categorical_levels(train, spec, pd)
    for column, levels in categorical_levels.items():
        unseen_levels = sorted(set(frame[column].astype(str)) - set(levels))
        if unseen_levels:
            raise ValueError(
                f"Validation/test {column} contains categories absent from training: "
                + ", ".join(unseen_levels)
            )
    covariates = _covariates(frame, spec, batch_codes, pd, categorical_levels)
    train_covariates = covariates.loc[train_mask].reset_index(drop=True)
    data = frame.loc[:, list(spec.feature_columns)].astype(float).to_numpy()
    train_data = data[train_mask.to_numpy()]
    model, _ = harmonizationLearn(train_data, train_covariates)
    adjusted = np.asarray(harmonizationApply(data, covariates, model))
    if adjusted.shape != data.shape:
        raise RuntimeError(
            f"Harmonizer returned shape {adjusted.shape}; expected {data.shape}"
        )

    result = frame.copy()
    result.loc[:, list(spec.feature_columns)] = adjusted
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    separator = "\t" if output_path.suffix.lower() == ".tsv" else ","
    result.to_csv(temporary_output, sep=separator, index=False)
    temporary_output.replace(output_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(model_path.suffix + ".tmp")
    with temporary_model.open("wb") as stream:
        pickle.dump(
            {
                "model": model,
                "batch_codes": batch_codes,
                "spec": spec,
                "feature_columns": spec.feature_columns,
                "categorical_levels": categorical_levels,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary_model.replace(model_path)

    package_version = _package_version("neuroHarmonize")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "neuroHarmonize-ComBat",
        "software_version": package_version,
        "input": str(input_path),
        "output": str(output_path),
        "model": str(model_path),
        "participant_column": spec.participant_column,
        "split_column": spec.split_column,
        "train_label": spec.train_label,
        "batch_column": spec.batch_column,
        "protected_columns": list(spec.protected_columns),
        "feature_columns": list(spec.feature_columns),
        "training_participants": train[spec.participant_column].astype(str).tolist(),
        "batch_codes": batch_codes,
        "categorical_levels": categorical_levels,
        "validation": details,
        "diagnostics": _batch_diagnostics(frame, result, spec),
        "leakage_policy": "fit-on-training-only; apply-frozen-model-to-all-splits",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_report.replace(report_path)
    return report


def harmonize_folds(
    input_path: Path,
    output_path: Path,
    output_root: Path,
    report_path: Path,
    spec: HarmonizationSpec,
    fold_column: str = "fold",
) -> dict[str, Any]:
    """Fit an independent harmonizer in each fold and apply only its frozen model."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Install NeuroPrep's harmonization extra: pip install 'neuroprep[harmonization]'"
        ) from exc
    frame = _read_table(input_path, pd)
    if fold_column not in frame.columns:
        raise ValueError(f"Fold-aware harmonization requires column: {fold_column}")
    adjusted_frames = []
    fold_reports = []
    fold_values = sorted(frame[fold_column].dropna().unique().tolist(), key=str)
    if not fold_values:
        raise ValueError("No fold assignments found")
    for fold in fold_values:
        subset = frame.loc[frame[fold_column] == fold].copy()
        fold_root = output_root / f"fold-{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        fold_input = fold_root / "input.tsv"
        fold_output = fold_root / "harmonized.tsv"
        fold_model = fold_root / "harmonizer.pkl"
        fold_report = fold_root / "harmonization-report.json"
        subset.to_csv(fold_input, sep="\t", index=False)
        report = harmonize_table(
            input_path=fold_input,
            output_path=fold_output,
            model_path=fold_model,
            report_path=fold_report,
            spec=spec,
        )
        adjusted = pd.read_csv(fold_output, sep="\t")
        adjusted_frames.append(adjusted)
        fold_reports.append(
            {
                "fold": fold,
                "model": str(fold_model),
                "output": str(fold_output),
                "report": str(fold_report),
                "training_participants": report["training_participants"],
                "batch_codes": report["batch_codes"],
            }
        )
    combined = pd.concat(adjusted_frames, axis=0, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    separator = "\t" if output_path.suffix.lower() == ".tsv" else ","
    combined.to_csv(temporary, sep=separator, index=False)
    temporary.replace(output_path)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "fold-aware-neuroHarmonize-ComBat",
        "fold_column": fold_column,
        "split_column": spec.split_column,
        "training_label": spec.train_label,
        "folds": fold_reports,
        "output": str(output_path),
        "leakage_policy": (
            "one independently fitted model per fold; fit on train rows only; "
            "the frozen fold model is applied unchanged to validation and test rows"
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_report.replace(report_path)
    return report


def validate_harmonization_frame(frame: Any, spec: HarmonizationSpec) -> dict[str, Any]:
    required = {
        spec.participant_column,
        spec.split_column,
        spec.batch_column,
        *spec.protected_columns,
        *spec.feature_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Harmonization table is missing columns: {missing}")
    if not spec.feature_columns:
        raise ValueError("At least one feature column is required")
    if frame[spec.participant_column].isna().any():
        raise ValueError("Participant identifiers cannot be missing")
    duplicated = frame[spec.participant_column].astype(str).duplicated()
    if duplicated.any() and not spec.allow_repeated_participants:
        examples = frame.loc[duplicated, spec.participant_column].astype(str).head(5).tolist()
        raise ValueError(f"Participants occur more than once: {examples}")
    if duplicated.any():
        split_counts = frame.groupby(spec.participant_column)[spec.split_column].nunique()
        if int(split_counts.max()) > 1:
            offenders = split_counts[split_counts > 1].index.astype(str).tolist()[:5]
            raise ValueError(f"Participant leakage across splits: {offenders}")
        batch_counts = frame.groupby(spec.participant_column)[spec.batch_column].nunique()
        if int(batch_counts.max()) > 1:
            offenders = batch_counts[batch_counts > 1].index.astype(str).tolist()[:5]
            raise ValueError(f"Participants map to multiple batches/sites: {offenders}")
    if frame[spec.split_column].isna().any():
        raise ValueError("Split assignments cannot be missing")
    train_mask = frame[spec.split_column].astype(str) == spec.train_label
    if int(train_mask.sum()) < 2:
        raise ValueError("Training split must contain at least two participants")
    if frame.loc[:, list(spec.feature_columns)].isna().any().any():
        raise ValueError("Feature columns contain missing values")
    numeric = frame.loc[:, list(spec.feature_columns)].apply(
        lambda column: np.isfinite(column.astype(float).to_numpy()).all()
    )
    if not bool(numeric.all()):
        raise ValueError("Feature columns must contain only finite numeric values")
    train_batches = frame.loc[train_mask, spec.batch_column].astype(str).nunique()
    if train_batches < 2:
        raise ValueError("ComBat training data require at least two batches")
    return {
        "rows": int(len(frame)),
        "training_rows": int(train_mask.sum()),
        "splits": sorted(frame[spec.split_column].astype(str).unique().tolist()),
        "training_batches": int(train_batches),
    }


def _read_table(path: Path, pd: Any) -> Any:
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=separator)


def _covariates(
    frame: Any,
    spec: HarmonizationSpec,
    batch_codes: dict[str, int],
    pd: Any,
    categorical_levels: dict[str, list[str]],
) -> Any:
    covariates = pd.DataFrame(
        {"SITE": frame[spec.batch_column].astype(str).map(batch_codes).astype(int)}
    )
    numeric_columns = [
        column
        for column in spec.protected_columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_columns = [
        column for column in spec.protected_columns if column not in numeric_columns
    ]
    if numeric_columns:
        numeric = frame.loc[:, numeric_columns].astype(float).reset_index(drop=True)
        covariates = pd.concat([covariates, numeric], axis=1)
    if categorical_columns:
        encoded = {
            column: pd.Categorical(
                frame[column].astype(str),
                categories=categorical_levels[column],
            )
            for column in categorical_columns
        }
        categorical = pd.get_dummies(pd.DataFrame(encoded), drop_first=True, dtype=float)
        covariates = pd.concat([covariates, categorical.reset_index(drop=True)], axis=1)
    return covariates


def _categorical_levels(train: Any, spec: HarmonizationSpec, pd: Any) -> dict[str, list[str]]:
    return {
        column: sorted(train[column].astype(str).unique().tolist())
        for column in spec.protected_columns
        if not pd.api.types.is_numeric_dtype(train[column])
    }


def _batch_diagnostics(before: Any, after: Any, spec: HarmonizationSpec) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for name, frame in (("before", before), ("after", after)):
        means = frame.groupby(spec.batch_column)[list(spec.feature_columns)].mean()
        diagnostics[name] = {
            str(batch): {feature: float(value) for feature, value in row.items()}
            for batch, row in means.iterrows()
        }
    return diagnostics


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"
