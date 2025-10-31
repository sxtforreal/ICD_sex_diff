#!/usr/bin/env python3
"""Generate a cohort-comparison TableOne for SCMR and an external dataset.

This helper script mirrors the preprocessing conventions from `SCMR.py` and
`evaluate_external.py`, but is streamlined for local interactive use. It loads
the SCMR cohort, reads an external dataset, harmonises overlapping features,
and produces a TableOne (or a fallback summary) comparing the two cohorts on
all shared variables, explicitly including `PE` (primary endpoint) and
`PE_Time`.

Example usage:

    python run_tableone_local.py \
        --external-path /path/to/external.xlsx \
        --external-sheet Sheet1 \
        --output-dir ./results/tableone_local

Outputs are written to the chosen directory as both Excel and CSV files.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_object_dtype,
)

import SCMR as scmr
import evaluate_external as eval_ext


def _progress(message: str) -> None:
    """Lightweight progress printer that always flushes."""

    try:
        print(f"[TableOneLocal] {message}", flush=True)
    except Exception:
        pass


def _infer_categorical(
    columns: Sequence[str],
    ref_df: pd.DataFrame,
    ext_df: pd.DataFrame,
) -> List[str]:
    """Infer which variables should be treated as categorical in TableOne."""

    categorical: List[str] = []
    seen = set()

    def _extend(values: Iterable[str]) -> None:
        for name in values:
            if name in columns and name not in seen:
                categorical.append(name)
                seen.add(name)

    try:
        _extend(scmr.BINARY_FEATURES)
    except Exception:
        pass

    _extend(["ICD", "CrCl>45", "Significant LGE", "NYHA>2", "PE"])

    try:
        _extend(scmr.NOMINAL_MULTICLASS_FEATURES)
    except Exception:
        pass

    try:
        _extend(scmr.ORDINAL_FEATURES)
    except Exception:
        pass

    for col in columns:
        if col in seen:
            continue
        series_candidates = []
        if col in ref_df.columns:
            series_candidates.append(ref_df[col])
        if col in ext_df.columns:
            series_candidates.append(ext_df[col])
        for series in series_candidates:
            dtype = series.dtype
            if is_categorical_dtype(dtype) or is_object_dtype(dtype) or is_bool_dtype(dtype):
                categorical.append(col)
                seen.add(col)
                break

    return categorical


def _ensure_table_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy of df restricted to columns, adding missing as NaN."""

    out = pd.DataFrame({})
    for col in columns:
        if col in df.columns:
            out[col] = df[col]
        else:
            out[col] = np.nan
    return out


def _build_fallback_summary(
    ref_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    variables: Sequence[str],
    categorical_cols: Sequence[str],
) -> pd.DataFrame:
    """Fallback summary when TableOne is unavailable."""

    rows: List[dict] = []
    categorical_set = set(categorical_cols)

    for var in variables:
        s_ref = ref_df[var] if var in ref_df.columns else pd.Series(dtype=float)
        s_ext = ext_df[var] if var in ext_df.columns else pd.Series(dtype=float)

        if var in categorical_set:
            combined = pd.concat(
                [s_ref.astype("object"), s_ext.astype("object")],
                axis=0,
                ignore_index=True,
            )
            levels = [lvl for lvl in pd.unique(combined.dropna())]
            if not levels:
                row = {"variable": var, "level": "Missing"}
                for label, series in (("SCMR", s_ref), ("External", s_ext)):
                    total = int(len(series))
                    missing = int(series.isna().sum())
                    row[label] = f"{missing} ({(missing / total * 100.0) if total else 0.0:.1f}%)" if total else ""
                rows.append(row)
                continue
            for level in levels:
                level_str = str(level)
                row = {"variable": var, "level": level_str}
                for label, series in (("SCMR", s_ref), ("External", s_ext)):
                    mask = series.notna()
                    non_missing = int(mask.sum())
                    if non_missing == 0:
                        row[label] = ""
                    else:
                        count = int((series.astype("object") == level).sum())
                        pct = (count / max(non_missing, 1)) * 100.0
                        row[label] = f"{count} ({pct:.1f}%)"
                rows.append(row)
        else:
            numeric_ref = pd.to_numeric(s_ref, errors="coerce")
            numeric_ext = pd.to_numeric(s_ext, errors="coerce")
            row = {"variable": var, "level": ""}
            row["SCMR_mean"] = float(numeric_ref.mean()) if len(numeric_ref) else np.nan
            row["SCMR_sd"] = float(numeric_ref.std()) if len(numeric_ref) else np.nan
            row["External_mean"] = (
                float(numeric_ext.mean()) if len(numeric_ext) else np.nan
            )
            row["External_sd"] = (
                float(numeric_ext.std()) if len(numeric_ext) else np.nan
            )
            row["SCMR_missing"] = int(numeric_ref.isna().sum())
            row["External_missing"] = int(numeric_ext.isna().sum())
            rows.append(row)

    return pd.DataFrame(rows)


def _generate_tableone_outputs(
    combined: pd.DataFrame,
    ref_table: pd.DataFrame,
    ext_table: pd.DataFrame,
    variables: Sequence[str],
    categorical_cols: Sequence[str],
    output_excel: str,
    output_csv: str,
) -> pd.DataFrame:
    """Attempt to build a TableOne; fall back to manual summary on failure."""

    try:
        from tableone import TableOne  # type: ignore

        tab1 = TableOne(
            combined,
            columns=list(variables),
            categorical=list(categorical_cols),
            groupby="Cohort",
            pval=True,
            overall=False,
            missing=True,
            label_suffix=True,
        )
        table_df = getattr(tab1, "tableone", None)
        if table_df is None:
            try:
                table_df = tab1.to_dataframe()  # type: ignore[attr-defined]
            except Exception:
                table_df = None
        if table_df is not None:
            if output_excel:
                try:
                    table_df.to_excel(output_excel)
                except Exception as exc:
                    warnings.warn(f"Failed to write TableOne Excel output: {exc}")
            if output_csv:
                try:
                    table_df.to_csv(output_csv)
                except Exception as exc:
                    warnings.warn(f"Failed to write TableOne CSV output: {exc}")
            return table_df
    except Exception as exc:
        warnings.warn(f"TableOne package unavailable or failed ({exc}); using fallback summary.")

    summary_df = _build_fallback_summary(ref_table, ext_table, variables, categorical_cols)
    if output_excel:
        try:
            summary_df.to_excel(output_excel, index=False)
        except Exception as exc:
            warnings.warn(f"Failed to write fallback Excel output: {exc}")
    if output_csv:
        try:
            summary_df.to_csv(output_csv, index=False)
        except Exception as exc:
            warnings.warn(f"Failed to write fallback CSV output: {exc}")
    return summary_df


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a TableOne comparing SCMR and an external cohort on shared features",
    )
    parser.add_argument(
        "--external-path",
        required=True,
        help="Path to the external dataset (CSV or Excel)",
    )
    parser.add_argument(
        "--external-sheet",
        default=None,
        help="Optional sheet name when reading Excel workbooks",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.getcwd(), "results", "tableone_local"),
        help="Directory to store outputs (default: ./results/tableone_local)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_excel = os.path.join(output_dir, "tableone_scmr_vs_external.xlsx")
    output_csv = os.path.join(output_dir, "tableone_scmr_vs_external.csv")
    feature_list_path = os.path.join(output_dir, "tableone_features.txt")

    _progress("Loading processed SCMR cohort...")
    ref_df = scmr.load_dataframes()

    _progress("Reading and harmonising external dataset...")
    ext_raw = eval_ext._read_external_dataframe(args.external_path, args.external_sheet)
    ext_raw = eval_ext._filter_rows_strict_binary(ext_raw)

    labels = ["VT/VF/SCD", "PE_Time", "ICD", "MRN"]
    feature_candidates = [c for c in ref_df.columns if c not in labels]
    requested_features = _dedupe_preserve_order(feature_candidates)

    _, ext_df = eval_ext._prepare_external_for_features(ext_raw, requested_features, labels)
    ext_df = eval_ext._augment_engineered_features_compat(ext_df)

    for df in (ref_df, ext_df):
        if "VT/VF/SCD" in df.columns and "PE" not in df.columns:
            df["PE"] = pd.to_numeric(df["VT/VF/SCD"], errors="coerce")
        if "PE_Time" in df.columns:
            df["PE_Time"] = pd.to_numeric(df["PE_Time"], errors="coerce")

    ref_order = [c for c in ref_df.columns if c not in {"MRN", "VT/VF/SCD"}]
    shared_columns = []
    for col in ref_order:
        if col in ext_df.columns:
            shared_columns.append(col)

    if "PE" in ref_df.columns and "PE" in ext_df.columns and "PE" not in shared_columns:
        shared_columns.insert(0, "PE")
    if (
        "PE_Time" in ref_df.columns
        and "PE_Time" in ext_df.columns
        and "PE_Time" not in shared_columns
    ):
        shared_columns.append("PE_Time")

    shared_columns = _dedupe_preserve_order(shared_columns)

    if not shared_columns:
        raise RuntimeError("No overlapping features found between SCMR and the external dataset.")

    categorical_cols = _infer_categorical(shared_columns, ref_df, ext_df)
    numeric_cols = [c for c in shared_columns if c not in set(categorical_cols)]

    ref_table = _ensure_table_columns(ref_df, shared_columns)
    ext_table = _ensure_table_columns(ext_df, shared_columns)

    for col in categorical_cols:
        if col in ref_table.columns:
            ref_table[col] = ref_table[col].astype("category")
        if col in ext_table.columns:
            ext_table[col] = ext_table[col].astype("category")

    for col in numeric_cols:
        ref_table[col] = pd.to_numeric(ref_table[col], errors="coerce")
        ext_table[col] = pd.to_numeric(ext_table[col], errors="coerce")

    ref_table["Cohort"] = "SCMR"
    ext_table["Cohort"] = "External"

    combined = pd.concat([ref_table, ext_table], ignore_index=True, sort=False)
    combined = combined[_dedupe_preserve_order(list(shared_columns) + ["Cohort"])]

    _progress(
        f"Found {len(shared_columns)} overlapping features (including PE and PE_Time where available).",
    )

    table_df = _generate_tableone_outputs(
        combined,
        ref_table,
        ext_table,
        shared_columns,
        categorical_cols,
        output_excel,
        output_csv,
    )

    try:
        with open(feature_list_path, "w", encoding="utf-8") as f:
            for name in shared_columns:
                f.write(f"{name}\n")
    except Exception as exc:
        warnings.warn(f"Failed to write feature list: {exc}")

    _progress(f"Completed. Outputs saved to: {output_dir}")
    if isinstance(table_df, pd.DataFrame):
        _progress(
            f"Summary rows: {len(table_df)} | Columns: {len(table_df.columns) if hasattr(table_df, 'columns') else 'n/a'}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
