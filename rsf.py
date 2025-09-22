from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

try:
    from missingpy import MissForest

    _HAS_MISSFOREST = True
except Exception:
    _HAS_MISSFOREST = False


def impute_misforest(X, random_seed):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    if not _HAS_MISSFOREST:
        # Fallback: simple mean imputation to keep script runnable
        X_imputed_scaled = X_scaled.fillna(X_scaled.mean())
    else:
        imputer = MissForest(random_state=random_seed)
        X_imputed_scaled = pd.DataFrame(
            imputer.fit_transform(X_scaled), columns=X.columns, index=X.index
        )
    X_imputed_unscaled = pd.DataFrame(
        scaler.inverse_transform(X_imputed_scaled), columns=X.columns, index=X.index
    )
    return X_imputed_unscaled


def conversion_and_imputation(df, features, labels):
    df = df.copy()
    df = df[features + labels]

    # Encode ordinal NYHA Class if present
    ordinal = "NYHA Class"
    if ordinal in df.columns:
        le = LabelEncoder()
        df[ordinal] = le.fit_transform(df[ordinal].astype(str))

    # Convert binary columns to numeric
    binary_cols = [
        "Female",
        "DM",
        "HTN",
        "HLP",
        "AF",
        "Beta Blocker",
        "ACEi/ARB/ARNi",
        "Aldosterone Antagonist",
        "VT/VF/SCD",
        "AAD",
        "CRT",
        "LGE_Circumural",
        "LGE_Ring-Like",
        "ICD",
    ]
    exist_bin = [c for c in binary_cols if c in df.columns]
    for c in exist_bin:
        if df[c].dtype == "object":
            df[c] = df[c].replace(
                {"Yes": 1, "No": 0, "Y": 1, "N": 0, "True": 1, "False": 0}
            )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Imputation on feature matrix
    X = df[features].copy()
    missing_cols = X.columns[X.isnull().any()].tolist()
    if missing_cols:
        imputed_part = impute_misforest(X[missing_cols], 0)
        imputed_X = X.copy()
        imputed_X[missing_cols] = imputed_part
    else:
        imputed_X = X.copy()

    imputed_X.index = df.index

    # Bring back key columns
    for col in ["MRN", "Female", "VT/VF/SCD", "ICD", "PE_Time"]:
        if col in df.columns:
            imputed_X[col] = df[col].values

    # Map to 0/1 by 0.5 threshold for binary cols
    for c in exist_bin:
        if c in imputed_X.columns:
            imputed_X[c] = (imputed_X[c] >= 0.5).astype(float)

    # Round NYHA Class
    if "NYHA Class" in imputed_X.columns:
        imputed_X["NYHA Class"] = imputed_X["NYHA Class"].round().astype("Int64")

    return imputed_X


def load_dataframes() -> pd.DataFrame:
    base = "/home/sunx/data/aiiih/projects/sunx/projects/ICD"
    icd = pd.read_excel(os.path.join(base, "LGE granularity.xlsx"), sheet_name="ICD")
    noicd = pd.read_excel(
        os.path.join(base, "LGE granularity.xlsx"), sheet_name="No_ICD"
    )

    # Stack dfs
    common_cols = icd.columns.intersection(noicd.columns)
    icd_common = icd[common_cols].copy()
    noicd_common = noicd[common_cols].copy()
    icd_common["ICD"] = 1
    noicd_common["ICD"] = 0
    nicm = pd.concat([icd_common, noicd_common], ignore_index=True)

    # PE time
    nicm["MRI Date"] = pd.to_datetime(nicm["MRI Date"])
    nicm["Date VT/VF/SCD"] = pd.to_datetime(nicm["Date VT/VF/SCD"])
    nicm["End follow-up date"] = pd.to_datetime(nicm["End follow-up date"])
    nicm["PE_Time"] = nicm.apply(
        lambda row: (
            (row["Date VT/VF/SCD"] - row["MRI Date"]).days
            if row["VT/VF/SCD"] == 1
            and pd.notna(row["Date VT/VF/SCD"])
            and pd.notna(row["MRI Date"])
            else (
                (row["End follow-up date"] - row["MRI Date"]).days
                if pd.notna(row["End follow-up date"]) and pd.notna(row["MRI Date"])
                else None
            )
        ),
        axis=1,
    )

    # Drop features
    nicm.drop(
        [
            "MRI Date",
            "Date VT/VF/SCD",
            "End follow-up date",
            "CRT Date",
            "LGE Burden 5SD",
            "LGE_Unnamed: 1",
            "LGE_Notes",
            "LGE_RV insertion sites (0 No, 1 yes)",
            "LGE_Score",
            "LGE_Unnamed: 27",
            "LGE_Unnamed: 28",
            "Cockcroft-Gault Creatinine Clearance (mL/min)",
        ],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    # Variables
    categorical = [
        "Female",
        "DM",
        "HTN",
        "HLP",
        "AF",
        "NYHA Class",
        "Beta Blocker",
        "ACEi/ARB/ARNi",
        "Aldosterone Antagonist",
        "VT/VF/SCD",
        "AAD",
        "CRT",
        "LGE_Extent (1; subendocardial, 2; mid mural, 3; epicardial, 4; transmural; 5 circumural)",
        "LGE_Circumural",
        "LGE_Ring-Like",
        "LGE_Basal anterior  (0; No, 1; yes)",
        "LGE_Basal anterior septum",
        "LGE_Basal inferoseptum",
        "LGE_Basal inferio",
        "LGE_Basal inferolateral",
        "LGE_Basal anterolateral ",
        "LGE_mid anterior",
        "LGE_mid anterior septum",
        "LGE_mid inferoseptum",
        "LGE_mid inferior",
        "LGE_mid inferolateral ",
        "LGE_mid anterolateral ",
        "LGE_apical anterior",
        "LGE_apical septum",
        "LGE_apical inferior",
        "LGE_apical lateral",
        "LGE_Apical cap",
        "LGE_RV insertion site (1 superior, 2 inferior. 3 both)",
        "ICD",
    ]
    nicm[categorical] = nicm[categorical].astype("object")
    var = nicm.columns.tolist()
    labels = ["MRN", "VT/VF/SCD", "ICD", "PE_Time"]
    granularity = [
        "LGE_LGE Burden 5SD",
        "LGE_Extent (1; subendocardial, 2; mid mural, 3; epicardial, 4; transmural; 5 circumural)",
        "LGE_Circumural",
        "LGE_Ring-Like",
        "LGE_Basal anterior  (0; No, 1; yes)",
        "LGE_Basal anterior septum",
        "LGE_Basal inferoseptum",
        "LGE_Basal inferio",
        "LGE_Basal inferolateral",
        "LGE_Basal anterolateral ",
        "LGE_mid anterior",
        "LGE_mid anterior septum",
        "LGE_mid inferoseptum",
        "LGE_mid inferior",
        "LGE_mid inferolateral ",
        "LGE_mid anterolateral ",
        "LGE_apical anterior",
        "LGE_apical septum",
        "LGE_apical inferior",
        "LGE_apical lateral",
        "LGE_Apical cap",
        "LGE_RV insertion site (1 superior, 2 inferior. 3 both)",
    ]
    nicm = nicm.dropna(subset=granularity)
    features = [v for v in var if v not in labels]

    # Imputation
    clean_df = conversion_and_imputation(nicm, features, labels)
    clean_df["NYHA Class"] = clean_df["NYHA Class"].replace({5: 4, 0: 1})

    return clean_df


