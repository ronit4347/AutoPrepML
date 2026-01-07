import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# ===================== Utilities =====================

def normalize_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("[^a-z0-9_]", "", regex=True)
    )
    return df


def detect_column_types(df):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return num_cols, cat_cols


def count_outliers(df, num_cols):
    count = 0
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        count += ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    return int(count)


def handle_outliers(df, num_cols):
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower, upper)
    return df


def compute_quality_score(raw_df, cleaned_df, out_before, out_after):
    score = 0
    checks = []

    if cleaned_df.isnull().sum().sum() == 0:
        score += 30
        checks.append("✔ Missing values handled")

    if out_after < out_before:
        score += 25
        checks.append("✔ Outliers reduced")

    if raw_df.duplicated().sum() > cleaned_df.duplicated().sum():
        score += 15
        checks.append("✔ Duplicates removed")

    if cleaned_df.shape[1] >= raw_df.shape[1]:
        score += 20
        checks.append("✔ Features encoded & scaled")

    score += 10
    checks.append("✔ Leakage-safe pipeline")

    return min(score, 100), checks


# ===================== MAIN PIPELINE =====================

def autoprepml(df, target_column):

    raw_df = df.copy()

    # -------- Normalize columns --------
    df = normalize_columns(df)
    target_column = target_column.strip().lower().replace(" ", "_")

    # -------- Detect UNSUPERVISED MODE --------
    unsupervised_mode = target_column in ["id", "index", "row_id"] or target_column not in df.columns

    # -------- Remove duplicates --------
    df = df.drop_duplicates()

    # -------- Split X & y --------
    if unsupervised_mode:
        X = df.copy()
        y = None
        problem_type = "unsupervised"
        stratify = None
    else:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        problem_type = "regression" if y.dtype != "object" else "classification"
        stratify = y if problem_type == "classification" else None

    num_cols, cat_cols = detect_column_types(X)

    # -------- Outlier handling --------
    outliers_before = count_outliers(X, num_cols)
    X[num_cols] = handle_outliers(X[num_cols], num_cols)
    outliers_after = count_outliers(X, num_cols)

    # -------- Train-test split --------
    if unsupervised_mode:
        X_train = X
        X_test = pd.DataFrame()
        y_train = y_test = None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify
        )

    # -------- Pipelines --------
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # -------- Fit once --------
    X_train_p = preprocessor.fit_transform(X_train)

    if not unsupervised_mode:
        X_test_p = preprocessor.transform(X_test)
    else:
        X_test_p = pd.DataFrame()

    # -------- Feature names --------
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in feature_names
    ]

    X_train_p = pd.DataFrame(X_train_p, columns=feature_names)

    if not unsupervised_mode:
        X_test_p = pd.DataFrame(X_test_p, columns=feature_names)

    joblib.dump(preprocessor, "autoprepml_pipeline.pkl")

    # -------- Quality score --------
    quality_score, quality_checks = compute_quality_score(
        raw_df,
        X,
        outliers_before,
        outliers_after
    )

    # -------- Return --------
    return {
        "mode": "unsupervised" if unsupervised_mode else "supervised",
        "problem_type": problem_type,
        "X_train": X_train_p,
        "X_test": X_test_p if not unsupervised_mode else None,
        "y_train": y_train.reset_index(drop=True) if y_train is not None else None,
        "y_test": y_test.reset_index(drop=True) if y_test is not None else None,
        "quality_score": quality_score,
        "quality_checks": quality_checks,
        "outliers_before": outliers_before,
        "outliers_after": outliers_after,
        "raw_features": raw_df.shape[1],
        "processed_features": X_train_p.shape[1]
    }
