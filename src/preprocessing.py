import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_dataset(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    """
    Remove duplicates and handle missing values safely.
    - Numeric columns: fill with median
    - Categorical columns: fill with 'missing'
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty!")

    if target_column and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    df = df.drop_duplicates().copy()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        df[c] = df[c].fillna("missing")

    return df

def scale_numeric(df: pd.DataFrame, numeric_cols: list[str], scaler: StandardScaler | None = None):
    """
    Standardize numerical columns.
    Returns (df_scaled, scaler) so you can reuse the scaler at inference time.
    """
    for col in numeric_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found for scaling")

    df = df.copy()
    scaler = scaler or StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler

def separate_features_target(df: pd.DataFrame, target_column: str):
    """
    Separate features and target variable.
    Returns X, y.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
