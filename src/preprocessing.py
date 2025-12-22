import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_dataset(df, target_column=None):
    """
    Remove duplicates and handle missing values.
    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Optional target column.
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.drop_duplicates()
    df = df.fillna(0)
    return df

def scale_numeric(df, numeric_cols):
    """
    Standardize numerical columns using StandardScaler.
    Args:
        df (pd.DataFrame): Input dataframe.
        numeric_cols (list): List of numeric column names.
    Returns:
        pd.DataFrame: Scaled dataframe.
    """
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def separate_features_target(df, target_column):
    """
    Separate features and target variable.
    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of target column.
    Returns:
        X (pd.DataFrame), y (pd.Series)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
