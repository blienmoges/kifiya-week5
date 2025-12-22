# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_dataset(df, target_column):
    """Removes duplicates and handles missing values"""
    df = df.drop_duplicates()
    df = df.fillna(0)  # or other strategy
    return df

def scale_numeric(df, numeric_cols):
    """Standardizes numeric features"""
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def separate_features_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
