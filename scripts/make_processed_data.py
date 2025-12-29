import os
import numpy as np
import pandas as pd


RAW_DIR = "data"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)


def ip_to_int(ip):
    """
    Fraud_Data.csv sometimes has ip_address as float/string.
    Convert to int safely.
    """
    # If already numeric:
    if pd.api.types.is_numeric_dtype(ip):
        return ip.fillna(0).astype("int64")

    # If string-ish, try numeric conversion
    ip_num = pd.to_numeric(ip, errors="coerce").fillna(0)
    return ip_num.astype("int64")


def build_fraud_processed():
    fraud_path = os.path.join(RAW_DIR, "Fraud_Data.csv")
    ipmap_path = os.path.join(RAW_DIR, "IpAddress_to_Country.csv")

    df = pd.read_csv(fraud_path)
    ipmap = pd.read_csv(ipmap_path)

    # --- Basic cleaning ---
    df = df.drop_duplicates().copy()

    # Parse timestamps
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")

    # Drop rows with invalid timestamps (small usually). If large, you can impute instead.
    df = df.dropna(subset=["signup_time", "purchase_time"]).copy()

    # Fix target
    df["class"] = pd.to_numeric(df["class"], errors="coerce").fillna(0).astype(int)

    # Ensure ip integer
    df["ip_int"] = ip_to_int(df["ip_address"])

    # --- Feature engineering (Task 1 minimum) ---
    # time_since_signup in seconds -> also in hours
    tdelta = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
    df["time_since_signup_seconds"] = tdelta
    df["time_since_signup_hours"] = tdelta / 3600.0

    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek  # 0=Mon

    # Simple velocity / frequency features
    # Transactions per user (overall)
    user_tx_counts = df.groupby("user_id")["purchase_time"].transform("count")
    df["user_tx_count_total"] = user_tx_counts

    # Users per device (overall)
    users_per_device = df.groupby("device_id")["user_id"].transform("nunique")
    df["users_per_device"] = users_per_device

    # Devices per user (overall)
    devices_per_user = df.groupby("user_id")["device_id"].transform("nunique")
    df["devices_per_user"] = devices_per_user

    # --- IP -> Country range join ---
    # Clean ipmap types
    ipmap["lower_bound_ip_address"] = pd.to_numeric(
        ipmap["lower_bound_ip_address"], errors="coerce"
    ).astype("int64")
    ipmap["upper_bound_ip_address"] = pd.to_numeric(
        ipmap["upper_bound_ip_address"], errors="coerce"
    ).astype("int64")

    ipmap = ipmap.dropna(subset=["lower_bound_ip_address", "upper_bound_ip_address"]).copy()
    ipmap = ipmap.sort_values("lower_bound_ip_address")

    # Sort fraud df for merge_asof
    df = df.sort_values("ip_int")

    # merge_asof: match each ip to nearest lower_bound <= ip
    merged = pd.merge_asof(
        df,
        ipmap[["lower_bound_ip_address", "upper_bound_ip_address", "country"]].sort_values("lower_bound_ip_address"),
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward",
        allow_exact_matches=True
    )

    # Keep only if ip_int <= upper_bound; else country unknown
    in_range = merged["ip_int"] <= merged["upper_bound_ip_address"]
    merged.loc[~in_range, "country"] = "Unknown"
    merged["country"] = merged["country"].fillna("Unknown")

    # Optional: drop helper columns you don’t want
    # (keep ip_int, it can help modeling)
    # merged = merged.drop(columns=["lower_bound_ip_address", "upper_bound_ip_address"])

    # Handle missing values simply (you can justify later)
    # Numeric: fill with median; categorical: fill Unknown
    num_cols = merged.select_dtypes(include="number").columns
    cat_cols = merged.select_dtypes(exclude="number").columns

    for c in num_cols:
        merged[c] = merged[c].fillna(merged[c].median())

    for c in cat_cols:
        merged[c] = merged[c].fillna("Unknown")

    out_path = os.path.join(OUT_DIR, "fraud_data_processed.csv")
    merged.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Shape:", merged.shape)
    print("Fraud rate:", merged["class"].mean())


def build_creditcard_processed():
    cc_path = os.path.join(RAW_DIR, "creditcard.csv")
    df = pd.read_csv(cc_path)

    df = df.drop_duplicates().copy()
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)

    # Basic missing handling
    num_cols = df.select_dtypes(include="number").columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    out_path = os.path.join(OUT_DIR, "creditcard_processed.csv")
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Shape:", df.shape)
    print("Fraud rate:", df["Class"].mean())


if __name__ == "__main__":
    build_fraud_processed()
    build_creditcard_processed()
