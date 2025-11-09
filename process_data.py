import os
import glob
import pandas as pd

def load_and_merge_data(data_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"📂 Found {len(csv_files)} CSV files.")
    df_list = [pd.read_csv(f) for f in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"])
    merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)
    print("✅ Data merged and sorted by timestamp.")
    return merged_df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Rolling average & volume sum for the last 10 minutes (10 rows here)
    df["rolling_avg_10"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["volume_sum_10"] = df["volume"].rolling(window=10, min_periods=1).sum()

    # Future close price (5 minutes ahead)
    df["future_close"] = df["close"].shift(-5)
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    df = df.dropna(subset=["future_close"])
    print("✅ Features and target column computed.")
    return df


def main():
    input_dir = "data"
    output_path = "processed_features.csv"

    df = load_and_merge_data(input_dir)
    df = compute_features(df)
    df.to_csv(output_path, index=False)
    print(f"💾 Processed data saved to: {output_path}")


if __name__ == "__main__":
    main()
