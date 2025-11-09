import os
import glob
import pandas as pd

def load_and_merge_data(data_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"📂 Found {len(csv_files)} CSV files.")

    df_list = []
    for file_path in csv_files:
        # Extract stock name from filename (everything before first "__")
        stock_name = os.path.basename(file_path).split("__")[0]

        df = pd.read_csv(file_path)
        df["stock_name"] = stock_name  # Add stock name column
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"])
    merged_df = merged_df.sort_values(["stock_name", "timestamp"]).reset_index(drop=True)

    print("✅ Data merged, stock names added, and sorted by timestamp.")
    return merged_df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Compute rolling features *per stock*
    df = df.groupby("stock_name", group_keys=False).apply(
        lambda x: x.assign(
            rolling_avg_10=x["close"].rolling(window=10, min_periods=1).mean(),
            volume_sum_10=x["volume"].rolling(window=10, min_periods=1).sum(),
        )
    )

    # Future close price and target (5 mins ahead)
    df["future_close"] = df.groupby("stock_name")["close"].shift(-5)
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    df = df.dropna(subset=["future_close"])
    print("✅ Features and target column computed.")
    return df


def main():
    input_dir = "data"
    output_path = "processed_features.parquet"

    df = load_and_merge_data(input_dir)
    df = compute_features(df)
    df.to_parquet(output_path, index=False)
    df.to_csv("processed_features.csv", index=False)
    print(f"💾 Processed data saved to: {output_path}")


if __name__ == "__main__":
    main()
