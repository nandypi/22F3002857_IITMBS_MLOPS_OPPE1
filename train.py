from feast import FeatureStore
import pandas as pd

# Initialize Feast store
store = FeatureStore(repo_path="feature_repo")

# Load a small random sample
df = pd.read_parquet("processed_features.parquet").sample(n=200, random_state=42)

# Add event timestamp column
df["event_timestamp"] = pd.to_datetime(df["timestamp"])

# Define entity DataFrame
entity_df = df[["stock_name", "event_timestamp"]]

# 🔍 Print preview of sampled data
print("✅ Sampled data preview (first 5 rows):")
print(df.head())

print("\n📊 Columns available:")
print(df.columns.tolist())

print("\n📈 Entity DataFrame sample:")
print(entity_df.head())
