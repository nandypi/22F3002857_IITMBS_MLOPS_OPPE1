from feast import FeatureStore
import pandas as pd

# Initialize Feast
store = FeatureStore(repo_path="feature_repo")

# Load your processed Parquet file
entity_df = pd.read_parquet("processed_features.parquet")
entity_df = entity_df.sample(n=1000, random_state=42)

# Feast expects an event timestamp column
if "timestamp" in entity_df.columns:
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["timestamp"])

# Retrieve features from Feast
feature_data = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "stock_features:rolling_avg_10",
        "stock_features:volume_sum_10",
        "stock_features:target",
    ],
).to_df()

# Display results
print("✅ Feast feature retrieval successful!")
print("Shape:", feature_data.shape)
print(feature_data.head())

# Save a local preview
feature_data.to_csv("feast_output_preview.csv", index=False)
print("\n💾 Saved preview to feast_output_preview.csv")
