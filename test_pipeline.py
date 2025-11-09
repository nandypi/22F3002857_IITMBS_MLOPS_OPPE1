import os
import pandas as pd
from feast import FeatureStore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def test_full_pipeline():
    """End-to-end test for Feast + model training + artifact saving."""

    print("🚀 Starting full pipeline test...")

    # 1️⃣ Load processed data
    assert os.path.exists("processed_features.parquet"), "❌ Missing processed_features.parquet file!"
    df = pd.read_parquet("processed_features.parquet").sample(100, random_state=42)
    print(f"✅ Loaded processed data — {df.shape[0]} rows")

    # 2️⃣ Initialize Feast and retrieve features
    store = FeatureStore(repo_path="feature_repo")

    df["event_timestamp"] = pd.to_datetime(df["timestamp"])
    entity_df = df[["stock_name", "event_timestamp"]]

    print("🔍 Fetching features from Feast...")
    features = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "stock_features:rolling_avg_10",
            "stock_features:volume_sum_10",
            "stock_features:target",
        ],
    ).to_df()

    assert not features.empty, "❌ No features retrieved!"
    print(f"✅ Feast retrieval success — got {features.shape[0]} rows")

    # 3️⃣ Train a tiny logistic regression model
    X = features[["rolling_avg_10", "volume_sum_10"]]
    y = features["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Model accuracy: {acc:.4f}")

    # 4️⃣ Save model + simple metric
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/test_model.pkl"
    joblib.dump(model, model_path)

    with open("artifacts/test_metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")

    print(f"💾 Model saved at {model_path}")
    print("✅ Full pipeline test passed!\n")

    # Optional assertion for CI pass/fail
    assert acc >= 0.4, f"❌ Accuracy too low ({acc:.4f}) — possible pipeline issue."

if __name__ == "__main__":
    test_full_pipeline()
