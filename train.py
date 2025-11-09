import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from feast import FeatureStore
import joblib
import mlflow
import mlflow.sklearn

# --------------------
# Feast feature retrieval
# --------------------
store = FeatureStore(repo_path="feature_repo")

# Load a small sample from processed features
df = pd.read_parquet("processed_features.parquet").sample(n=100, random_state=42)

# Feast expects event_timestamp column
df["event_timestamp"] = pd.to_datetime(df["timestamp"])
entity_df = df[["stock_name", "event_timestamp"]]

print("🚀 Retrieving features from Feast...")
feature_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "stock_features:rolling_avg_10",
        "stock_features:volume_sum_10",
        "stock_features:target",
    ],
).to_df()

print("✅ Feast feature retrieval successful!")
print("Feature data shape:", feature_df.shape)
print(feature_df.head())

# --------------------
# Prepare features and target
# --------------------
X = feature_df[["rolling_avg_10", "volume_sum_10"]]
y = feature_df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------
# MLflow setup
# --------------------
mlflow.set_tracking_uri("http://localhost:8000")
mlflow.set_experiment("stock_movement_logreg")

with mlflow.start_run():
    # Hyperparameter grid
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"]
    }

    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # --------------------
    # Evaluation
    # --------------------
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\n🎯 Model Accuracy: {accuracy:.4f}")
    print("Best Params:", grid.best_params_)

    # --------------------
    # Log with MLflow
    # --------------------
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="logreg_model",
        input_example=X_test.iloc[:5]
    )

    # --------------------
    # Save artifacts manually
    # --------------------
    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(best_model, "artifacts/logreg_model.pkl")

    with open("artifacts/metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n" + report + "\n")
        f.write("Confusion Matrix:\n" + str(conf_matrix) + "\n")

    coeffs = pd.DataFrame({
        "feature": X.columns,
        "coefficient": best_model.coef_[0]
    })
    coeffs.to_csv("artifacts/coefficients.csv", index=False)

    mlflow.log_artifact("artifacts/metrics.txt")
    mlflow.log_artifact("artifacts/coefficients.csv")
    mlflow.log_artifact("artifacts/logreg_model.pkl")

    print("\n💾 Model, metrics, and coefficients logged to MLflow and saved in 'artifacts/' folder.")
