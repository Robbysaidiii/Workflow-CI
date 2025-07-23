import argparse
import os
import pandas as pd
import mlflow
import joblib
import shutil
from mlflow.artifacts import download_artifacts

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def main(data_path):
    with mlflow.start_run() as run:
        # Load dataset
        df = pd.read_csv(data_path)

        # Features & label
        X = pd.get_dummies(df.drop(columns=["salary_bin"]))
        le = LabelEncoder()
        y = le.fit_transform(df["salary_bin"])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"✅ Akurasi: {acc:.4f}")

        # Log metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", acc)

        # Log raw model & label encoder
        os.makedirs("outputs", exist_ok=True)
        joblib.dump(model, "outputs/model.pkl")
        joblib.dump(le, "outputs/label_encoder.pkl")
        mlflow.log_artifact("outputs/model.pkl")
        mlflow.log_artifact("outputs/label_encoder.pkl")

        # Log as MLflow model
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

        # Download model artifact from remote (DAGsHub)
        run_id = run.info.run_id
        source_path = download_artifacts(artifact_uri=f"runs:/{run_id}/sklearn-model")
        target_path = "outputs/mlflow-model"

        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.copytree(source_path, target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data_preprocessed.csv')
    args = parser.parse_args()
    main(args.data_path)
