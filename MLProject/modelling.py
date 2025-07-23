import argparse
import os
import pandas as pd
import mlflow
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def main(data_path):
    # Start MLflow run
    with mlflow.start_run():

        # Load dataset
        df = pd.read_csv(data_path)

        # Features & label
        X = pd.get_dummies(df.drop(columns=["salary_bin"]))
        le = LabelEncoder()
        y = le.fit_transform(df["salary_bin"])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Model training
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluation
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"✅ Akurasi: {acc:.4f}")

        # Logging to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", acc)

        # Save artifacts
        os.makedirs("outputs", exist_ok=True)
        model_path = "outputs/model.pkl"
        le_path = "outputs/label_encoder.pkl"

        joblib.dump(model, model_path)
        joblib.dump(le, le_path)

        # Log artifacts to MLflow
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(le_path)

        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data_preprocessed.csv')
    args = parser.parse_args()
    main(args.data_path)
