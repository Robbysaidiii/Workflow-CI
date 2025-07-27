# modelling.py

import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def main(data_path):
    # === 1. Load data ===
    df = pd.read_csv(data_path)

    X = pd.get_dummies(df.drop(columns=["salary_bin"]))
    y = LabelEncoder().fit_transform(df["salary_bin"])

    # === 2. Split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # === 3. Train model ===
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # === 4. Evaluate ===
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    # === 5. Log to MLflow ===
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(model,  path="outputs/mlflow-model")

        print("✅ Model berhasil dicatat.")
        print(f"🔢 Akurasi: {acc:.4f}")
        print(f"🎯 F1-score: {f1:.4f}")
        print(f"📌 Precision: {precision:.4f}")
        print(f"📌 Recall: {recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
