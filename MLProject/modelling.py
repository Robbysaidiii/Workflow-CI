import argparse
import pandas as pd
import mlflow
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import mlflow.sklearn  # ⬅️ Tambahan penting

def main(data_path):
    with mlflow.start_run():
        df = pd.read_csv(data_path)
        X = pd.get_dummies(df.drop(columns=["salary_bin"]))
        y = LabelEncoder().fit_transform(df["salary_bin"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print("✅ Akurasi:", acc)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", acc)

        # Logging sebagai artifact biasa (opsional)
        os.makedirs("outputs", exist_ok=True)
        joblib.dump(model, "outputs/model.pkl")
        mlflow.log_artifact("outputs/model.pkl")

        # ✅ Logging model secara resmi dengan MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None  
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data_preprocessed.csv')
    args = parser.parse_args()
    main(args.data_path)
