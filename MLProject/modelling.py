import argparse
import pandas as pd
import mlflow
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def main(data_path):
    # Hapus: mlflow.set_tracking_uri()
    # Hapus: mlflow.set_experiment()

    with mlflow.start_run():  # ❌ JANGAN nested, biarkan CLI kontrol
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

        os.makedirs("outputs", exist_ok=True)
        model_path = "outputs/model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data_preprocessed.csv')
    args = parser.parse_args()
    main(args.data_path)
