import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data_preprocessed.csv')
args = parser.parse_args()

mlflow.set_tracking_uri("https://dagshub.com/Robbysaidiii/my-first-repo.mlflow")
mlflow.set_experiment("Salary Prediction")

with mlflow.start_run():
    # Load data
    df = pd.read_csv(args.data_path)
    X = pd.get_dummies(df.drop(columns=["salary_bin"]))
    y = LabelEncoder().fit_transform(df["salary_bin"])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("✅ Akurasi:", acc)

    # Logging
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, artifact_path="model")
