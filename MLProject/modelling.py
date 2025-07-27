import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import argparse
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib


def main(data_path):
    """
    Main training function
    """
    # === 1. Setup MLflow tracking dengan DagsHub ===
    try:
        # Set DagsHub MLflow tracking URI
        mlflow_tracking_uri = "https://dagshub.com/Robbysaidiii/my-first-repo.mlflow"
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Setup authentication untuk CI/CD
        if os.getenv('GITHUB_ACTIONS'):
            # CI/CD environment - use username/password authentication
            dagshub_username = os.getenv('DAGSHUB_USERNAME', 'Robbysaidiii')
            dagshub_password = os.getenv('DAGSHUB_PASSWORD')
            
            if dagshub_password:
                os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
                os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_password
                print(f"🔧 CI/CD Mode - Using DagsHub with username: {dagshub_username}")
            else:
                print("⚠️  DAGSHUB_PASSWORD not found, may face authentication issues")
        else:
            # Local development - use dagshub.init
            dagshub.init(repo_owner="Robbysaidiii", repo_name="my-first-repo", mlflow=True)
            print("🔧 Local Mode - Connected to DagsHub via dagshub.init")
            
        print(f"📊 MLflow Tracking URI: {mlflow_tracking_uri}")
        
    except Exception as e:
        print(f"❌ DagsHub setup failed: {e}")
        # Don't fallback to local, let it fail to ensure DagsHub tracking
        raise
    
    mlflow.set_experiment("Income Classification Tuning")
    
    # === 2. Load data ===
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} tidak ditemukan!")
    
    df = pd.read_csv(data_path)
    print(f"📊 Data loaded: {df.shape}")
    
    # Cek apakah kolom target ada
    if "salary_bin" not in df.columns:
        raise ValueError("Kolom target 'salary_bin' tidak ditemukan di data.")
    
    # === 3. Pisahkan fitur dan target ===
    X = df.drop(columns=["salary_bin"])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["salary_bin"])
    
    # === 4. Deteksi kolom numerik dan kategorikal ===
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    print("📊 Fitur kategorikal:", categorical_cols)
    print("🔢 Fitur numerik:", numerical_cols)
    
    # === 5. Preprocessing pipeline ===
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    
    # === 6. Pipeline modelling ===
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    
    # === 7. Parameter grid untuk tuning ===
    param_grid = {
        "classifier__n_estimators": [100, 150, 200],
        "classifier__max_depth": [10, 15, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2]
    }
    
    # === 8. Split data ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    
    # === 9. Grid Search ===
    print("🔍 Starting hyperparameter tuning...")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="accuracy",
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"✅ Best parameters: {grid.best_params_}")
    
    # === 10. Evaluasi model ===
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    
    # === 11. Create outputs directory ===
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # === 12. Logging ke MLflow ===
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("data_shape", str(df.shape))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("cv_folds", 3)
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_dict(class_report, "classification_report.json")
        
        # Save and log model
        model_path = outputs_dir / "mlflow-model"
        mlflow.sklearn.log_model(
            best_model, 
            "model",
            registered_model_name="income-classification-model"
        )
        
        # Save model locally for Docker
        mlflow.sklearn.save_model(best_model, str(model_path))
        
        # Save label encoder
        joblib.dump(label_encoder, outputs_dir / "label_encoder.pkl")
        
        # Save feature info
        feature_info = {
            "categorical_cols": categorical_cols,
            "numerical_cols": numerical_cols
        }
        mlflow.log_dict(feature_info, "feature_info.json")
        
        print("✅ Model dan pipeline berhasil dicatat di DagsHub.")
        print(f"🔢 Akurasi: {acc:.4f}")
        print(f"🎯 F1-score: {f1:.4f}")
        print(f"📌 Precision: {precision:.4f}")
        print(f"📌 Recall: {recall:.4f}")
        print(f"💾 Model saved to: {model_path}")
        
        return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Income Classification Model")
    parser.add_argument("--data_path", type=str, default="data_preprocessed.csv",
                       help="Path to the preprocessed data file")
    
    args = parser.parse_args()
    
    try:
        accuracy = main(args.data_path)
        print(f"Training completed successfully with accuracy: {accuracy:.4f}")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        sys.exit(1)