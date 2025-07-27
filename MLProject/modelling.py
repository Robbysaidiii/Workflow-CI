import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# === 0. Parsing argumen dari MLflow ===
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data_preprocessed.csv")
args = parser.parse_args()

# === FIX: Clear any orphaned run state ===
try:
    mlflow.end_run()
except:
    pass

# Clear environment variable if exists
if 'MLFLOW_RUN_ID' in os.environ:
    del os.environ['MLFLOW_RUN_ID']

# === 1. Inisialisasi koneksi ke DagsHub ===
# Different approach for GitHub Actions vs local
if os.environ.get('GITHUB_ACTIONS'):
    # In GitHub Actions: Use direct MLflow connection
    print("🤖 Running in GitHub Actions - using direct MLflow connection")
    mlflow.set_tracking_uri("https://dagshub.com/Robbysaidiii/my-first-repo.mlflow")
    
    # Verify environment variables are set
    if not os.environ.get('MLFLOW_TRACKING_USERNAME'):
        raise ValueError("MLFLOW_TRACKING_USERNAME not set")
    if not os.environ.get('MLFLOW_TRACKING_PASSWORD'):
        raise ValueError("MLFLOW_TRACKING_PASSWORD not set")
    
    # Set experiment directly
    try:
        mlflow.set_experiment("Income Classification Tuning")
        print("✅ MLflow experiment set successfully")
    except Exception as e:
        print(f"❌ Failed to set experiment: {e}")
        # Create experiment if it doesn't exist
        experiment_id = mlflow.create_experiment("Income Classification Tuning")
        mlflow.set_experiment("Income Classification Tuning")
        print("✅ Created and set new experiment")
else:
    # Local development: Use dagshub.init()
    print("🏠 Running locally - using DagsHub init")
    try:
        dagshub.init(repo_owner="Robbysaidiii", repo_name="my-first-repo", mlflow=True)
        mlflow.set_experiment("Income Classification Tuning")
        print("✅ DagsHub initialization successful")
    except Exception as e:
        print(f"⚠️ DagsHub init failed: {e}")
        print("📝 Using local MLflow tracking")
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Income Classification Tuning")

# === 2. Load data ===
df = pd.read_csv(args.data_path)

# Cek apakah kolom target ada
if "salary_bin" not in df.columns:
    raise ValueError("Kolom target 'salary_bin' tidak ditemukan di data.")

# === 3. Pisahkan fitur dan target ===
X = df.drop(columns=["salary_bin"])
y = LabelEncoder().fit_transform(df["salary_bin"])

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
    "classifier__n_estimators": [100, 150],
    "classifier__max_depth": [10, 15],
}

# === 8. Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# === 9. Grid Search ===
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring="accuracy"
)

print("🔄 Starting Grid Search...")
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("✅ Grid Search completed")

# === 10. Evaluasi model ===
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")

# === 11. Logging ke MLflow ===
# Force new run with unique name
import time
run_name = f"income_classification_{int(time.time())}"

try:
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Simpan pipeline lengkap (preprocessing + model)
        mlflow.sklearn.log_model(best_model, "model")
        
        # Log current run ID for later use
        current_run = mlflow.active_run()
        print(f"🆔 Current run ID: {current_run.info.run_id}")
        
        print("✅ Model dan pipeline berhasil dicatat di MLflow.")
        print(f"🔢 Akurasi: {acc:.4f}")
        print(f"🎯 F1-score: {f1:.4f}")
        print(f"📌 Precision: {precision:.4f}")
        print(f"📌 Recall: {recall:.4f}")

except Exception as e:
    print(f"❌ MLflow logging failed: {e}")
    print("📊 Model Results (not logged):")
    print(f"🔢 Akurasi: {acc:.4f}")
    print(f"🎯 F1-score: {f1:.4f}")
    print(f"📌 Precision: {precision:.4f}")
    print(f"📌 Recall: {recall:.4f}")
    raise