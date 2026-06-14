#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path

# Rutas
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "bondora_processed.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

print("=" * 60)
print("ENTRENAMIENTO SIMPLIFICADO")
print("=" * 60)

# 1. Cargar datos
print("\n1. Cargando datos...")
df = pd.read_csv(DATA_FILE)
print(f"   Datos: {df.shape}")

# 2. Separar X e y
X = df.drop(columns=['DEFAULT'])
y = df['DEFAULT']

# 3. Seleccionar solo columnas numéricas
X = X.select_dtypes(include=[np.number])
print(f"   Features numéricas: {X.shape[1]}")

# 4. Dividir
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# 5. Entrenar XGBoost
print("\n2. Entrenando XGBoost...")
model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.5,
    reg_lambda=2,
    random_state=42,
    eval_metric='auc'
)
model.fit(X_train, y_train)
print("   ✅ Modelo entrenado")

# 6. Evaluar
print("\n3. Evaluando...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"   F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"   AUC:       {roc_auc_score(y_test, y_proba):.4f}")

# 7. Guardar
ARTIFACTS_DIR.mkdir(exist_ok=True)
joblib.dump(model, ARTIFACTS_DIR / "model_xgboost.pkl")
print(f"\n✅ Modelo guardado en: {ARTIFACTS_DIR / 'model_xgboost.pkl'}")