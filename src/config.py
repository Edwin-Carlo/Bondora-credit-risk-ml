# =============================================
# MÓDULO DE CONFIGURACIÓN
# =============================================

import os
from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Archivos de datos
RAW_DATA_FILE = DATA_RAW_DIR / "Bondora_raw.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "bondora_processed.csv"
ENGINEERED_DATA_FILE = DATA_PROCESSED_DIR / "bondora_engineered.csv"

# Archivos de artefactos
MODEL_FILE = ARTIFACTS_DIR / "model_xgboost.pkl"
PIPELINE_FILE = ARTIFACTS_DIR / "pipeline_preprocesamiento.pkl"
FEATURES_FILE = ARTIFACTS_DIR / "features.txt"
METRICS_FILE = ARTIFACTS_DIR / "metrics.txt"

# Parámetros del modelo
XGB_PARAMS = {
    'n_estimators': 400,
    'max_depth': 5,
    'learning_rate': 0.03,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.5,
    'reg_lambda': 2,
    'random_state': 42,
    'eval_metric': 'auc'
}

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1
}

# Columnas numéricas para tratamiento de outliers
NUM_COLS = [
    'DebtToIncome', 'FreeCash', 'IncomeTotal', 'ExistingLiabilities',
    'LiabilitiesTotal', 'MonthlyPayment', 'AppliedAmount', 'Amount',
    'NoOfPreviousLoansBeforeLoan', 'AmountOfPreviousLoansBeforeLoan',
    'PreviousEarlyRepaymentsCountBeforeLoan', 'Rating', 'Age',
    'EmploymentStatus', 'EmploymentDurationCurrentEmployer', 'Education',
    'VerificationType', 'Interest', 'LoanDuration', 'MonthlyPaymentDay'
]

# Columnas categóricas para rare encoding
RARE_ENCODE_COLS = ['HomeOwnershipType', 'MaritalStatus']

# Columnas para one-hot encoding
ONEHOT_COLS = ['HomeOwnershipType', 'MaritalStatus']

# Mapeo de categorías
MAP_ANTIGUEDAD = {
    'TrialPeriod': 0.25,
    'UpTo1Year': 0.5,
    'UpTo2Years': 1.5,
    'UpTo3Years': 2.5,
    'UpTo4Years': 3.5,
    'UpTo5Years': 4.5,
    'MoreThan5Years': 8.0,
    'Retiree': 10.0,
    'Other': 3.0
}

MAP_RATING = {
    'AA': 1, 'A': 2, 'B': 3,
    'C': 4, 'D': 5, 'E': 6,
    'F': 7, 'HR': 8
}

# Semilla para reproducibilidad
RANDOM_SEED = 42
TEST_SIZE = 0.3