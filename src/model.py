# =============================================
# MÓDULO DE MODELO (POO + SOLID)
# =============================================

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from xgboost import XGBClassifier
import joblib
import os


class RiskModel(BaseEstimator, ClassifierMixin):
    """
    Clase base para modelos de riesgo crediticio.
    
    Principio SOLID: Open/Closed - Abierto para extensión, cerrado para modificación.
    """
    
    def __init__(self, model_type='xgboost', params=None, random_state=42):
        """
        Inicializa el modelo.
        
        Parámetros:
        - model_type: 'xgboost' o 'random_forest'
        - params: diccionario de parámetros del modelo
        - random_state: semilla para reproducibilidad
        """
        self.model_type = model_type
        self.params = params or {}
        self.random_state = random_state
        self.model = None
        
def _create_model(self):
    """Crea la instancia del modelo según el tipo."""
    if self.model_type == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(**self.params, random_state=self.random_state)
    elif self.model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**self.params, random_state=self.random_state)
    else:
        raise ValueError(f"Modelo {self.model_type} no soportado")
    
    def fit(self, X, y):
        """Entrena el modelo."""
        self.model = self._create_model()
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Realiza predicciones de clase."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Realiza predicciones de probabilidad."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo y retorna métricas.
        
        Retorna:
        - Diccionario con accuracy, precision, recall, f1, auc
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        return metrics
    
    def get_confusion_matrix(self, X_test, y_test):
        """Retorna la matriz de confusión."""
        y_pred = self.predict(X_test)
        return confusion_matrix(y_test, y_pred)
    
    def save(self, filepath):
        """Guarda el modelo en disco."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        """Carga el modelo desde disco."""
        self.model = joblib.load(filepath)
        return self


class RiskModelOptimizer:
    """
    Clase para optimización de hiperparámetros.
    
    Principio SOLID: Single Responsibility - Solo se encarga de optimizar.
    """
    
    def __init__(self, model_type='xgboost', param_grid=None, cv=5, scoring='roc_auc'):
        """
        Inicializa el optimizador.
        
        Parámetros:
        - model_type: tipo de modelo a optimizar
        - param_grid: diccionario de parámetros a probar
        - cv: número de folds para validación cruzada
        - scoring: métrica a optimizar
        """
        self.model_type = model_type
        self.param_grid = param_grid or self._default_param_grid()
        self.cv = cv
        self.scoring = scoring
        self.best_model = None
        self.best_params = None
        self.best_score = None
    
    def _default_param_grid(self):
        """Parámetros por defecto para búsqueda."""
        if self.model_type == 'xgboost':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        else:
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            }
    
    def _create_base_model(self):
        """Crea el modelo base."""
        if self.model_type == 'xgboost':
            return XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=42)
    
    def optimize(self, X_train, y_train):
        """
        Ejecuta la optimización de hiperparámetros.
        
        Retorna:
        - Mejor modelo, mejores parámetros, mejor puntuación
        """
        base_model = self._create_base_model()
        
        grid_search = GridSearchCV(
            base_model,
            self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        return self.best_model, self.best_params, self.best_score