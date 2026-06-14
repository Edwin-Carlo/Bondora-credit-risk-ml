# =============================================
# MÓDULO DE PREDICCIÓN
# =============================================

import pandas as pd
import numpy as np
import joblib


class Predictor:
    """
    Clase para realizar predicciones con el modelo entrenado.
    
    Principio SOLID: Single Responsibility - Solo se encarga de predecir.
    """
    
    def __init__(self, model_path=None, pipeline_path=None):
        """
        Inicializa el predictor.
        
        Parámetros:
        - model_path: ruta al archivo del modelo (.pkl)
        - pipeline_path: ruta al archivo del pipeline (.pkl)
        """
        self.model = None
        self.pipeline = None
        
        if model_path:
            self.load_model(model_path)
        if pipeline_path:
            self.load_pipeline(pipeline_path)
    
    def load_model(self, filepath):
        """Carga el modelo desde disco."""
        self.model = joblib.load(filepath)
        return self
    
    def load_pipeline(self, filepath):
        """Carga el pipeline desde disco."""
        self.pipeline = joblib.load(filepath)
        return self
    
    def predict(self, data):
        """
        Realiza predicción de clase.
        
        Parámetros:
        - data: DataFrame o array con los datos de entrada
        
        Retorna:
        - Array con predicciones (0 o 1)
        """
        if self.pipeline is not None:
            data = self.pipeline.transform(data)
        return self.model.predict(data)
    
    def predict_proba(self, data):
        """
        Realiza predicción de probabilidad.
        
        Parámetros:
        - data: DataFrame o array con los datos de entrada
        
        Retorna:
        - Array con probabilidades (0 a 1)
        """
        if self.pipeline is not None:
            data = self.pipeline.transform(data)
        return self.model.predict_proba(data)[:, 1]
    
    def predict_with_risk_level(self, data, thresholds=(0.3, 0.7)):
        """
        Realiza predicción con niveles de riesgo.
        
        Parámetros:
        - data: DataFrame o array con los datos de entrada
        - thresholds: tupla (bajo, alto) con los umbrales de riesgo
        
        Retorna:
        - DataFrame con predicciones y niveles de riesgo
        """
        proba = self.predict_proba(data)
        
        results = pd.DataFrame({
            'probabilidad_default': proba,
            'prediccion': (proba >= 0.5).astype(int)
        })
        
        # Asignar nivel de riesgo
        results['nivel_riesgo'] = 'Medio'
        results.loc[proba < thresholds[0], 'nivel_riesgo'] = 'Bajo'
        results.loc[proba > thresholds[1], 'nivel_riesgo'] = 'Alto'
        
        return results