# =============================================
# MÓDULO DE INGENIERÍA DE CARACTERÍSTICAS
# =============================================

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import MAP_ANTIGUEDAD, MAP_RATING


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Clase para la ingeniería de características.
    
    Responsabilidades:
    - Mapear categorías a valores numéricos
    - Crear nuevas features derivadas
    """
    
    def __init__(self):
        self.map_antiguedad = MAP_ANTIGUEDAD
        self.map_rating = MAP_RATING
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Aplica transformaciones de features."""
        df = X.copy()
        
        # 1. Mapear EmploymentDurationCurrentEmployer
        if 'EmploymentDurationCurrentEmployer' in df.columns:
            df['EmploymentDurationCurrentEmployer'] = (
                df['EmploymentDurationCurrentEmployer'].replace(self.map_antiguedad)
            )
        
        # 2. Mapear Rating
        if 'Rating' in df.columns:
            df['Rating'] = df['Rating'].replace(self.map_rating)
        
        # 3. Convertir a string para codificación posterior
        if 'HomeOwnershipType' in df.columns:
            df['HomeOwnershipType'] = df['HomeOwnershipType'].astype(str)
        
        if 'MaritalStatus' in df.columns:
            df['MaritalStatus'] = df['MaritalStatus'].astype(str)
        
        # 4. Crear feature: Tasa de interés relativa al monto
        if 'Interest' in df.columns and 'Amount' in df.columns:
            df['Interest_per_Amount'] = df['Interest'] / (df['Amount'] + 1)
        
        # 5. Crear feature: Pago mensual relativo al ingreso
        if 'MonthlyPayment' in df.columns and 'IncomeTotal' in df.columns:
            df['Payment_to_Income'] = df['MonthlyPayment'] / (df['IncomeTotal'] + 1)
        
        # 6. Crear feature: Edad al cuadrado
        if 'Age' in df.columns:
            df['Age_squared'] = df['Age'] ** 2
        
        # 7. Crear feature: Log del monto
        if 'Amount' in df.columns:
            df['Amount_log'] = np.log1p(df['Amount'])
        
        # 8. Crear feature: Log del ingreso
        if 'IncomeTotal' in df.columns:
            df['IncomeTotal_log'] = np.log1p(df['IncomeTotal'])
        
        return df