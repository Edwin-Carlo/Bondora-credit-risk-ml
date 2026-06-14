# =============================================
# MÓDULO DE PREPROCESAMIENTO
# =============================================

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import MAP_ANTIGUEDAD, MAP_RATING


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Clase para el preprocesamiento de datos de Bondora.
    
    Responsabilidades:
    - Cargar y validar datos
    - Crear variable objetivo DEFAULT
    - Eliminar préstamos activos (Current)
    - Eliminar columnas de identificación y leakage temporal
    - Eliminar columnas constantes
    - Imputar valores nulos
    """
    
    def __init__(self, target_col='DEFAULT', drop_current=True):
        """
        Inicializa el preprocesador.
        
        Parámetros:
        - target_col: nombre de la variable objetivo a crear
        - drop_current: si es True, elimina préstamos con Status='Current'
        """
        self.target_col = target_col
        self.drop_current = drop_current
        self.id_columns = [
            'LoanId', 'LoanNumber', 'UserName', 'ReportAsOfEOD',
            'ListedOnUTC', 'BiddingStartedOn', 'LoanApplicationStartedDate',
            'LoanDate', 'ContractEndDate', 'FirstPaymentDate',
            'MaturityDate_Original', 'MaturityDate_Last', 'LastPaymentOn',
            'DebtOccuredOn', 'DebtOccuredOnForSecondary', 'StageActiveSince',
            'GracePeriodStart', 'GracePeriodEnd', 'NextPaymentDate', 'ReScheduledOn'
        ]
        
        self.post_default_cols = [
            'PrincipalOverdueBySchedule', 'PlannedPrincipalPostDefault',
            'PlannedInterestPostDefault', 'EAD1', 'EAD2', 'PrincipalRecovery',
            'InterestRecovery', 'RecoveryStage', 'ExpectedLoss', 'LossGivenDefault',
            'ExpectedReturn', 'ProbabilityOfDefault', 'PrincipalWriteOffs',
            'InterestAndPenaltyWriteOffs', 'PrincipalDebtServicingCost',
            'InterestAndPenaltyDebtServicingCost', 'ActiveLateCategory',
            'ActiveLateLastPaymentCategory', 'WorseLateCategory', 'CurrentDebtDaysPrimary',
            'CurrentDebtDaysSecondary', 'NrOfScheduledPayments', 'NextPaymentNr',
            'CreditScoreFiAsiakasTietoRiskGrade', 'CreditScoreEsMicroL', 'CreditScoreEeMini'
        ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Aplica el preprocesamiento a los datos."""
        df = X.copy()
        
        # 1. Crear variable objetivo
        df[self.target_col] = df['DefaultDate'].notna().astype(int)
        
        # 2. Eliminar préstamos activos
        if self.drop_current and 'Status' in df.columns:
            df = df[df['Status'] != 'Current']
        
        # 3. Eliminar columnas de identificación
        cols_to_drop = [col for col in self.id_columns if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # 4. Eliminar columnas post-default (leakage)
        cols_to_drop = [col for col in self.post_default_cols if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # 5. Eliminar DefaultDate (ya se usó para crear target)
        if 'DefaultDate' in df.columns:
            df = df.drop(columns=['DefaultDate'])
        
        # 6. Eliminar columnas constantes
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        df = df.drop(columns=constant_cols)
        
        # 7. Imputar nulos
        for col in df.columns:
            if df[col].isnull().sum() > 0 and col != self.target_col:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        df[col] = df[col].fillna('Unknown')
        
        return df
    
    def get_feature_names(self):
        """Retorna los nombres de las columnas después del preprocesamiento."""
        return self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Clase para el tratamiento de outliers usando winsorización.
    
    Principio SOLID: Single Responsibility - Solo maneja outliers.
    """
    
    def __init__(self, columns, lower_percentile=0.01, upper_percentile=0.99):
        """
        Inicializa el manejador de outliers.
        
        Parámetros:
        - columns: lista de columnas a procesar
        - lower_percentile: percentil inferior para capping
        - upper_percentile: percentil superior para capping
        """
        self.columns = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds = {}
        self.upper_bounds = {}
    
    def fit(self, X, y=None):
        """Calcula los límites para cada columna."""
        for col in self.columns:
            if col in X.columns:
                self.lower_bounds[col] = X[col].quantile(self.lower_percentile)
                self.upper_bounds[col] = X[col].quantile(self.upper_percentile)
        return self
    
    def transform(self, X):
        """Aplica winsorización a los datos."""
        df = X.copy()
        for col in self.columns:
            if col in df.columns:
                df[col] = df[col].clip(self.lower_bounds[col], self.upper_bounds[col])
        return df