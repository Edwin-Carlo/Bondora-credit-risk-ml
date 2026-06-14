# =============================================
# MÓDULO DE UTILIDADES
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def outliers_col(df):
    """
    Detecta outliers en cada columna numérica usando el método IQR.
    
    Parámetros:
    - df: DataFrame a analizar
    
    Retorna:
    - DataFrame con conteo de outliers por columna
    """
    resultados = []
    for columna in df:
        if df[columna].dtype != object:
            q1 = stats.scoreatpercentile(df[columna], 25)
            q3 = stats.scoreatpercentile(df[columna], 75)
            iqr = q3 - q1
            lim_inf = q1 - 1.5 * iqr
            lim_sup = q3 + 1.5 * iqr
            n_outliers_inf = len(df[(df[columna] < lim_inf)])
            n_outliers_sup = len(df[(df[columna] > lim_sup)])
            resultados.append({
                'columna': df[columna].name,
                'outliers_inf': n_outliers_inf,
                'outliers_sup': n_outliers_sup,
                'total_outliers': n_outliers_inf + n_outliers_sup,
                'porcentaje': (n_outliers_inf + n_outliers_sup) / len(df) * 100
            })
    return pd.DataFrame(resultados).sort_values('total_outliers', ascending=False)

def plot_distribuciones(df, features, target='DEFAULT'):
    """
    Grafica histogramas de features por clase de target.
    
    Parámetros:
    - df: DataFrame con los datos
    - features: lista de columnas a graficar
    - target: nombre de la columna objetivo
    """
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        if col in df.columns:
            for valor in [0, 1]:
                subset = df[df[target] == valor]
                axes[i].hist(subset[col], bins=30, alpha=0.6,
                            label=f'{target}={valor}', density=True)
            axes[i].set_title(f'Distribución de {col}')
            axes[i].set_xlabel(col)
            axes[i].legend()
        else:
            axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_matriz_correlacion(df, features, target='DEFAULT'):
    """
    Grafica la matriz de correlación de las features con el target.
    
    Parámetros:
    - df: DataFrame con los datos
    - features: lista de columnas a incluir
    - target: nombre de la columna objetivo
    """
    corr = df[features + [target]].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    return plt.gcf()