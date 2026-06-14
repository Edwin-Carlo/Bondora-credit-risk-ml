#!/usr/bin/env python
# =============================================
# SCRIPT DE PREDICCIÓN
# =============================================

import sys
import os
import argparse

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np

from src.config import ARTIFACTS_DIR
from src.predict import Predictor


def main():
    parser = argparse.ArgumentParser(description='Predicción de riesgo crediticio')
    parser.add_argument('--input', type=str, required=True, help='Archivo CSV con datos de entrada')
    parser.add_argument('--output', type=str, default='predicciones.csv', help='Archivo de salida')
    parser.add_argument('--threshold_low', type=float, default=0.3, help='Umbral bajo de riesgo')
    parser.add_argument('--threshold_high', type=float, default=0.7, help='Umbral alto de riesgo')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("INICIANDO PREDICCIÓN")
    print("=" * 60)
    
    # 1. Cargar datos
    print(f"\n1. Cargando datos desde: {args.input}")
    data = pd.read_csv(args.input)
    print(f"   Datos cargados: {data.shape}")
    
    # 2. Inicializar predictor
    print("\n2. Cargando modelo y pipeline...")
    predictor = Predictor(
        model_path=ARTIFACTS_DIR / "model_xgboost.pkl",
        pipeline_path=ARTIFACTS_DIR / "pipeline_preprocesamiento.pkl"
    )
    
    # 3. Realizar predicciones
    print("\n3. Realizando predicciones...")
    results = predictor.predict_with_risk_level(
        data,
        thresholds=(args.threshold_low, args.threshold_high)
    )
    
    # 4. Mostrar resumen
    print("\n4. Resumen de predicciones:")
    print(f"   Total préstamos: {len(results)}")
    print(f"   Riesgo Bajo: {(results['nivel_riesgo'] == 'Bajo').sum()}")
    print(f"   Riesgo Medio: {(results['nivel_riesgo'] == 'Medio').sum()}")
    print(f"   Riesgo Alto: {(results['nivel_riesgo'] == 'Alto').sum()}")
    print(f"   Defaults predichos: {results['prediccion'].sum()}")
    
    # 5. Guardar resultados
    print(f"\n5. Guardando resultados en: {args.output}")
    results.to_csv(args.output, index=False)
    
    print("\n" + "=" * 60)
    print("✅ PREDICCIÓN COMPLETADA")
    print("=" * 60)


if __name__ == "__main__":
    main()