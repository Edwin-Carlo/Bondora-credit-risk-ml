# Bondora Credit Risk ML

## Clasificación de Riesgo Crediticio en Plataforma P2P

---

## a. Problema de Machine Learning

### Contexto de Negocio

Las plataformas de préstamos Peer‑to‑Peer (P2P) conectan directamente a inversores con prestatarios, eliminando intermediarios financieros tradicionales. Bondora, fundada en 2008 en Estonia, es una de las plataformas más consolidadas en Europa.

**El desafío:** Evaluar la solvencia de los solicitantes y predecir la probabilidad de default (incumplimiento de pago) antes de aprobar un préstamo.

### Definición del Problema

| Característica | Descripción |
|----------------|-------------|
| **Tipo de aprendizaje** | Supervisado |
| **Subproblema** | Clasificación Binaria |
| **Variable objetivo** | `DEFAULT` |
| **Valores** | `1` = Default (incumple), `0` = No Default (paga correctamente) |

### Creación de la Variable Objetivo

```python
df['DEFAULT'] = df['DefaultDate'].notna().astype(int)

## b. Diagrama de Flujo del Proyecto


