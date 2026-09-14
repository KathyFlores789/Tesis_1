from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "datos_mantenimiento_quito.csv"
)

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "regressor_v2.joblib"
)

REPORTS_DIR = ROOT_DIR / "reports"

REPORTS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

FEATURES = [
    "marca",
    "modelo",
    "anio",
    "tipo_vehiculo",
    "combustible",
    "cilindraje_l",
    "transmision",
    "kilometraje_acumulado_km",
    "km_desde_ultimo_mantenimiento",
    "dias_desde_ultimo_mantenimiento",
    "temperatura_promedio_motor_c",
    "presion_aceite_kpa",
    "frenadas_bruscas_por_100km",
    "tiempo_ralenti_min_dia",
    "altitud_operacion_m",
    "zona_operacion_quito",
    "tiene_codigo_obd_activo",
    "tipo_mantenimiento_actual",
    "sistema_intervenido_actual",
    "repuesto_cambiado_actual",
    "costo_mantenimiento_actual_usd"
]

TARGET = "costo_falla_futura_usd"


# ============================================================
# CARGA DEL DATASET
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print("=" * 70)
print("EVALUACIÓN FORMAL DEL REGRESOR V2")
print("=" * 70)

print(
    "\nRegistros totales:",
    len(df)
)


# ============================================================
# ORDEN TEMPORAL
# ============================================================

df["fecha_mantenimiento"] = pd.to_datetime(
    df["fecha_mantenimiento"],
    errors="coerce"
)

df = (
    df
    .sort_values("fecha_mantenimiento")
    .reset_index(drop=True)
)


# ============================================================
# REGISTROS EVALUABLES PARA COSTO
#
# Solo se pueden evaluar registros donde:
# - hubo falla futura real
# - existe costo futuro real conocido
#
# Los vehículos nuevos agregados desde Streamlit no tienen
# aún esas etiquetas reales, por lo que se excluyen.
# ============================================================

df_reg = df[
    (df["falla_futura_30d"] == 1)
    & (df[TARGET].notna())
].copy()

print(
    "Registros con falla y costo conocido:",
    len(df_reg)
)


# ============================================================
# VARIABLES
# ============================================================

X = df_reg[FEATURES].copy()
y = df_reg[TARGET].copy()


# ============================================================
# PARTICIÓN TEMPORAL
#
# Misma lógica usada durante el entrenamiento:
#
# 70 % Train
# 15 % Validation
# 15 % Test
#
# El modelo final fue entrenado con Train + Validation.
# Test se mantiene separado para evaluación final.
# ============================================================

n = len(df_reg)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end].copy()
y_train = y.iloc[:train_end].copy()

X_val = X.iloc[train_end:val_end].copy()
y_val = y.iloc[train_end:val_end].copy()

X_test = X.iloc[val_end:].copy()
y_test = y.iloc[val_end:].copy()

print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# CARGAR MODELO FINAL
# ============================================================

regresor = joblib.load(
    MODEL_PATH
)

print(
    "\nModelo cargado desde:"
)

print(
    MODEL_PATH
)


# ============================================================
# PREDICCIÓN
#
# El modelo fue entrenado sobre:
# log1p(costo)
#
# Por eso la salida debe convertirse a USD con expm1().
# ============================================================

pred_log = regresor.predict(
    X_test
)

pred_usd = np.expm1(
    pred_log
)

# Evitar valores negativos
pred_usd = np.maximum(
    pred_usd,
    0
)


# ============================================================
# MÉTRICAS
# ============================================================

mae = mean_absolute_error(
    y_test,
    pred_usd
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred_usd
    )
)

r2 = r2_score(
    y_test,
    pred_usd
)


# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "=" * 70)
print("RESULTADOS - TEST")
print("=" * 70)

print(
    f"\nMAE  : {mae:.2f}"
)

print(
    f"RMSE : {rmse:.2f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# ERROR ABSOLUTO POR REGISTRO
# ============================================================

errores_absolutos = np.abs(
    y_test.values - pred_usd
)

print(
    "\nError absoluto medio:",
    round(
        errores_absolutos.mean(),
        2
    )
)

print(
    "Error absoluto mediano:",
    round(
        np.median(errores_absolutos),
        2
    )
)


# ============================================================
# GUARDAR MÉTRICAS
# ============================================================

metricas = pd.DataFrame([
    {
        "modelo": "regressor_v2",
        "dataset": "test",
        "n_test": len(X_test),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "error_absoluto_mediano": np.median(
            errores_absolutos
        )
    }
])

METRICS_PATH = (
    REPORTS_DIR
    / "metricas_regresor_v2_test.csv"
)

metricas.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# GUARDAR PREDICCIONES
# ============================================================

resultados_test = pd.DataFrame({
    "indice_original": X_test.index,
    "costo_real_usd": y_test.values,
    "costo_estimado_usd": pred_usd,
    "error_absoluto_usd": errores_absolutos
})

PREDICTIONS_PATH = (
    REPORTS_DIR
    / "predicciones_regresor_v2_test.csv"
)

resultados_test.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# GRÁFICO REAL VS ESTIMADO
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.scatter(
    y_test.values,
    pred_usd
)

min_val = min(
    y_test.min(),
    pred_usd.min()
)

max_val = max(
    y_test.max(),
    pred_usd.max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel(
    "Costo real (USD)"
)

plt.ylabel(
    "Costo estimado (USD)"
)

plt.title(
    "Costo real vs costo estimado - regressor_v2"
)

plt.tight_layout()

SCATTER_PATH = (
    REPORTS_DIR
    / "real_vs_estimado_regresor_v2.png"
)

plt.savefig(
    SCATTER_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRÁFICO DE ERRORES
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.hist(
    errores_absolutos,
    bins=15
)

plt.xlabel(
    "Error absoluto (USD)"
)

plt.ylabel(
    "Frecuencia"
)

plt.title(
    "Distribución del error absoluto - regressor_v2"
)

plt.tight_layout()

ERRORS_PATH = (
    REPORTS_DIR
    / "errores_regresor_v2.png"
)

plt.savefig(
    ERRORS_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ARCHIVOS GENERADOS
# ============================================================

print("\n" + "=" * 70)
print("ARCHIVOS GENERADOS")
print("=" * 70)

print(
    "\nMétricas:"
)

print(
    METRICS_PATH
)

print(
    "\nPredicciones:"
)

print(
    PREDICTIONS_PATH
)

print(
    "\nReal vs estimado:"
)

print(
    SCATTER_PATH
)

print(
    "\nDistribución de errores:"
)

print(
    ERRORS_PATH
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("EVALUACIÓN FINALIZADA")
print("=" * 70)