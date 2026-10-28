
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "datos_mantenimiento_quito.csv"
MODEL_PATH = ROOT / "models" / "regressor_v2.joblib"

# Cargar registros y modelo existente
df = pd.read_csv(DATA_PATH)
modelo = joblib.load(MODEL_PATH)

# Ordenar temporalmente
df["fecha_mantenimiento"] = pd.to_datetime(
    df["fecha_mantenimiento"],
    errors="coerce"
)
df = df.sort_values("fecha_mantenimiento").reset_index(drop=True)

# El modelo de costos se entrenó únicamente con casos de falla
df_reg = df[df["falla_futura_30d"] == 1].copy()

FEATURES = [
    "marca", "modelo", "anio", "tipo_vehiculo",
    "combustible", "cilindraje_l", "transmision",
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

n = len(df_reg)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Misma partición que train_regressor_v2.py
y_train_final = df_reg[TARGET].iloc[:val_end]
y_test = df_reg[TARGET].iloc[val_end:]

X_test = df_reg[FEATURES].iloc[val_end:]

# Baseline: predecir siempre la mediana de Train + Validation
mediana = y_train_final.median()
pred_baseline = np.full(len(y_test), mediana)

mae_baseline = mean_absolute_error(y_test, pred_baseline)

# Modelo final: salida logarítmica convertida a USD
pred_log = modelo.predict(X_test)
pred_modelo = np.maximum(np.expm1(pred_log), 0)

mae_modelo = mean_absolute_error(y_test, pred_modelo)

# Identificar algoritmo
algoritmo = type(modelo.named_steps["model"]).__name__

print("\nCOMPARACIÓN DE ESTIMACIÓN DE COSTOS")
print("Algoritmo:", algoritmo)
print("Registros de prueba:", len(y_test))
print(f"Mediana de entrenamiento: USD {mediana:.2f}")
print(f"MAE del modelo: USD {mae_modelo:.2f}")
print(f"MAE del baseline: USD {mae_baseline:.2f}")
print(f"Diferencia de MAE: USD {mae_baseline - mae_modelo:.2f}")
