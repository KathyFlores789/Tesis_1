from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "datos_mantenimiento_quito.csv"
MODELS_DIR = ROOT_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


# ============================================================
# CARGA DE DATOS
# ============================================================

df = pd.read_csv(DATA_PATH)

df["fecha_mantenimiento"] = pd.to_datetime(
    df["fecha_mantenimiento"],
    errors="coerce"
)

df = df.sort_values("fecha_mantenimiento").reset_index(drop=True)


# ============================================================
# SOLO REGISTROS CON FALLA FUTURA
# ============================================================

df_reg = df[df["falla_futura_30d"] == 1].copy()

print("Registros totales:", len(df))
print("Registros con falla para regresión:", len(df_reg))


# ============================================================
# VARIABLES PREDICTORAS
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

X = df_reg[FEATURES].copy()
y = df_reg[TARGET].copy()


# ============================================================
# PARTICIÓN TEMPORAL
# ============================================================

n = len(df_reg)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# RANGOS DE COSTO DEFINIDOS EXCLUSIVAMENTE CON TRAIN
# ============================================================

Q1_COSTO = y_train.quantile(0.25)
Q3_COSTO = y_train.quantile(0.75)

print("\n" + "=" * 60)
print("RANGOS DE COSTO - DEFINIDOS SOLO CON TRAIN")
print("=" * 60)

print(f"Percentil 25 (Q1): ${Q1_COSTO:.2f}")
print(f"Percentil 75 (Q3): ${Q3_COSTO:.2f}")

print("\nCategorías:")
print(f"BAJO  : costo < ${Q1_COSTO:.2f}")
print(f"MEDIO : ${Q1_COSTO:.2f} <= costo <= ${Q3_COSTO:.2f}")
print(f"ALTO  : costo > ${Q3_COSTO:.2f}")


# ============================================================
# PREPROCESAMIENTO
# ============================================================

categorical_features = [
    "marca",
    "modelo",
    "tipo_vehiculo",
    "combustible",
    "transmision",
    "zona_operacion_quito",
    "tipo_mantenimiento_actual",
    "sistema_intervenido_actual",
    "repuesto_cambiado_actual"
]

numerical_features = [
    "anio",
    "cilindraje_l",
    "kilometraje_acumulado_km",
    "km_desde_ultimo_mantenimiento",
    "dias_desde_ultimo_mantenimiento",
    "temperatura_promedio_motor_c",
    "presion_aceite_kpa",
    "frenadas_bruscas_por_100km",
    "tiempo_ralenti_min_dia",
    "altitud_operacion_m",
    "tiene_codigo_obd_activo",
    "costo_mantenimiento_actual_usd"
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# MODELOS A COMPARAR
# ============================================================

models = {
    "LinearRegression": LinearRegression(),

    "Ridge": Ridge(
        alpha=1.0
    ),

    "ElasticNet": ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=5000,
        random_state=42
    ),

    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),

    "GradientBoostingRegressor": GradientBoostingRegressor(
        random_state=42
    )
}


# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN EN VALIDATION
# ============================================================

results = []

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Entrenamiento
    pipeline.fit(X_train, y_train)

    # Predicción sobre Validation
    y_pred = pipeline.predict(X_val)

    # Métricas
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    results.append({
        "modelo": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pipeline": pipeline
    })

    print("\n" + "=" * 60)
    print(f"MODELO: {name}")
    print("=" * 60)

    print("MAE  :", round(mae, 2))
    print("RMSE :", round(rmse, 2))
    print("R²   :", round(r2, 4))



# ============================================================
# SELECCIÓN DEL MEJOR MODELO
# Criterio principal: menor MAE
# Desempate: menor RMSE
# ============================================================

results_sorted = sorted(
    results,
    key=lambda x: (x["mae"], x["rmse"])
)

best_result = results_sorted[0]
best_model = best_result["pipeline"]

print("\n" + "=" * 60)
print("MEJOR REGRESOR EN VALIDATION")
print("=" * 60)

print("Modelo:", best_result["modelo"])
print("MAE :", round(best_result["mae"], 2))
print("RMSE:", round(best_result["rmse"], 2))
print("R²  :", round(best_result["r2"], 4))



# ============================================================
# EXPERIMENTO 2: REGRESIÓN CON TRANSFORMACIÓN LOGARÍTMICA
# DEL COSTO FUTURO
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENTO 2 - OBJETIVO LOGARÍTMICO")
print("=" * 60)

# Transformación logarítmica del objetivo
y_train_log = np.log1p(y_train)

log_results = []

for name, model in models.items():

    pipeline_log = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Entrenar utilizando log(costo + 1)
    pipeline_log.fit(X_train, y_train_log)

    # Predicción en escala logarítmica
    y_pred_log = pipeline_log.predict(X_val)

    # Regresar las predicciones a dólares
    y_pred_usd = np.expm1(y_pred_log)

    # Evitar costos negativos
    y_pred_usd = np.maximum(y_pred_usd, 0)

    # Métricas calculadas nuevamente en USD
    mae = mean_absolute_error(y_val, y_pred_usd)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_usd))
    r2 = r2_score(y_val, y_pred_usd)

    log_results.append({
        "modelo": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pipeline": pipeline_log
    })

    print("\n" + "-" * 60)
    print(f"MODELO: {name}")
    print("-" * 60)

    print("MAE  :", round(mae, 2))
    print("RMSE :", round(rmse, 2))
    print("R²   :", round(r2, 4))


# ============================================================
# MEJOR MODELO CON TRANSFORMACIÓN LOG
# ============================================================

log_results_sorted = sorted(
    log_results,
    key=lambda x: (x["mae"], x["rmse"])
)

best_log_result = log_results_sorted[0]
best_log_model = best_log_result["pipeline"]

print("\n" + "=" * 60)
print("MEJOR REGRESOR LOG EN VALIDATION")
print("=" * 60)

print("Modelo:", best_log_result["modelo"])
print("MAE :", round(best_log_result["mae"], 2))
print("RMSE:", round(best_log_result["rmse"], 2))
print("R²  :", round(best_log_result["r2"], 4))