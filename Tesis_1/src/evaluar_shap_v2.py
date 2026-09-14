from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


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
    / "classifier_v2.joblib"
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

TARGET = "falla_futura_30d"


# ============================================================
# CARGA DEL DATASET
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print("=" * 70)
print("EVALUACIÓN FORMAL DE EXPLICABILIDAD SHAP V2")
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
# DATASET DE TEST EVALUABLE
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()

X_test = X.iloc[850:].copy()
y_test = y.iloc[850:].copy()

# Excluir registros nuevos sin etiqueta real
mask_evaluable = y_test.isin([0, 1])

X_test = X_test.loc[mask_evaluable].copy()
y_test = y_test.loc[mask_evaluable].astype(int).copy()

print(
    "Registros evaluables de Test:",
    len(X_test)
)


# ============================================================
# CARGAR MODELO DESPLEGADO
# ============================================================

pipeline = joblib.load(
    MODEL_PATH
)

preprocessor = pipeline.named_steps["preprocessor"]
clasificador = pipeline.named_steps["model"]

print(
    "\nModelo cargado desde:"
)

print(
    MODEL_PATH
)


# ============================================================
# TRANSFORMAR TEST CON EL MISMO PREPROCESAMIENTO
# ============================================================

X_test_transformado = preprocessor.transform(
    X_test
)

if hasattr(
    X_test_transformado,
    "toarray"
):
    X_test_transformado = (
        X_test_transformado.toarray()
    )


# ============================================================
# NOMBRES DE VARIABLES TRANSFORMADAS
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)

feature_names_limpios = [
    nombre
    .replace("num__", "")
    .replace("cat__", "")
    for nombre in feature_names
]


# ============================================================
# BACKGROUND SHAP
# ============================================================

X_background = (
    X_test_transformado[:100]
)

explainer = shap.LinearExplainer(
    clasificador,
    X_background,
    feature_names=feature_names_limpios
)


# ============================================================
# CÁLCULO SHAP EN TEST
# ============================================================

shap_values = explainer(
    X_test_transformado
)

valores_shap = shap_values.values


# ============================================================
# IMPORTANCIA GLOBAL
# ============================================================

mean_abs_shap = np.abs(
    valores_shap
).mean(
    axis=0
)

df_importancia_global = pd.DataFrame({
    "feature": feature_names_limpios,
    "mean_abs_shap": mean_abs_shap
})

df_importancia_global = (
    df_importancia_global
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("TOP 15 VARIABLES MÁS INFLUYENTES - SHAP GLOBAL")
print("=" * 70)

print(
    df_importancia_global
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# GUARDAR IMPORTANCIA GLOBAL
# ============================================================

GLOBAL_CSV_PATH = (
    REPORTS_DIR
    / "shap_importancia_global_v2.csv"
)

df_importancia_global.to_csv(
    GLOBAL_CSV_PATH,
    index=False
)


# ============================================================
# GRÁFICO GLOBAL
# ============================================================

top_global = (
    df_importancia_global
    .head(15)
    .sort_values(
        "mean_abs_shap",
        ascending=True
    )
)

plt.figure(
    figsize=(9, 7)
)

plt.barh(
    top_global["feature"],
    top_global["mean_abs_shap"]
)

plt.xlabel(
    "Impacto SHAP medio absoluto"
)

plt.ylabel(
    "Variable"
)

plt.title(
    "Importancia global SHAP - classifier_v2"
)

plt.tight_layout()

GLOBAL_PNG_PATH = (
    REPORTS_DIR
    / "shap_importancia_global_v2.png"
)

plt.savefig(
    GLOBAL_PNG_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RESUMEN DE DIRECCIÓN DE IMPACTOS
# ============================================================

impacto_promedio = (
    valores_shap
    .mean(
        axis=0
    )
)

df_direccion = pd.DataFrame({
    "feature": feature_names_limpios,
    "shap_promedio": impacto_promedio,
    "mean_abs_shap": mean_abs_shap
})

df_direccion = (
    df_direccion
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)

DIRECTION_CSV_PATH = (
    REPORTS_DIR
    / "shap_direccion_global_v2.csv"
)

df_direccion.to_csv(
    DIRECTION_CSV_PATH,
    index=False
)


# ============================================================
# EXPLICACIONES LOCALES
# ============================================================

probabilidades = pipeline.predict_proba(
    X_test
)[:, 1]

predicciones = (
    probabilidades >= 0.50
).astype(int)

muestras_locales = []

n_muestras = min(
    10,
    len(X_test)
)

for i in range(
    n_muestras
):

    fila_shap = (
        valores_shap[i]
    )

    indices_ordenados = np.argsort(
        np.abs(
            fila_shap
        )
    )[::-1]

    top_factores = []

    for idx in indices_ordenados[:5]:

        top_factores.append({
            "factor": feature_names_limpios[idx],
            "shap": float(
                fila_shap[idx]
            )
        })

    muestras_locales.append({
        "indice_test": int(
            X_test.index[i]
        ),
        "valor_real": int(
            y_test.iloc[i]
        ),
        "prediccion": int(
            predicciones[i]
        ),
        "probabilidad_falla": float(
            probabilidades[i]
        ),
        "factor_1": top_factores[0]["factor"],
        "shap_1": top_factores[0]["shap"],
        "factor_2": top_factores[1]["factor"],
        "shap_2": top_factores[1]["shap"],
        "factor_3": top_factores[2]["factor"],
        "shap_3": top_factores[2]["shap"],
        "factor_4": top_factores[3]["factor"],
        "shap_4": top_factores[3]["shap"],
        "factor_5": top_factores[4]["factor"],
        "shap_5": top_factores[4]["shap"]
    })


df_locales = pd.DataFrame(
    muestras_locales
)

LOCAL_CSV_PATH = (
    REPORTS_DIR
    / "shap_explicaciones_locales_v2.csv"
)

df_locales.to_csv(
    LOCAL_CSV_PATH,
    index=False
)


# ============================================================
# CONSISTENCIA BÁSICA DE EXPLICACIONES
# ============================================================

# Verifica que la suma de SHAP + valor base reconstruya
# aproximadamente la salida del modelo en log-odds.

base_values = (
    shap_values.base_values
)

reconstruccion = (
    base_values
    + valores_shap.sum(
        axis=1
    )
)

log_odds_modelo = (
    clasificador.decision_function(
        X_test_transformado
    )
)

diferencia = np.abs(
    reconstruccion
    - log_odds_modelo
)

error_reconstruccion_medio = (
    diferencia.mean()
)

error_reconstruccion_max = (
    diferencia.max()
)

print("\n" + "=" * 70)
print("CONSISTENCIA DE LA EXPLICACIÓN")
print("=" * 70)

print(
    f"\nError medio de reconstrucción: "
    f"{error_reconstruccion_medio:.10f}"
)

print(
    f"Error máximo de reconstrucción: "
    f"{error_reconstruccion_max:.10f}"
)


# ============================================================
# GUARDAR MÉTRICAS DE CONSISTENCIA
# ============================================================

metricas_shap = pd.DataFrame([
    {
        "modelo": "classifier_v2",
        "dataset": "test",
        "n_test": len(X_test),
        "n_features_transformadas": len(
            feature_names_limpios
        ),
        "error_reconstruccion_medio": (
            error_reconstruccion_medio
        ),
        "error_reconstruccion_max": (
            error_reconstruccion_max
        )
    }
])

METRICS_PATH = (
    REPORTS_DIR
    / "metricas_shap_v2.csv"
)

metricas_shap.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# ARCHIVOS GENERADOS
# ============================================================

print("\n" + "=" * 70)
print("ARCHIVOS GENERADOS")
print("=" * 70)

print(
    "\nImportancia global CSV:"
)

print(
    GLOBAL_CSV_PATH
)

print(
    "\nImportancia global PNG:"
)

print(
    GLOBAL_PNG_PATH
)

print(
    "\nDirección global:"
)

print(
    DIRECTION_CSV_PATH
)

print(
    "\nExplicaciones locales:"
)

print(
    LOCAL_CSV_PATH
)

print(
    "\nMétricas SHAP:"
)

print(
    METRICS_PATH
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("EVALUACIÓN SHAP FINALIZADA")
print("=" * 70)