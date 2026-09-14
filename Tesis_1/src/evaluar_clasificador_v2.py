from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
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
    / "classifier_v2.joblib"
)

REPORTS_DIR = ROOT_DIR / "reports"

REPORTS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

UMBRAL = 0.50

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
# CARGAR DATASET
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print("=" * 70)
print("EVALUACIÓN FORMAL DEL CLASIFICADOR V2")
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
# VARIABLES
# ============================================================

X = df[FEATURES].copy()

y = df[TARGET].copy()


# ============================================================
# MISMA PARTICIÓN TEMPORAL DEL ENTRENAMIENTO
#
# 1000 registros:
#
# Train       -> 0:699
# Validation  -> 699:850
# Test        -> 850:1000
#
# classifier_v2 fue posteriormente reentrenado con
# Train + Validation = 850 registros.
#
# Test permanece completamente separado.
# ============================================================

# ============================================================
# TEST ORIGINAL CON ETIQUETA REAL
#
# Los vehículos añadidos posteriormente desde la aplicación
# no poseen una etiqueta real de falla futura a 30 días.
# Por tanto, se excluyen de la evaluación supervisada.
# ============================================================

X_test = X.iloc[850:].copy()
y_test = y.iloc[850:].copy()

# Conservar únicamente registros cuya etiqueta real sea 0 o 1
mask_evaluable = y_test.isin([0, 1])

X_test = X_test.loc[mask_evaluable].copy()
y_test = y_test.loc[mask_evaluable].astype(int).copy()

print(
    "Registros evaluables de Test:",
    len(X_test)
)

print(
    "Positivos en Test:",
    int((y_test == 1).sum())
)

print(
    "Negativos en Test:",
    int((y_test == 0).sum())
)

print(
    "Registros de Test:",
    len(X_test)
)

print(
    "Positivos en Test:",
    int(y_test.sum())
)

print(
    "Negativos en Test:",
    int((y_test == 0).sum())
)


# ============================================================
# CARGAR MODELO DESPLEGADO
# ============================================================

modelo = joblib.load(
    MODEL_PATH
)

print(
    "\nModelo cargado desde:"
)

print(
    MODEL_PATH
)


# ============================================================
# PROBABILIDADES
# ============================================================

probabilidades = modelo.predict_proba(
    X_test
)[:, 1]


# ============================================================
# PREDICCIÓN CON EL MISMO UMBRAL DE STREAMLIT
# ============================================================

predicciones = (
    probabilidades >= UMBRAL
).astype(int)


# ============================================================
# MÉTRICAS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predicciones
)

precision = precision_score(
    y_test,
    predicciones,
    zero_division=0
)

recall = recall_score(
    y_test,
    predicciones,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predicciones,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probabilidades
)


# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "=" * 70)
print("RESULTADOS - TEST")
print("=" * 70)

print(
    f"\nUmbral     : {UMBRAL:.2f}"
)

print(
    f"Accuracy   : {accuracy:.4f}"
)

print(
    f"Precision  : {precision:.4f}"
)

print(
    f"Recall     : {recall:.4f}"
)

print(
    f"F1-score   : {f1:.4f}"
)

print(
    f"ROC-AUC    : {auc:.4f}"
)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

cm = confusion_matrix(
    y_test,
    predicciones
)

print(
    "\nMatriz de confusión:"
)

print(
    cm
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification report:"
)

print(
    classification_report(
        y_test,
        predicciones,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# GUARDAR MÉTRICAS
# ============================================================

metricas = pd.DataFrame([
    {
        "modelo": "classifier_v2",
        "dataset": "test",
        "n_test": len(X_test),
        "umbral": UMBRAL,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }
])

METRICS_PATH = (
    REPORTS_DIR
    / "metricas_clasificador_v2_test.csv"
)

metricas.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# GUARDAR PREDICCIONES INDIVIDUALES
# ============================================================

resultados_test = pd.DataFrame({
    "indice_original": X_test.index,
    "valor_real": y_test.values,
    "probabilidad_falla": probabilidades,
    "prediccion": predicciones
})

PREDICTIONS_PATH = (
    REPORTS_DIR
    / "predicciones_clasificador_v2_test.csv"
)

resultados_test.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# GUARDAR MATRIZ DE CONFUSIÓN COMO IMAGEN
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Sin falla",
        "Falla"
    ]
)

fig, ax = plt.subplots(
    figsize=(6, 5)
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "Matriz de confusión - classifier_v2"
)

plt.tight_layout()

CONFUSION_PATH = (
    REPORTS_DIR
    / "matriz_confusion_clasificador_v2.png"
)

plt.savefig(
    CONFUSION_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL
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
    "\nMatriz de confusión:"
)

print(
    CONFUSION_PATH
)

print("\n" + "=" * 70)
print("EVALUACIÓN FINALIZADA")
print("=" * 70)