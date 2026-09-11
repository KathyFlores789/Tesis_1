from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
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

df_cost = df[df["falla_futura_30d"] == 1].copy()

print("Registros totales:", len(df))
print("Registros con falla para costo:", len(df_cost))


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

X = df_cost[FEATURES].copy()

# El costo real se usa únicamente para construir la clase objetivo
y_cost = df_cost["costo_falla_futura_usd"].copy()


# ============================================================
# PARTICIÓN TEMPORAL
# ============================================================

n = len(df_cost)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_cost_train = y_cost.iloc[:train_end]
y_cost_val = y_cost.iloc[train_end:val_end]
y_cost_test = y_cost.iloc[val_end:]

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# RANGOS DEFINIDOS SOLO CON TRAIN
# ============================================================

Q1_COSTO = y_cost_train.quantile(0.25)
Q3_COSTO = y_cost_train.quantile(0.75)

print("\n" + "=" * 60)
print("RANGOS DE COSTO")
print("=" * 60)

print(f"Q1: ${Q1_COSTO:.2f}")
print(f"Q3: ${Q3_COSTO:.2f}")


# ============================================================
# FUNCIÓN PARA CREAR LAS CATEGORÍAS
# ============================================================

def categorizar_costo(costo):
    if costo < Q1_COSTO:
        return "BAJO"
    elif costo <= Q3_COSTO:
        return "MEDIO"
    else:
        return "ALTO"


y_train = y_cost_train.apply(categorizar_costo)
y_val = y_cost_val.apply(categorizar_costo)
y_test = y_cost_test.apply(categorizar_costo)

print("\nDistribución Train:")
print(y_train.value_counts())

print("\nDistribución Validation:")
print(y_val.value_counts())

print("\nDistribución Test:")
print(y_test.value_counts())


# ============================================================
# IDENTIFICAR VARIABLES NUMÉRICAS Y CATEGÓRICAS
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

print("\nVariables numéricas:")
print(numeric_features)

print("\nVariables categóricas:")
print(categorical_features)


# ============================================================
# PREPROCESAMIENTO
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# MODELOS A COMPARAR
# ============================================================

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingClassifier(
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

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)

    precision = precision_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    results.append({
        "modelo": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pipeline": pipeline
    })

    print("\n" + "=" * 60)
    print("MODELO:", name)
    print("=" * 60)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Macro :", round(f1, 4))

    print("\nMatriz de confusión:")
    print(
        confusion_matrix(
            y_val,
            y_pred,
            labels=["BAJO", "MEDIO", "ALTO"]
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            y_pred,
            labels=["BAJO", "MEDIO", "ALTO"],
            zero_division=0
        )
    )


# ============================================================
# SELECCIÓN DEL MEJOR MODELO EN VALIDATION
# ============================================================

results_sorted = sorted(
    results,
    key=lambda x: x["f1"],
    reverse=True
)

best_result = results_sorted[0]
best_model = best_result["pipeline"]

print("\n" + "=" * 60)
print("MEJOR MODELO DE CATEGORÍA DE COSTO - VALIDATION")
print("=" * 60)

print("Modelo   :", best_result["modelo"])
print("Accuracy :", round(best_result["accuracy"], 4))
print("Precision:", round(best_result["precision"], 4))
print("Recall   :", round(best_result["recall"], 4))
print("F1 Macro :", round(best_result["f1"], 4))


# ============================================================
# EVALUACIÓN FINAL EN TEST
# ============================================================

y_test_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)

test_precision = precision_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 60)
print("EVALUACIÓN FINAL - TEST")
print("=" * 60)

print("Modelo   :", best_result["modelo"])
print("Accuracy :", round(test_accuracy, 4))
print("Precision:", round(test_precision, 4))
print("Recall   :", round(test_recall, 4))
print("F1 Macro :", round(test_f1, 4))

print("\nMatriz de confusión:")
print(
    confusion_matrix(
        y_test,
        y_test_pred,
        labels=["BAJO", "MEDIO", "ALTO"]
    )
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        labels=["BAJO", "MEDIO", "ALTO"],
        zero_division=0
    )
)