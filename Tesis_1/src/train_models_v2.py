from pathlib import Path

import pandas as pd
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
    roc_auc_score,
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

TARGET = "falla_futura_30d"

X = df[FEATURES].copy()
y = df[TARGET].copy()


# ============================================================
# PARTICIÓN TEMPORAL
# ============================================================

X_train = X.iloc[:699]
y_train = y.iloc[:699]

X_val = X.iloc[699:850]
y_val = y.iloc[699:850]

X_test = X.iloc[850:]
y_test = y.iloc[850:]

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

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
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
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

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_val)[:, 1]
    else:
        y_prob = None

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    if y_prob is not None:
        auc = roc_auc_score(y_val, y_prob)
    else:
        auc = None

    results.append({
        "modelo": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "pipeline": pipeline
    })

    print("\n" + "=" * 60)
    print(f"MODELO: {name}")
    print("=" * 60)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1       :", round(f1, 4))

    if auc is not None:
        print("AUC      :", round(auc, 4))

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_val, y_pred))

    print("\nClassification report:")
    print(classification_report(y_val, y_pred, zero_division=0))



# ============================================================
# SELECCIÓN DEL MEJOR MODELO
# ============================================================

results_sorted = sorted(
    results,
    key=lambda x: (x["f1"], x["auc"] if x["auc"] is not None else 0),
    reverse=True
)

best_result = results_sorted[0]
best_model = best_result["pipeline"]

print("\n" + "=" * 60)
print("MEJOR MODELO EN VALIDATION")
print("=" * 60)

print("Modelo:", best_result["modelo"])
print("F1:", round(best_result["f1"], 4))

if best_result["auc"] is not None:
    print("AUC:", round(best_result["auc"], 4))



# ============================================================
# EVALUACIÓN FINAL EN TEST
# ============================================================

y_test_pred = best_model.predict(X_test)
y_test_prob = best_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_auc = roc_auc_score(y_test, y_test_prob)

print("\n" + "=" * 60)
print("EVALUACIÓN FINAL - TEST")
print("=" * 60)

print("Modelo    :", best_result["modelo"])
print("Accuracy  :", round(test_accuracy, 4))
print("Precision :", round(test_precision, 4))
print("Recall    :", round(test_recall, 4))
print("F1        :", round(test_f1, 4))
print("AUC       :", round(test_auc, 4))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification report:")
print(classification_report(y_test, y_test_pred, zero_division=0))



# ============================================================
# AJUSTE DEL UMBRAL USANDO VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("AJUSTE DE UMBRAL - VALIDATION")
print("=" * 60)

# Probabilidades del mejor modelo sobre Validation
y_val_prob = best_model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:

    y_val_pred_threshold = (y_val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val,
        y_val_pred_threshold,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred_threshold,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_pred_threshold,
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(
        f"Umbral: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )


# ============================================================
# SELECCIONAR UMBRAL CON MAYOR F1
# ============================================================

best_threshold_result = max(
    threshold_results,
    key=lambda x: x["f1"]
)

BEST_THRESHOLD = best_threshold_result["threshold"]

print("\n" + "=" * 60)
print("MEJOR UMBRAL EN VALIDATION")
print("=" * 60)

print("Umbral seleccionado:", BEST_THRESHOLD)
print(
    "Precision:",
    round(best_threshold_result["precision"], 4)
)
print(
    "Recall:",
    round(best_threshold_result["recall"], 4)
)
print(
    "F1:",
    round(best_threshold_result["f1"], 4)
)


# ============================================================
# MODELO FINAL PARA PRODUCCIÓN
# Reentrenamiento con Train + Validation
# ============================================================

print("\n" + "=" * 60)
print("ENTRENAMIENTO DEL MODELO FINAL")
print("=" * 60)

X_train_final = pd.concat(
    [X_train, X_val],
    axis=0
)

y_train_final = pd.concat(
    [y_train, y_val],
    axis=0
)

final_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

final_model.fit(
    X_train_final,
    y_train_final
)

FINAL_MODEL_PATH = MODELS_DIR / "classifier_v2.joblib"

joblib.dump(
    final_model,
    FINAL_MODEL_PATH
)

print("Modelo final entrenado con:", len(X_train_final), "registros")
print("Modelo guardado en:")
print(FINAL_MODEL_PATH)
print("Umbral definitivo:", BEST_THRESHOLD)