from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

ROOT_DIR = Path(__file__).resolve().parents[1]

HISTORICO_PATH = ROOT_DIR / "data" / "raw" / "datos_mantenimiento_quito.csv"
SEGUIMIENTO_PATH = ROOT_DIR / "data" / "seguimiento" / "vehiculos_seguimiento.csv"

MODELO_ACTUAL_PATH = ROOT_DIR / "models" / "classifier_v2.joblib"
CANDIDATOS_DIR = ROOT_DIR / "models" / "candidatos"
REPORTES_DIR = ROOT_DIR / "reports" / "reentrenamiento"

MODELO_CANDIDATO_PATH = CANDIDATOS_DIR / "classifier_candidato_prueba.joblib"
METRICAS_PATH = REPORTES_DIR / "comparacion_classifier_actual_vs_candidato.csv"
METADATA_PATH = REPORTES_DIR / "metadata_classifier_candidato_prueba.json"

UMBRAL = 0.50

FEATURES = [
    "marca", "modelo", "anio", "tipo_vehiculo", "combustible",
    "cilindraje_l", "transmision", "kilometraje_acumulado_km",
    "km_desde_ultimo_mantenimiento", "dias_desde_ultimo_mantenimiento",
    "temperatura_promedio_motor_c", "presion_aceite_kpa",
    "frenadas_bruscas_por_100km", "tiempo_ralenti_min_dia",
    "altitud_operacion_m", "zona_operacion_quito",
    "tiene_codigo_obd_activo", "tipo_mantenimiento_actual",
    "sistema_intervenido_actual", "repuesto_cambiado_actual",
    "costo_mantenimiento_actual_usd"
]

TARGET = "falla_futura_30d"

CATEGORICAL_FEATURES = [
    "marca", "modelo", "tipo_vehiculo", "combustible", "transmision",
    "zona_operacion_quito", "tipo_mantenimiento_actual",
    "sistema_intervenido_actual", "repuesto_cambiado_actual"
]

NUMERICAL_FEATURES = [
    "anio", "cilindraje_l", "kilometraje_acumulado_km",
    "km_desde_ultimo_mantenimiento", "dias_desde_ultimo_mantenimiento",
    "temperatura_promedio_motor_c", "presion_aceite_kpa",
    "frenadas_bruscas_por_100km", "tiempo_ralenti_min_dia",
    "altitud_operacion_m", "tiene_codigo_obd_activo",
    "costo_mantenimiento_actual_usd"
]


def metricas(modelo, X, y):
    prob = modelo.predict_proba(X)[:, 1]
    pred = (prob >= UMBRAL).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])

    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def main():
    print("=" * 76)
    print("REENTRENAMIENTO CONTROLADO - CLASSIFIER CANDIDATO (PRUEBA)")
    print("=" * 76)
    print("IMPORTANTE: los registros nuevos actuales son ficticios de prueba.")
    print("Este script NO reemplaza classifier_v2.joblib.\n")

    for ruta in [HISTORICO_PATH, SEGUIMIENTO_PATH, MODELO_ACTUAL_PATH]:
        if not ruta.is_file():
            raise FileNotFoundError(f"No se encontró: {ruta}")

    historico = pd.read_csv(HISTORICO_PATH)
    seguimiento = pd.read_csv(SEGUIMIENTO_PATH)

    # Mantener exactamente la referencia temporal del entrenamiento/evaluación V2:
    # 0:850 = Train + Validation original; 850:1000 = Test original intocable.
    historico["fecha_mantenimiento"] = pd.to_datetime(
        historico["fecha_mantenimiento"], errors="coerce"
    )
    historico = historico.sort_values("fecha_mantenimiento").reset_index(drop=True)

    if len(historico) != 1000:
        raise ValueError(
            f"Se esperaban 1000 históricos para reproducir el split V2; hay {len(historico)}."
        )

    base_train_original = historico.iloc[:850].copy()
    test_original = historico.iloc[850:1000].copy()

    estado = seguimiento["estado_seguimiento"].fillna("").astype(str).str.strip().str.casefold()
    nuevos_validados = seguimiento.loc[estado.eq("validado")].copy()

    # Quality Gate mínimo para los nuevos datos.
    y_nuevos = pd.to_numeric(nuevos_validados[TARGET], errors="coerce")
    nuevos_validados = nuevos_validados.loc[y_nuevos.isin([0, 1])].copy()
    nuevos_validados[TARGET] = pd.to_numeric(
        nuevos_validados[TARGET], errors="coerce"
    ).astype(int)

    faltantes = [c for c in FEATURES + [TARGET] if c not in nuevos_validados.columns]
    if faltantes:
        raise ValueError("Faltan columnas en seguimiento: " + ", ".join(faltantes))

    ids_test = set(test_original["vehiculo_id"].astype(str))
    ids_nuevos = set(nuevos_validados["vehiculo_id"].astype(str))
    solapados = ids_test.intersection(ids_nuevos)
    if solapados:
        raise ValueError(
            "Hay IDs nuevos dentro del Test original: " + ", ".join(sorted(solapados))
        )

    # El candidato aprende de los 850 casos originales de desarrollo
    # + únicamente los nuevos casos cuyo resultado real ya fue validado.
    train_candidato = pd.concat(
        [base_train_original, nuevos_validados],
        ignore_index=True
    )

    X_train = train_candidato[FEATURES].copy()
    y_train = pd.to_numeric(train_candidato[TARGET], errors="raise").astype(int)

    X_test = test_original[FEATURES].copy()
    y_test = pd.to_numeric(test_original[TARGET], errors="raise").astype(int)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])

    candidato = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ])

    print(f"Train original V2:          {len(base_train_original)}")
    print(f"Nuevos validados añadidos: {len(nuevos_validados)}")
    print(f"Train candidato:            {len(train_candidato)}")
    print(f"Test original preservado:   {len(test_original)}")

    candidato.fit(X_train, y_train)

    actual = joblib.load(MODELO_ACTUAL_PATH)

    m_actual = metricas(actual, X_test, y_test)
    m_candidato = metricas(candidato, X_test, y_test)

    comparacion = pd.DataFrame([
        {"modelo": "classifier_v2_actual", **m_actual},
        {"modelo": "classifier_candidato_prueba", **m_candidato},
    ])

    CANDIDATOS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTES_DIR.mkdir(parents=True, exist_ok=True)

    # Se guarda APARTE. Nunca se sobrescribe el modelo desplegado.
    joblib.dump(candidato, MODELO_CANDIDATO_PATH)
    comparacion.to_csv(METRICAS_PATH, index=False)

    metadata = {
        "tipo": "classifier_candidato_prueba",
        "uso": "prueba_pipeline_reentrenamiento_no_promover",
        "umbral_evaluacion": UMBRAL,
        "historicos_desarrollo": len(base_train_original),
        "nuevos_validados": len(nuevos_validados),
        "total_entrenamiento_candidato": len(train_candidato),
        "test_original_preservado": len(test_original),
        "nota": (
            "Los nuevos registros usados en esta ejecución son ficticios. "
            "El modelo candidato no debe sustituir al modelo desplegado."
        ),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n" + "=" * 76)
    print("COMPARACIÓN SOBRE EL MISMO TEST ORIGINAL")
    print("=" * 76)
    print(comparacion.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    delta_f1 = m_candidato["f1"] - m_actual["f1"]
    delta_auc = m_candidato["roc_auc"] - m_actual["roc_auc"]

    print("\nDiferencia candidato - actual:")
    print(f"F1      : {delta_f1:+.4f}")
    print(f"ROC-AUC : {delta_auc:+.4f}")

    print("\n" + "=" * 76)
    print("RESULTADO")
    print("=" * 76)
    print("Modelo actual: NO MODIFICADO")
    print("App V10:       NO MODIFICADA")
    print("Candidato:     guardado únicamente para prueba")
    print(f"\nModelo candidato:\n{MODELO_CANDIDATO_PATH}")
    print(f"\nComparación:\n{METRICAS_PATH}")
    print(f"\nMetadata:\n{METADATA_PATH}")
    print("\nNO PROMOVER ESTE CANDIDATO: contiene registros ficticios de prueba.")


if __name__ == "__main__":
    main()
