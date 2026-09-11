"""
src/audit_and_freeze_classifier.py
----------------------------------
Audita formalmente el clasificador de Falla Futura a 30 días (Etapa A),
imprime la tabla de métricas en Validation y persiste el artefacto
congelado en `models/best_classifier.joblib`.
"""

import sys
import logging
from pathlib import Path
import warnings
import joblib

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, brier_score_loss

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import load_dataset
from src.preprocessing import aplicar_limpieza_base
from src.features import construir_features_t0
from src.split_data import ejecutar_particion_temporal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Definición explícita de las 25 variables T0
FEATURES_T0_CLASSIFICATION = [
    'marca', 'modelo', 'anio', 'tipo_vehiculo', 'combustible', 
    'transmision', 'zona_operacion_quito', 'tiene_codigo_obd_activo', 
    'tipo_mantenimiento_actual', 'sistema_intervenido_actual', 'repuesto_cambiado_actual',
    'cilindraje_l', 'kilometraje_acumulado_km', 'km_desde_ultimo_mantenimiento', 
    'dias_desde_ultimo_mantenimiento', 'temperatura_promedio_motor_c', 'presion_aceite_kpa', 
    'frenadas_bruscas_por_100km', 'tiempo_ralenti_min_dia', 'altitud_operacion_m', 
    'costo_mantenimiento_actual_usd', 'edad_vehiculo_años', 'km_promedio_estimado_diario', 
    'indice_estres_termico_altitud', 'ratio_recorrido_reciente'
]

def crear_preprocesador(df_example):
    num_cols = df_example.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_example.select_dtypes(exclude=[np.number]).columns.tolist()

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    return preprocessor

def auditar_y_congelar_clasificador():
    print("=" * 80)
    print("ETAPA A — AUDITORÍA Y CONGELAMIENTO DEL CLASIFICADOR (models/best_classifier.joblib)")
    print("=" * 80)

    # 1. Carga de Datos y Partición
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    X_train_g, X_val_g, X_test_g, y_train_g, y_val_g, y_test_g = ejecutar_particion_temporal(df_full)

    df_train = df_full.loc[X_train_g.index].copy()
    df_val = df_full.loc[X_val_g.index].copy()

    X_train = df_train[FEATURES_T0_CLASSIFICATION].copy()
    y_train = df_train["falla_futura_30d"].astype(int).copy()

    X_val = df_val[FEATURES_T0_CLASSIFICATION].copy()
    y_val = df_val["falla_futura_30d"].astype(int).copy()

    logging.info(f"Train Set: {len(X_train)} registros (Tasa Falla: {y_train.mean():.2%})")
    logging.info(f"Val Set  : {len(X_val)} registros (Tasa Falla: {y_val.mean():.2%})")

    preprocessor = crear_preprocesador(X_train)

    # 2. Definición de Candidatos
    candidatos = {
        "LogisticRegression": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
    }

    # Intentar agregar XGBoost si está disponible
    try:
        from xgboost import XGBClassifier
        candidatos["XGBoostClassifier"] = XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
    except ImportError:
        pass

    resultados = []
    best_model_name = None
    best_auc = -1.0
    best_pipeline = None

    print("\n--------------------------------------------------------------------------------")
    print("EVALUACIÓN DE CANDIDATOS EN VALIDATION SET")
    print("--------------------------------------------------------------------------------")

    for name, model in candidatos.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        
        # Entrenar EXCLUSIVAMENTE en Train
        pipeline.fit(X_train, y_train)

        # Predicciones en Validation
        y_prob_val = pipeline.predict_proba(X_val)[:, 1]
        y_pred_val = (y_prob_val >= 0.5).astype(int)

        auc = roc_auc_score(y_val, y_prob_val)
        prec = precision_score(y_val, y_pred_val, zero_division=0)
        rec = recall_score(y_val, y_pred_val, zero_division=0)
        f1 = f1_score(y_val, y_pred_val, zero_division=0)
        brier = brier_score_loss(y_val, y_prob_val)

        resultados.append({
            "Modelo": name,
            "ROC-AUC": auc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Brier Score": brier
        })

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_pipeline = pipeline

    df_res = pd.DataFrame(resultados).sort_values(by="ROC-AUC", ascending=False)
    print(df_res.to_string(index=False))

    print(f"\n---> Modelo seleccionado para producción/tesis: {best_model_name} (ROC-AUC = {best_auc:.4f})")

    # 3. Guardar el artefacto en models/
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "best_classifier.joblib"

    joblib.dump(best_pipeline, out_path)
    print(f"\n[OK] Clasificador congelado guardado exitosamente en: {out_path}")
    print("=" * 80)

if __name__ == "__main__":
    auditar_y_congelar_clasificador()