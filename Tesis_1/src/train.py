"""
src/train.py
------------
Entrenamiento y evaluación de baselines (Random Forest y XGBoost)
utilizando Pipelines independientes, ajuste exclusivo sobre Train y 
validación temporal sobre Validation.

¡El conjunto Test se mantiene estrictamente reservado!
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Modelos y Pipelines
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Métricas
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Configuración de rutas del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import load_dataset
from src.preprocessing import aplicar_limpieza_base
from src.features import construir_features_t0
from src.split_data import ejecutar_particion_temporal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def crear_preprocesador_num_cat(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Identifica automáticamente columnas numéricas y categóricas para construir
    un ColumnTransformer independiente.
    
    Evita Data Leakage: Las estadísticas (media, desviación, categorías) se calculan 
    únicamente al hacer .fit() sobre X_train.
    """
    num_cols = X_train.select_dtypes(include=[np.number, 'int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Pipeline para variables numéricas: Imputación por mediana + Escalado
    pipeline_num = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para variables categóricas: Imputación + OneHotEncoding
    pipeline_cat = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', pipeline_num, num_cols),
            ('cat', pipeline_cat, cat_cols)
        ],
        remainder='drop'
    )

    return preprocessor


def construir_pipelines(X_train: pd.DataFrame):
    """
    Instancia los dos Pipelines independientes: RandomForest y XGBoost.
    """
    preproc_rf = crear_preprocesador_num_cat(X_train)
    preproc_xgb = crear_preprocesador_num_cat(X_train)

    # 1. Pipeline Random Forest Baseline
    pipeline_rf = Pipeline([
        ('preprocessor', preproc_rf),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 2. Pipeline XGBoost Baseline
    pipeline_xgb = Pipeline([
        ('preprocessor', preproc_xgb),
        ('classifier', XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
        ))
    ])

    return {'Random Forest': pipeline_rf, 'XGBoost': pipeline_xgb}


def evaluar_modelo(model_name: str, pipeline: Pipeline, X_train, y_train, X_val, y_val) -> dict:
    """
    Ajusta el pipeline únicamente sobre Train y evalúa métricas sobre Validation.
    """
    logging.info(f"Entrenando Pipeline: {model_name}...")
    
    # Fit SOLO con Train
    pipeline.fit(X_train, y_train)

    # Predicciones SOLO en Validation
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    # Cálculo de métricas sobre Validation
    metricas = {
        'Modelo': model_name,
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred, zero_division=0),
        'Recall': recall_score(y_val, y_pred, zero_division=0),
        'F1-Score': f1_score(y_val, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_val, y_proba)
    }

    return metricas


def ejecutar_entrenamiento_baseline():
    # 1. Carga, Limpieza e Ingeniería de Features T0
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    # 2. Partición Temporal Estricta (Reserva Test)
    X_train, X_val, X_test, y_train, y_val, y_test = ejecutar_particion_temporal(df_full)

    # 3. Construcción de Pipelines Independientes
    pipelines = construir_pipelines(X_train)

    # 4. Entrenamiento y Evaluación sobre VALIDATION
    resultados = []
    modelos_entrenados = {}

    for nombre_modelo, pipeline in pipelines.items():
        res = evaluar_modelo(nombre_modelo, pipeline, X_train, y_train, X_val, y_val)
        resultados.append(res)
        modelos_entrenados[nombre_modelo] = pipeline

    # 5. Formatear y Mostrar Tabla de Resultados de Validación
    df_resultados = pd.DataFrame(resultados)
    
    print("\n" + "="*80)
    print("EVALUACIÓN DE BASELINES EN EL CONJUNTO DE VALIDACIÓN (TEMPORAL)")
    print("="*80)
    print(df_resultados.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*80)
    print(" NOTA METODOLÓGICA: El conjunto TEST permanece 100% reservado e intacto.")
    print("="*80 + "\n")

    return modelos_entrenados, df_resultados


if __name__ == "__main__":
    ejecutar_entrenamiento_baseline()