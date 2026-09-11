"""
src/tune_classifier.py
-----------------------
BLOQUE 6: Optimización de Hiperparámetros para XGBoost con Validación Temporal Interna (CV).

Metodología Rígida:
1. TRAIN (Pasado) se divide internamente usando TimeSeriesSplit (ej. 3 folds cronológicos).
2. Optuna optimiza el ROC-AUC PROMEDIO de los folds temporales DENTRO DE TRAIN.
3. El modelo con mejores hiperparámetros se re-entrena sobre el TRAIN COMPLETO.
4. Se evalúan las métricas en VALIDATION (Hold-out temporal intermedio).
5. EL CONJUNTO TEST PERMANECE 100% INTACTO Y FUERA DEL PROCESO.
"""

import logging
import sys
from pathlib import Path
import warnings
import json

import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Desactivar advertencias de Optuna para mantener una consola limpia
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import load_dataset
from src.preprocessing import aplicar_limpieza_base
from src.features import construir_features_t0
from src.split_data import ejecutar_particion_temporal
from src.train import crear_preprocesador_num_cat

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def objetivo_optuna_tscv(trial, X_train_raw, y_train_raw, n_splits=3):
    """
    Función de evaluación para Optuna usando TimeSeriesSplit exclusivamente dentro de TRAIN.
    Aplica el Preprocesador DENTRO de cada fold para evitar cualquier fuga de información (Data Leakage).
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=25),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.20, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 2.0, step=0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_jobs': -1
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores_auc_folds = []

    # Iterar cronológicamente sobre los folds internos de TRAIN
    for fold, (train_fold_idx, val_fold_idx) in enumerate(tscv.split(X_train_raw)):
        X_tr_fold = X_train_raw.iloc[train_fold_idx]
        y_tr_fold = y_train_raw.iloc[train_fold_idx]
        
        X_va_fold = X_train_raw.iloc[val_fold_idx]
        y_va_fold = y_train_raw.iloc[val_fold_idx]

        # 1. Fit del preprocesador SOLO en el sub-train del fold
        preproc_fold = crear_preprocesador_num_cat(X_tr_fold)
        X_tr_trans = preproc_fold.fit_transform(X_tr_fold)
        X_va_trans = preproc_fold.transform(X_va_fold)

        # 2. Entrenar XGBoost
        model = XGBClassifier(**params)
        model.fit(X_tr_trans, y_tr_fold)

        # 3. Evaluar en el sub-validation del fold
        y_proba_fold = model.predict_proba(X_va_trans)[:, 1]
        
        # Manejo preventivo si un fold tiene una sola clase en y_va_fold
        if len(np.unique(y_va_fold)) > 1:
            auc_fold = roc_auc_score(y_va_fold, y_proba_fold)
            scores_auc_folds.append(auc_fold)

    # Retornar el promedio de ROC-AUC entre todos los folds temporales
    return np.mean(scores_auc_folds)


def ejecutar_optimizacion_estricta(n_trials: int = 50, n_splits_cv: int = 3):
    logging.info("Cargando datos y generando Split Temporal principal...")
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    # 1. Separación Temporal Externa (Train / Validation / Test)
    X_train, X_val, X_test, y_train, y_val, y_test = ejecutar_particion_temporal(df_full)

    logging.info(f"Iniciando Optuna mediante TimeSeriesSplit ({n_splits_cv} Folds) ÚNICAMENTE en TRAIN...")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objetivo_optuna_tscv(trial, X_train, y_train, n_splits=n_splits_cv),
        n_trials=n_trials,
        show_progress_bar=True
    )

    mejores_params = study.best_params
    mejor_auc_cv_train = study.best_value

    logging.info(f"Optimización finalizada. Mejor ROC-AUC Promedio (CV sobre TRAIN): {mejor_auc_cv_train:.4f}")

    # 2. Re-entrenamiento del Pipeline Completo sobre TODO el conjunto TRAIN
    logging.info("Ajustando Preprocesador y Modelo final sobre el conjunto TRAIN COMPLETO...")
    preprocessor_train = crear_preprocesador_num_cat(X_train)
    X_train_trans = preprocessor_train.fit_transform(X_train)
    X_val_trans = preprocessor_train.transform(X_val)

    mejores_params.update({
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_jobs': -1
    })

    modelo_final = XGBClassifier(**mejores_params)
    modelo_final.fit(X_train_trans, y_train)

    # 3. Evaluación en el conjunto de VALIDATION (Hold-out temporal real)
    y_pred_val = modelo_final.predict(X_val_trans)
    y_proba_val = modelo_final.predict_proba(X_val_trans)[:, 1]

    acc = accuracy_score(y_val, y_pred_val)
    prec = precision_score(y_val, y_pred_val, zero_division=0)
    rec = recall_score(y_val, y_pred_val, zero_division=0)
    f1 = f1_score(y_val, y_pred_val, zero_division=0)
    auc = roc_auc_score(y_val, y_proba_val)

    # 4. Impresión de Reporte Metodológico
    print("\n" + "="*80)
    print("BLOQUE 6: OPTIMIZACIÓN XGBOOST CON TIME-SERIES SPLIT EN TRAIN")
    print("="*80)
    print("Mejores Hiperparámetros Encontrados (Vía TimeSeriesSplit en Train):")
    print(json.dumps(study.best_params, indent=2))
    print("-" * 80)
    print(f"ROC-AUC Promedio en Folds de TRAIN (CV) : {mejor_auc_cv_train:.4f}")
    print("-" * 80)
    print(f"EVALUACIÓN EN VALIDATION SET (Hold-out Intermedio Limpio):")
    print(f"  - ROC-AUC   : {auc:.4f}  (Baseline XGBoost: 0.7599)")
    print(f"  - F1-Score  : {f1:.4f}  (Baseline XGBoost: 0.6569)")
    print(f"  - Recall    : {rec:.4f}  (Baseline XGBoost: 0.7500)")
    print(f"  - Precision : {prec:.4f}  (Baseline XGBoost: 0.5844)")
    print(f"  - Accuracy  : {acc:.4f}  (Baseline XGBoost: 0.6887)")
    print("="*80)
    print(" NOTA METODOLÓGICA: VALIDATION no fue visto durante la búsqueda de Optuna.")
    print(" NOTA METODOLÓGICA: El conjunto TEST permanece 100% reservado e intacto.")
    print("="*80 + "\n")

    return mejores_params, modelo_final


if __name__ == "__main__":
    ejecutar_optimizacion_estricta(n_trials=50, n_splits_cv=3)