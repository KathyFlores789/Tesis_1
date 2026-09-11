"""
src/tune_threshold.py
----------------------
BLOQUE 6.1: Selección Metodológica del Umbral de Decisión (Decision Threshold).

Metodología:
1. Re-entrena el pipeline con los mejores hiperparámetros de Optuna sobre TRAIN.
2. Extrae las probabilidades predichas P(Falla) sobre VALIDATION.
3. Evalúa un barrido de umbrales tau in [0.20, 0.80] midiendo:
   - Precision, Recall, F1-Score, Accuracy
   - Conteo de Matriz de Confusión: TP, FP, TN, FN
   - Tasa de Falsas Alarmas (FP)
4. Muestra un análisis comparativo para seleccionar un umbral operacionalmente defendible.
5. EL CONJUNTO TEST NO SE TOCA.
"""

import logging
import sys
from pathlib import Path
import warnings
import json

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

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

# Mejores hiperparámetros obtenidos con TimeSeriesSplit en TRAIN
BEST_PARAMS = {
    "n_estimators": 125,
    "max_depth": 3,
    "learning_rate": 0.02701341304669535,
    "subsample": 0.6,
    "colsample_bytree": 0.7,
    "scale_pos_weight": 1.4,
    "reg_alpha": 0.09204329627205295,
    "reg_lambda": 0.10692613897465054,
    "random_state": 42,
    "eval_metric": "logloss",
    "n_jobs": -1
}


def analizar_umbrales_validation():
    logging.info("Cargando datos y realizando Split Temporal...")
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    X_train, X_val, X_test, y_train, y_val, y_test = ejecutar_particion_temporal(df_full)

    # Preprocesamiento estrictamente ajustado en TRAIN
    preprocessor = crear_preprocesador_num_cat(X_train)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)

    # Entrenar modelo optimizado sobre TRAIN
    modelo = XGBClassifier(**BEST_PARAMS)
    modelo.fit(X_train_trans, y_train)

    # Probabilidades de falla en VALIDATION
    y_proba_val = modelo.predict_proba(X_val_trans)[:, 1]

    umbrales = np.arange(0.20, 0.85, 0.05)
    resultados = []

    for tau in umbrales:
        y_pred_tau = (y_proba_val >= tau).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_tau).ravel()
        
        rec = recall_score(y_val, y_pred_tau, zero_division=0)
        prec = precision_score(y_val, y_pred_tau, zero_division=0)
        f1 = f1_score(y_val, y_pred_tau, zero_division=0)
        acc = accuracy_score(y_val, y_pred_tau)

        resultados.append({
            "Umbral (tau)": round(tau, 2),
            "Recall (Captura)": round(rec, 4),
            "Precision": round(prec, 4),
            "F1-Score": round(f1, 4),
            "Accuracy": round(acc, 4),
            "TP (Fallas Detectadas)": tp,
            "FP (Falsas Alarmas)": fp,
            "FN (Fallas Omitidas)": fn,
            "TN (Correctos Sanos)": tn
        })

    df_res = pd.DataFrame(resultados)

    print("\n" + "="*95)
    print("ANÁLISIS DE UMBRALES DE DECISIÓN (DECISION THRESHOLD SWEEP) - CONJUNTO VALIDACIÓN")
    print("="*95)
    print(df_res.to_string(index=False))
    print("="*95)
    print(" NOTA: El conjunto TEST permanece 100% reservado e intacto.")
    print("="*95 + "\n")

    return df_res


if __name__ == "__main__":
    analizar_umbrales_validation()