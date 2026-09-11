"""
src/audit_fase_a_b_descriptivo_clasificador.py
------------------------------------------------
PASO A: Análisis Descriptivo de costo_falla_futura_usd (Train vs Validation)
PASO B: Inspección y Auditoría de Compatibilidad del Clasificador
"""

import logging
import sys
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import load_dataset
from src.preprocessing import aplicar_limpieza_base
from src.features import construir_features_t0
from src.split_data import ejecutar_particion_temporal
from src.audit_regression_features import FEATURES_T0_REGRESION_COSTO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def ejecutar_paso_a_y_b():
    print("\n" + "=" * 80)
    print("PASO A — ANÁLISIS DESCRIPTIVO DEL COSTO DE FALLA (USD)")
    print("=" * 80)

    # Load Data
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    (X_train_g, X_val_g, X_test_g, y_train_g, y_val_g, y_test_g) = ejecutar_particion_temporal(df_full)

    df_train_g = df_full.loc[X_train_g.index].copy()
    df_val_g = df_full.loc[X_val_g.index].copy()

    # Cost Subsets
    train_cost = df_train_g[df_train_g["falla_futura_30d"] == 1]["costo_falla_futura_usd"]
    val_cost = df_val_g[df_val_g["falla_futura_30d"] == 1]["costo_falla_futura_usd"]

    def obtener_estadisticas(series, nombre):
        return {
            "Subset": nombre,
            "N": len(series),
            "Min": series.min(),
            "P25": series.quantile(0.25),
            "Mediana (P50)": series.median(),
            "Media": series.mean(),
            "P75": series.quantile(0.75),
            "Max": series.max(),
            "Desv. Est.": series.std(),
            "IQR": series.quantile(0.75) - series.quantile(0.25)
        }

    df_stats = pd.DataFrame([
        obtener_estadisticas(train_cost, "TRAIN_COSTO"),
        obtener_estadisticas(val_cost, "VALIDATION_COSTO")
    ])

    print(df_stats.to_string(index=False))
    print("-" * 80)

    # Contextualización del MAE
    mae_elasticnet = 98.43
    mediana_val = val_cost.median()
    media_val = val_cost.mean()

    print("\nCONTEXTUALIZACIÓN DEL ERROR (MAE = $98.43 EN VALIDATION_COSTO):")
    print(f"  • Porcentaje del MAE respecto a la MEDIANA en Val (${mediana_val:.2f}) : {(mae_elasticnet / mediana_val)*100:.2f}%")
    print(f"  • Porcentaje del MAE respecto a la MEDIA en Val   (${media_val:.2f}) : {(mae_elasticnet / media_val)*100:.2f}%")
    print(f"  • Rango Intercuartílico (IQR) en Val              : ${df_stats.loc[1, 'IQR']:.2f}")

    print("\n" + "=" * 80)
    print("PASO B — REVISIÓN Y COMPATIBILIDAD DEL CLASIFICADOR")
    print("=" * 80)

    # Verificar si existe modelo guardado del clasificador
    model_path = ROOT_DIR / "models" / "best_classifier.joblib"
    if not model_path.exists():
        # Buscar alternativas en la carpeta models
        models_dir = ROOT_DIR / "models"
        available_models = list(models_dir.glob("*.joblib")) if models_dir.exists() else []
        print(f"Archivo por defecto not encontrado en: {model_path}")
        print(f"Modelos guardados encontrados en 'models/': {[m.name for m in available_models]}")
    else:
        import joblib
        clf_pipeline = joblib.load(model_path)
        print(f"Clasificador cargado exitosamente desde: {model_path}")
        print(f"Tipo de Estimador/Pipeline: {type(clf_pipeline)}")
        if hasattr(clf_pipeline, "feature_names_in_"):
            print(f"Número de Features esperadas por el Clasificador: {len(clf_pipeline.feature_names_in_)}")
            print(f"Features del Clasificador: {list(clf_pipeline.feature_names_in_)}")
        
        # Probar predict_proba en Validation
        X_val_all = df_val_g[FEATURES_T0_REGRESION_COSTO]
        if hasattr(clf_pipeline, "predict_proba"):
            probs = clf_pipeline.predict_proba(X_val_all)[:, 1]
            print(f"\n`predict_proba()` ejecutado con éxito en VALIDATION_G (N={len(probs)}).")
            print(f"  • Probabilidad Mínima : {probs.min():.4f}")
            print(f"  • Probabilidad Media   : {probs.mean():.4f}")
            print(f"  • Probabilidad Máxima  : {probs.max():.4f}")

if __name__ == "__main__":
    ejecutar_paso_a_y_b()