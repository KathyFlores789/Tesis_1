"""
src/analyze_regression_targets.py
----------------------------------
BLOQUE 8.1: Análisis Exploratorio y Diagnóstico Estadístico de Targets Continuos
(costo_falla_futura_usd y tiempo_reparacion_horas).

Metodología:
1. Carga los datos e identifica los targets de regresión.
2. Analiza el volumen total vs. volumen positivo (falla_futura_30d = 1) por split temporal.
3. Evalúa la distribución condicional (media, mediana, std, min, max, skewness) para Y > 0.
4. Verifica la consistencia de ceros entre falla y los targets de impacto económico/operativo.
5. NO ENTRIENA NINGÚN MODELO DE REGRESIÓN.
"""

import logging
import sys
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import load_dataset
from src.preprocessing import aplicar_limpieza_base
from src.features import construir_features_t0
from src.split_data import ejecutar_particion_temporal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def analizar_targets_regresion():
    logging.info("Cargando dataset y ejecutando split temporal para diagnóstico de targets...")
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    # Identificar nombres de los targets
    target_clf = "falla_futura_30d"
    target_costo = "costo_falla_futura_usd"
    target_tiempo = "tiempo_reparacion_horas"

    # Verificación de existencia de columnas
    cols_requeridas = [target_clf, target_costo, target_tiempo]
    for col in cols_requeridas:
        if col not in df_full.columns:
            raise KeyError(f"La columna requerida '{col}' no se encuentra en el DataFrame.")

    # Partición temporal
    X_train, X_val, X_test, y_train, y_val, y_test = ejecutar_particion_temporal(df_full)

    # Reconstruir DataFrames por split con sus correspondientes targets
    train_df = df_full.loc[X_train.index]
    val_df = df_full.loc[X_val.index]
    test_df = df_full.loc[X_test.index]

    splits = {
        "TRAIN": train_df,
        "VALIDATION": val_df,
        "TEST": test_df
    }

    print("\n" + "="*90)
    print("BLOQUE 8.1: DIAGNÓSTICO ESTADÍSTICO DE TARGETS PARA REGRESIÓN (COSTO Y TIEMPO)")
    print("="*90)

    # 1. Análisis de Masa de Datos y Proporción de Ceros
    res_volumen = []
    for nombre, df_split in splits.items():
        total_obs = len(df_split)
        fallas_pos = df_split[target_clf].sum()
        pct_fallas = (fallas_pos / total_obs) * 100

        # Ceros en costo y tiempo
        costo_ceros = (df_split[target_costo] == 0).sum()
        tiempo_ceros = (df_split[target_tiempo] == 0).sum()

        res_volumen.append({
            "Split": nombre,
            "Total Obs": total_obs,
            "Fallas (Y=1)": int(fallas_pos),
            "% Fallas": round(pct_fallas, 2),
            "Costo = 0": int(costo_ceros),
            "Tiempo = 0": int(tiempo_ceros)
        })

    df_volumen = pd.DataFrame(res_volumen)
    print("\n1. VOLUMEN DE OBSERVACIONES Y PROPORCIÓN DE CEROS:")
    print("-" * 90)
    print(df_volumen.to_string(index=False))

    # 2. Análisis Estadístico Condicional (Solo Casos donde Y > 0)
    print("\n2. ESTADÍSTICA DESCRIPTIVA CONDICIONAL (ÚNICAMENTE REGISTROS CON FALLA Y > 0):")
    print("-" * 90)

    for target_name, label in [(target_costo, "COSTO (USD)"), (target_tiempo, "TIEMPO (HORAS)")]:
        print(f"\n---> TARGET: {label}")
        res_cond = []
        for nombre, df_split in splits.items():
            pos_data = df_split[df_split[target_clf] == 1][target_name]
            
            res_cond.append({
                "Split": nombre,
                "N Positivos": len(pos_data),
                "Media": round(pos_data.mean(), 2),
                "Mediana": round(pos_data.median(), 2),
                "Std Dev": round(pos_data.std(), 2),
                "Mínimo": round(pos_data.min(), 2),
                "Máximo": round(pos_data.max(), 2),
                "Skewness": round(pos_data.skew(), 2)
            })
        
        df_cond = pd.DataFrame(res_cond)
        print(df_cond.to_string(index=False))

    # 3. Verificación de Integridad Lógica (Cero-Falla)
    print("\n" + "-" * 90)
    print("3. VERIFICACIÓN DE INTEGRIDAD Y CONSISTENCIA LÓGICA:")
    print("-" * 90)

    inconsistencias_costo = df_full[(df_full[target_clf] == 0) & (df_full[target_costo] > 0)]
    inconsistencias_tiempo = df_full[(df_full[target_clf] == 0) & (df_full[target_tiempo] > 0)]

    print(f"  - Registros con Falla=0 pero Costo > 0  : {len(inconsistencias_costo)}")
    print(f"  - Registros con Falla=0 pero Tiempo > 0 : {len(inconsistencias_tiempo)}")
    print("="*90 + "\n")


if __name__ == "__main__":
    analizar_targets_regresion()