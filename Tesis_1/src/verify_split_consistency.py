"""
src/verify_split_consistency.py
--------------------------------
Diagnóstico exhaustivo de consistencia de índices y valores del target
falla_futura_30d entre el Split Temporal y el Pipeline del Clasificador.
"""

import sys
import logging
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def verificar_consistencia_split():
    print("=" * 80)
    print("AUDITORÍA DE CONSISTENCIA: TARGET Y PARTICIONES (TRAIN & VALIDATION)")
    print("=" * 80)

    # 1. Carga idéntica
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    # 2. Ejecutar partición
    X_train_g, X_val_g, X_test_g, y_train_g, y_val_g, y_test_g = ejecutar_particion_temporal(df_full)

    # 3. Comparación directa del Target entregado por split_data vs df_full.loc[...]
    y_train_directo = df_full.loc[X_train_g.index, "falla_futura_30d"]
    y_val_directo = df_full.loc[X_val_g.index, "falla_futura_30d"]

    print("\n--- 1. ANÁLISIS EN TRAIN (N = {}) ---".format(len(X_train_g)))
    print(f"y_train_g (retornado por split_data) -> Total Fallas: {y_train_g.sum()} | Tasa: {y_train_g.mean():.4%}")
    print(f"y_train_directo (df_full.loc[X_train.index]) -> Total Fallas: {y_train_directo.sum()} | Tasa: {y_train_directo.mean():.4%}")
    
    coincidencia_train = (y_train_g.values == y_train_directo.values).all()
    print(f"¿Los valores de y_train son 100% idénticos registro a registro?: {coincidencia_train}")

    print("\n--- 2. ANÁLISIS EN VALIDATION (N = {}) ---".format(len(X_val_g)))
    print(f"y_val_g (retornado por split_data) -> Total Fallas: {y_val_g.sum()} | Tasa: {y_val_g.mean():.4%}")
    print(f"y_val_directo (df_full.loc[X_val.index]) -> Total Fallas: {y_val_directo.sum()} | Tasa: {y_val_directo.mean():.4%}")

    coincidencia_val = (y_val_g.values == y_val_directo.values).all()
    print(f"¿Los valores de y_val son 100% idénticos registro a registro?: {coincidencia_val}")

    if not coincidencia_train or not coincidencia_val:
        print("\n" + "!" * 80)
        print("ALERTA: Se detectó una desalineación de índices o de transformación en el target.")
        print("!" * 80)
        
        # Identificar índices discordantes
        diff_train = np.where(y_train_g.values != y_train_directo.values)[0]
        print(f"Número de filas con desalineación en Train: {len(diff_train)}")
    else:
        print("\n" + "=" * 80)
        print("[OK] VERIFICACIÓN EXITOSA: Ambos pipelines trabajan exactamente con las mismas filas y los mismos labels.")
        print("=" * 80)

if __name__ == "__main__":
    verificar_consistencia_split()