"""
src/features.py
---------------
Módulo de Ingeniería de Características sintéticas en T0.
Usa mapeo explícito de columnas y semántica rigurosa.
"""

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def construir_features_t0(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera características T0 de dominio automotriz y geográfico de Quito.
    """
    df_feat = df.copy()

    # 1. Antigüedad del vehículo (Años aproximados por año de fabricación)
    if 'anio' in df_feat.columns and 'fecha_mantenimiento' in df_feat.columns:
        df_feat['fecha_mantenimiento_dt'] = pd.to_datetime(df_feat['fecha_mantenimiento'], errors='coerce')
        año_mant = df_feat['fecha_mantenimiento_dt'].dt.year
        df_feat['edad_vehiculo_años'] = np.maximum(0, año_mant - df_feat['anio'])

    # 2. Kilometraje Promedio Estimado Diario (Semántica precisa)
    if 'kilometraje_acumulado_km' in df_feat.columns and 'edad_vehiculo_años' in df_feat.columns:
        dias_vida = np.maximum(1, df_feat['edad_vehiculo_años'] * 365.25)
        df_feat['km_promedio_estimado_diario'] = (df_feat['kilometraje_acumulado_km'] / dias_vida).round(2)

    # 3. Índice Sintético de Estrés Térmico y Altitudinal (Propuesto: Quito ~2800m / 90°C)
    if 'altitud_operacion_m' in df_feat.columns and 'temperatura_promedio_motor_c' in df_feat.columns:
        df_feat['indice_estres_termico_altitud'] = (
            (df_feat['altitud_operacion_m'] / 2800.0) * (df_feat['temperatura_promedio_motor_c'] / 90.0)
        ).round(3)

    # 4. Proporción del recorrido reciente sobre acumulado (Estricto sin fallbacks incongruentes)
    if 'km_desde_ultimo_mantenimiento' in df_feat.columns and 'kilometraje_acumulado_km' in df_feat.columns:
        df_feat['ratio_recorrido_reciente'] = np.where(
            df_feat['kilometraje_acumulado_km'] > 0,
            (df_feat['km_desde_ultimo_mantenimiento'] / df_feat['kilometraje_acumulado_km']).round(4),
            0.0
        )

    logging.info(f"Ingeniería de características T0 completada. Total columnas: {df_feat.shape[1]}")
    return df_feat


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))

    from src.data_loader import load_dataset
    from src.preprocessing import aplicar_limpieza_base

    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_features = construir_features_t0(df_clean)

    print("\n" + "="*80)
    print("MUESTRA DE NUEVAS CARACTERÍSTICAS T0 (REFACTORIZADO)")
    print("="*80)
    nuevas = ['edad_vehiculo_años', 'km_promedio_estimado_diario', 'indice_estres_termico_altitud', 'ratio_recorrido_reciente']
    print(df_features[nuevas].head().to_string(index=False))
    print("="*80)