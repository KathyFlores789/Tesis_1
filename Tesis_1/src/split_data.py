"""
src/split_data.py
-----------------
Separación Causal Estricta (T0 vs T1) mediante listas explícitas 
y división cronológica (Time-Based Split) con umbrales de fecha no solapados.
Preserva el índice original de df_full para mantener trazabilidad perfecta.
"""

import logging
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import load_dataset
from src.preprocessing import aplicar_limpieza_base
from src.features import construir_features_t0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# DECLARACIÓN EXPLÍCITA DE VARIABLES T1 Y METADATOS PROHIBIDOS EN X
VARIABLES_EXPLICITAS_T1 = [
    'falla_futura_30d',
    'dias_hasta_falla',
    'fecha_falla_futura',
    'sistema_falla_futura',
    'repuesto_falla_futura',
    'costo_falla_futura_usd',
    'tiempo_reparacion_horas'
]

METADATOS_EXCLUIDOS = [
    'vehiculo_id',
    'fecha_mantenimiento',
    'fecha_mantenimiento_dt'
]


def separar_x_y(df: pd.DataFrame, target_col: str = 'falla_futura_30d'):
    """
    Aísla explícitamente variables del futuro (T1) y metadatos no predictivos,
    asegurando la preservación estricta de índices entre X e y.
    """
    cols_a_excluir = set(VARIABLES_EXPLICITAS_T1 + METADATOS_EXCLUIDOS)
    features_cols = [c for c in df.columns if c not in cols_a_excluir]

    X = df[features_cols].copy()
    y = df[target_col].copy()

    # Validación de alineación interna de índices
    assert X.index.equals(y.index), "ERROR CRÍTICO: X e y no tienen índices idénticos tras separar."

    logging.info(f"Separación explícita completada. Predictores en X ({len(features_cols)}): {features_cols}")
    return X, y


def ejecutar_particion_temporal(df_full: pd.DataFrame, target_col: str = 'falla_futura_30d', test_ratio=0.15, val_ratio=0.15):
    """
    Ejecuta un Time-Based Split respetando fechas únicas y preservando los índices
    originales de df_full para garantizar trazabilidad.
    """
    # 1. Asegurar formato datetime y ordenar SIN destruir el índice original
    df_sorted = df_full.copy()
    df_sorted['fecha_mantenimiento'] = pd.to_datetime(df_sorted['fecha_mantenimiento'])
    df_sorted = df_sorted.sort_values(by='fecha_mantenimiento')  # NOTA: Se evita reset_index(drop=True)

    n_total = len(df_sorted)
    target_train_idx = int(n_total * (1.0 - val_ratio - test_ratio))
    target_val_idx = int(n_total * (1.0 - test_ratio))

    # 2. Determinar fechas de corte basadas en los índices objetivo
    fecha_corte_train = df_sorted.iloc[target_train_idx]['fecha_mantenimiento']
    fecha_corte_val = df_sorted.iloc[target_val_idx]['fecha_mantenimiento']

    # 3. Asignación por umbral estricto de fechas (evita partir un mismo día)
    df_train = df_sorted[df_sorted['fecha_mantenimiento'] < fecha_corte_train].copy()
    df_val = df_sorted[
        (df_sorted['fecha_mantenimiento'] >= fecha_corte_train) & 
        (df_sorted['fecha_mantenimiento'] < fecha_corte_val)
    ].copy()
    df_test = df_sorted[df_sorted['fecha_mantenimiento'] >= fecha_corte_val].copy()

    # 4. Verificación matemática de secuencia temporal
    max_train = df_train['fecha_mantenimiento'].max()
    min_val = df_val['fecha_mantenimiento'].min()
    max_val = df_val['fecha_mantenimiento'].max()
    min_test = df_test['fecha_mantenimiento'].min()

    assert max_train < min_val, f"¡ERROR DE SOLAPAMIENTO! max(Train): {max_train} >= min(Val): {min_val}"
    assert max_val < min_test, f"¡ERROR DE SOLAPAMIENTO! max(Val): {max_val} >= min(Test): {min_test}"

    # 5. Reporte detallado en consola
    print("\n" + "="*80)
    print("REPORTE DE PARTIZIÓN TEMPORAL STRICT (DATE-BASED SPLIT)")
    print("="*80)
    print("VERIFICACIÓN AUTOMÁTICA DE ORDEN CRONOLÓGICO: PASADA [OK]")
    print(f"  - max(fecha_train) < min(fecha_val) : {max_train.strftime('%Y-%m-%d')} < {min_val.strftime('%Y-%m-%d')}")
    print(f"  - max(fecha_val)   < min(fecha_test): {max_val.strftime('%Y-%m-%d')} < {min_test.strftime('%Y-%m-%d')}")
    print("-" * 80)
    print("1. TRAIN SET (Pasado):")
    print(f"   Rango Fechas : {df_train['fecha_mantenimiento'].min().strftime('%Y-%m-%d')} a {max_train.strftime('%Y-%m-%d')}")
    print(f"   Registros    : {len(df_train)} ({len(df_train)/n_total:.1%})")
    print(f"   Tasa Falla   : {df_train[target_col].mean():.2%}")
    
    print("\n2. VALIDATION SET (Intermedio):")
    print(f"   Rango Fechas : {min_val.strftime('%Y-%m-%d')} a {max_val.strftime('%Y-%m-%d')}")
    print(f"   Registros    : {len(df_val)} ({len(df_val)/n_total:.1%})")
    print(f"   Tasa Falla   : {df_val[target_col].mean():.2%}")

    print("\n3. TEST SET (Futuro):")
    print(f"   Rango Fechas : {min_test.strftime('%Y-%m-%d')} a {df_test['fecha_mantenimiento'].max().strftime('%Y-%m-%d')}")
    print(f"   Registros    : {len(df_test)} ({len(df_test)/n_total:.1%})")
    print(f"   Tasa Falla   : {df_test[target_col].mean():.2%}")
    print("="*80 + "\n")

    # 6. Extraer X e y
    X_train, y_train = separar_x_y(df_train, target_col)
    X_val, y_val = separar_x_y(df_val, target_col)
    X_test, y_test = separar_x_y(df_test, target_col)

    # 7. Aserciones estrictas de alineación de índices post-separación
    assert X_train.index.equals(y_train.index), "ERROR CRÍTICO: Índices de X_train y y_train no coinciden."
    assert X_val.index.equals(y_val.index), "ERROR CRÍTICO: Índices de X_val y y_val no coinciden."
    assert X_test.index.equals(y_test.index), "ERROR CRÍTICO: Índices de X_test y y_test no coinciden."

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    X_train, X_val, X_test, y_train, y_val, y_test = ejecutar_particion_temporal(df_full)