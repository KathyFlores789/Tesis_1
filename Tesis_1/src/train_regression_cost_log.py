"""
src/train_regression_cost_log.py
---------------------------------

EXPERIMENTO 1.1: REGRESIÓN CONDICIONAL DE COSTO
CON TRANSFORMACIÓN LOGARÍTMICA DEL TARGET.

Cambio controlado respecto al baseline:
    y_log = log1p(costo_falla_futura_usd)

Todo lo demás permanece idéntico al baseline:
    - mismas 20 features T0
    - mismo split temporal
    - mismo filtro falla_futura_30d == 1
    - mismo preprocesador
    - mismo Random Forest
    - mismos hiperparámetros
    - TEST_COSTO permanece reservado
"""

import logging
import sys
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ==============================================================
# CONFIGURACIÓN DEL PROYECTO
# ==============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# ==============================================================
# IMPORTACIONES
# ==============================================================

from src.data_loader import load_dataset
from src.preprocessing import aplicar_limpieza_base
from src.features import construir_features_t0
from src.split_data import ejecutar_particion_temporal
from src.train import crear_preprocesador_num_cat

from src.audit_regression_features import (
    FEATURES_T0_REGRESION_COSTO
)

# ==============================================================
# CONFIGURACIÓN
# ==============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

TARGET_CLF = "falla_futura_30d"
TARGET_COSTO = "costo_falla_futura_usd"


# ==============================================================
# MÉTRICAS
# ==============================================================

def evaluar_metricas(y_true, y_pred, subset_name):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    print(
        f"\n--- RENDIMIENTO EN "
        f"{subset_name.upper()}_COSTO ---"
    )

    print(f"MAE  : ${mae:.2f}")
    print(f"RMSE : ${rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ==============================================================
# FUNCIÓN PRINCIPAL
# ==============================================================

def ejecutar_experimento_1_1():

    # ----------------------------------------------------------
    # 1. CARGA Y FEATURES
    # ----------------------------------------------------------

    logging.info(
        "Cargando dataset y construyendo features T0..."
    )

    df_raw = load_dataset()

    df_clean = aplicar_limpieza_base(
        df_raw
    )

    df_full = construir_features_t0(
        df_clean
    )

    # ----------------------------------------------------------
    # 2. SPLIT TEMPORAL GLOBAL
    # ----------------------------------------------------------

    logging.info(
        "Ejecutando Split Temporal Global..."
    )

    (
        X_train_g,
        X_val_g,
        X_test_g,
        y_train_g,
        y_val_g,
        y_test_g
    ) = ejecutar_particion_temporal(
        df_full
    )

    # ----------------------------------------------------------
    # 3. RECONSTRUIR DATAFRAMES POR SPLIT
    # ----------------------------------------------------------

    df_train_g = df_full.loc[
        X_train_g.index
    ].copy()

    df_val_g = df_full.loc[
        X_val_g.index
    ].copy()

    df_test_g = df_full.loc[
        X_test_g.index
    ].copy()

    # ----------------------------------------------------------
    # 4. FILTRO CONDICIONAL
    # ----------------------------------------------------------

    logging.info(
        "Filtrando falla_futura_30d == 1 "
        "dentro de cada split..."
    )

    train_cost = df_train_g[
        df_train_g[TARGET_CLF] == 1
    ].copy()

    val_cost = df_val_g[
        df_val_g[TARGET_CLF] == 1
    ].copy()

    test_cost = df_test_g[
        df_test_g[TARGET_CLF] == 1
    ].copy()

    # ----------------------------------------------------------
    # 5. VERIFICACIÓN DEL TARGET
    # ----------------------------------------------------------

    if TARGET_COSTO not in df_full.columns:

        raise KeyError(
            f"No existe el target "
            f"'{TARGET_COSTO}' en df_full."
        )

    # ----------------------------------------------------------
    # 6. MATRICES X / y
    # ----------------------------------------------------------

    X_train_cost = train_cost[
        FEATURES_T0_REGRESION_COSTO
    ].copy()

    y_train_cost = train_cost[
        TARGET_COSTO
    ].copy()

    X_val_cost = val_cost[
        FEATURES_T0_REGRESION_COSTO
    ].copy()

    y_val_cost = val_cost[
        TARGET_COSTO
    ].copy()

    X_test_cost = test_cost[
        FEATURES_T0_REGRESION_COSTO
    ].copy()

    y_test_cost = test_cost[
        TARGET_COSTO
    ].copy()

    # ----------------------------------------------------------
    # 7. AUDITORÍA DE MUESTRAS
    # ----------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "EXPERIMENTO 1.1 — AUDITORÍA DE MUESTRAS"
    )
    print("=" * 80)

    print(
        f"TRAIN_COSTO      : "
        f"{len(X_train_cost)} registros"
    )

    print(
        f"VALIDATION_COSTO : "
        f"{len(X_val_cost)} registros"
    )

    print(
        f"TEST_COSTO       : "
        f"{len(X_test_cost)} registros "
        f"[RESERVADO]"
    )

    print(
        f"FEATURES T0      : "
        f"{len(FEATURES_T0_REGRESION_COSTO)}"
    )

    print("=" * 80)

    # ----------------------------------------------------------
    # 8. VERIFICACIÓN CONDICIONAL
    # ----------------------------------------------------------

    assert (
        train_cost[TARGET_CLF] == 1
    ).all()

    assert (
        val_cost[TARGET_CLF] == 1
    ).all()

    assert (
        test_cost[TARGET_CLF] == 1
    ).all()

    # ----------------------------------------------------------
    # 9. TRANSFORMACIÓN LOGARÍTMICA
    #
    # ÚNICO CAMBIO RESPECTO AL BASELINE
    # ----------------------------------------------------------

    if (y_train_cost < 0).any():
        raise ValueError(
            "El target contiene valores negativos. "
            "log1p no es válido."
        )

    if (y_val_cost < 0).any():
        raise ValueError(
            "VALIDATION contiene valores negativos."
        )

    y_train_log = np.log1p(
        y_train_cost
    )

    # ----------------------------------------------------------
    # 10. PREPROCESADOR
    # ----------------------------------------------------------

    logging.info(
        "Ajustando preprocesador "
        "EXCLUSIVAMENTE sobre TRAIN_COSTO..."
    )

    preprocessor = crear_preprocesador_num_cat(
        X_train_cost
    )

    # ----------------------------------------------------------
    # 11. RANDOM FOREST
    #
    # EXACTAMENTE IGUAL AL BASELINE
    # ----------------------------------------------------------

    modelo_rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    # ----------------------------------------------------------
    # 12. PIPELINE
    # ----------------------------------------------------------

    pipeline_rf = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            modelo_rf
        )
    ])

    # ----------------------------------------------------------
    # 13. ENTRENAMIENTO SOBRE log1p(COSTO)
    # ----------------------------------------------------------

    logging.info(
        "Entrenando Random Forest sobre "
        "log1p(COSTO)..."
    )

    pipeline_rf.fit(
        X_train_cost,
        y_train_log
    )

    # ----------------------------------------------------------
    # 14. PREDICCIONES EN ESCALA LOG
    # ----------------------------------------------------------

    pred_train_log = pipeline_rf.predict(
        X_train_cost
    )

    pred_val_log = pipeline_rf.predict(
        X_val_cost
    )

    # ----------------------------------------------------------
    # 15. INVERSIÓN DE LA TRANSFORMACIÓN
    #
    # expm1(log1p(y)) = y
    # ----------------------------------------------------------

    pred_train = np.expm1(
        pred_train_log
    )

    pred_val = np.expm1(
        pred_val_log
    )

    # ----------------------------------------------------------
    # 16. SEGURIDAD CONTRA PREDICCIONES NEGATIVAS
    # ----------------------------------------------------------

    pred_train = np.maximum(
        pred_train,
        0
    )

    pred_val = np.maximum(
        pred_val,
        0
    )

    # ----------------------------------------------------------
    # 17. EVALUACIÓN EN ESCALA ORIGINAL USD
    # ----------------------------------------------------------

    metricas_train = evaluar_metricas(
        y_train_cost,
        pred_train,
        "TRAIN"
    )

    metricas_val = evaluar_metricas(
        y_val_cost,
        pred_val,
        "VALIDATION"
    )

    # ----------------------------------------------------------
    # 18. IMPORTANCIA DE VARIABLES
    # ----------------------------------------------------------

    feature_names = (
        pipeline_rf
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = (
        pipeline_rf
        .named_steps["regressor"]
        .feature_importances_
    )

    df_importancia = pd.DataFrame({

        "Predictor": feature_names,

        "Importancia": importances

    }).sort_values(
        by="Importancia",
        ascending=False
    )

    print("\n" + "=" * 80)

    print(
        "IMPORTANCIA DE VARIABLES — "
        "RF + log1p(COSTO)"
    )

    print("=" * 80)

    print(
        "\nTOP 15 PREDICTORES:"
    )

    print(
        df_importancia
        .head(15)
        .to_string(index=False)
    )

    # ----------------------------------------------------------
    # 19. INFORME FINAL
    # ----------------------------------------------------------

    print("\n" + "=" * 80)

    print(
        "EXPERIMENTO 1.1 COMPLETADO"
    )

    print("=" * 80)

    print(
        "\nTransformación aplicada al target:"
    )

    print(
        "    y_log = log1p(costo_falla_futura_usd)"
    )

    print(
        "\nLas métricas fueron calculadas "
        "en escala original USD."
    )

    print(
        "\nEl modelo fue entrenado exclusivamente "
        "con observaciones donde:"
    )

    print(
        f"    {TARGET_CLF} == 1"
    )

    print(
        "\nTEST_COSTO NO FUE UTILIZADO "
        "DURANTE EL ENTRENAMIENTO NI LA VALIDACIÓN."
    )

    print("=" * 80)

    return {
        "modelo": pipeline_rf,
        "metricas_train": metricas_train,
        "metricas_validation": metricas_val,
        "importancia": df_importancia
    }


# ==============================================================
# EJECUCIÓN
# ==============================================================

if __name__ == "__main__":

    ejecutar_experimento_1_1()