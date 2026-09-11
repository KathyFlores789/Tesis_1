"""
src/train_regression_cost.py
-----------------------------

BLOQUE 8.2: BASELINE DE REGRESIÓN CONDICIONAL DEL COSTO
DE FALLA FUTURA (USD).

Objetivo:
    Predecir costo_falla_futura_usd condicionado a que
    falla_futura_30d == 1.

Metodología:

1. Cargar dataset.
2. Construir features T0.
3. Ejecutar SPLIT TEMPORAL GLOBAL.
4. Filtrar falla_futura_30d == 1 dentro de cada split.
5. Utilizar exclusivamente la whitelist T0 autorizada.
6. Ajustar preprocesador SOLO sobre TRAIN_COSTO.
7. Entrenar Random Forest Regressor.
8. Evaluar en TRAIN_COSTO y VALIDATION_COSTO.
9. TEST_COSTO permanece completamente intacto.

IMPORTANTE:
    El modelo NO predice si ocurrirá una falla.
    Esa tarea pertenece al clasificador de la Etapa 1.

    Este modelo responde:

        "Si ocurre una falla futura,
         ¿cuál será aproximadamente su costo?"
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
# IMPORTACIONES DEL PROYECTO
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

def ejecutar_baseline_regresion_costo():

    # ----------------------------------------------------------
    # 1. CARGA Y CONSTRUCCIÓN DE FEATURES
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
    #
    # IMPORTANTE:
    # PRIMERO split temporal
    # DESPUÉS falla == 1
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
    # 7. VALIDACIONES DE SEGURIDAD
    # ----------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "AUDITORÍA DE MUESTRAS PARA REGRESIÓN "
        "CONDICIONAL DE COSTO"
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
    # 8. COMPROBAR QUE TODOS LOS CASOS SON FALLA = 1
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
    # 9. PREPROCESAMIENTO
    #
    # SOLO TRAIN_COSTO
    # ----------------------------------------------------------

    logging.info(
        "Ajustando preprocesador "
        "EXCLUSIVAMENTE sobre TRAIN_COSTO..."
    )

    preprocessor = crear_preprocesador_num_cat(
        X_train_cost
    )


    # ----------------------------------------------------------
    # 10. RANDOM FOREST
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
    # 11. PIPELINE
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
    # 12. ENTRENAMIENTO
    # ----------------------------------------------------------

    logging.info(
        "Entrenando Random Forest "
        "sobre TRAIN_COSTO..."
    )

    pipeline_rf.fit(
        X_train_cost,
        y_train_cost
    )


    # ----------------------------------------------------------
    # 13. PREDICCIONES
    # ----------------------------------------------------------

    pred_train = pipeline_rf.predict(
        X_train_cost
    )

    pred_val = pipeline_rf.predict(
        X_val_cost
    )


    # ----------------------------------------------------------
    # 14. EVALUACIÓN
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
    # 15. IMPORTANCIA DE VARIABLES
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
        "RANDOM FOREST"
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
    # 16. INFORME FINAL
    # ----------------------------------------------------------

    print("\n" + "=" * 80)

    print(
        "BLOQUE 8.2 COMPLETADO"
    )

    print("=" * 80)

    print(
        "\nEl modelo fue entrenado exclusivamente "
        "con observaciones donde:"
    )

    print(
        f"    {TARGET_CLF} == 1"
    )

    print(
        "\nEl target de regresión es:"
    )

    print(
        f"    {TARGET_COSTO}"
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

    ejecutar_baseline_regresion_costo()