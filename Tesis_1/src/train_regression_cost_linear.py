"""
src/train_regression_cost_linear.py
------------------------------------
EXPERIMENTO 1.2: REGRESIÓN CONDICIONAL DE COSTO CON MODELOS LINEALES REGULARIZADOS
Modelos: Ridge y ElasticNet sobre log1p(costo_falla_futura_usd)

Controles estrictos:
  - Mismas 20 features T0
  - Mismo split temporal (Train/Val/Test reservado)
  - Mismo filtro: falla_futura_30d == 1
  - Preprocesador con StandardScaler (crucial para modelos lineales) ajustado SOLAMENTE sobre TRAIN
  - Inversión expm1(y) y métricas evaluadas estrictamente en USD reales
"""

import logging
import sys
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import load_dataset
from src.preprocessing import aplicar_limpieza_base
from src.features import construir_features_t0
from src.split_data import ejecutar_particion_temporal
from src.train import crear_preprocesador_num_cat
from src.audit_regression_features import FEATURES_T0_REGRESION_COSTO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_CLF = "falla_futura_30d"
TARGET_COSTO = "costo_falla_futura_usd"

def evaluar_metricas(y_true, y_pred, subset_name, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- RENDIMIENTO [{model_name.upper()}] EN {subset_name.upper()}_COSTO ---")
    print(f"MAE  : ${mae:.2f}")
    print(f"RMSE : ${rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def ejecutar_experimento_1_2():
    # 1. Carga y Features
    logging.info("Cargando dataset y construyendo features T0...")
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    # 2. Split Temporal Global
    (X_train_g, X_val_g, X_test_g, y_train_g, y_val_g, y_test_g) = ejecutar_particion_temporal(df_full)

    df_train_g = df_full.loc[X_train_g.index].copy()
    df_val_g = df_full.loc[X_val_g.index].copy()
    df_test_g = df_full.loc[X_test_g.index].copy()

    # 3. Filtro Condicional
    train_cost = df_train_g[df_train_g[TARGET_CLF] == 1].copy()
    val_cost = df_val_g[df_val_g[TARGET_CLF] == 1].copy()
    test_cost = df_test_g[df_test_g[TARGET_CLF] == 1].copy()

    X_train = train_cost[FEATURES_T0_REGRESION_COSTO].copy()
    y_train_usd = train_cost[TARGET_COSTO].copy()

    X_val = val_cost[FEATURES_T0_REGRESION_COSTO].copy()
    y_val_usd = val_cost[TARGET_COSTO].copy()

    print("\n" + "=" * 80)
    print("EXPERIMENTO 1.2 — MODELOS LINEALES REGULARIZADOS (RIDGE & ELASTICNET)")
    print("=" * 80)
    print(f"TRAIN_COSTO      : {len(X_train)} registros")
    print(f"VALIDATION_COSTO : {len(X_val)} registros")
    print(f"TEST_COSTO       : {len(test_cost)} registros [RESERVADO]")
    print(f"FEATURES T0      : {len(FEATURES_T0_REGRESION_COSTO)}")
    print("=" * 80)

    # 4. Transformación Logarítmica Target
    y_train_log = np.log1p(y_train_usd)

    # 5. Preprocesador con Escalado Estándar
    preprocessor = crear_preprocesador_num_cat(X_train)

    # 6. Definición de Modelos Lineales
    modelos = {
        "Ridge (alpha=1.0)": Ridge(alpha=1.0, random_state=42),
        "Ridge (alpha=10.0)": Ridge(alpha=10.0, random_state=42),
        "ElasticNet (alpha=0.1, l1=0.5)": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "ElasticNet (alpha=0.5, l1=0.2)": ElasticNet(alpha=0.5, l1_ratio=0.2, random_state=42)
    }

    resultados = {}

    for nombre, reg_model in modelos.items():
        logging.info(f"Entrenando {nombre}...")
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", reg_model)
        ])

        pipeline.fit(X_train, y_train_log)

        # Predicciones y Reconversión
        pred_train_log = pipeline.predict(X_train)
        pred_val_log = pipeline.predict(X_val)

        pred_train_usd = np.maximum(np.expm1(pred_train_log), 0)
        pred_val_usd = np.maximum(np.expm1(pred_val_log), 0)

        # Evaluación
        m_train = evaluar_metricas(y_train_usd, pred_train_usd, "TRAIN", nombre)
        m_val = evaluar_metricas(y_val_usd, pred_val_usd, "VALIDATION", nombre)

        resultados[nombre] = {
            "pipeline": pipeline,
            "m_train": m_train,
            "m_val": m_val
        }

    # 7. Tabla Resumen Comparativa
    print("\n" + "=" * 80)
    print("TABLA COMPARATIVA FINAL — EXPERIMENTO 1.2 (EVALUADO EN USD REALES)")
    print("=" * 80)
    
    tabla_data = []
    
    # baseline y exp 1.1 de referencia
    tabla_data.append({"Modelo": "Baseline RF (original)", "MAE Val": 110.40, "RMSE Val": 162.04, "R2 Val": -0.0110})
    tabla_data.append({"Modelo": "Exp 1.1 RF + log1p", "MAE Val": 101.68, "RMSE Val": 164.20, "R2 Val": -0.0382})
    
    for nombre, res in resultados.items():
        mv = res["m_val"]
        tabla_data.append({
            "Modelo": nombre,
            "MAE Val": mv["MAE"],
            "RMSE Val": mv["RMSE"],
            "R2 Val": mv["R2"]
        })

    df_resumen = pd.DataFrame(tabla_data)
    print(df_resumen.to_string(index=False))
    print("=" * 80 + "\n")

    return resultados

if __name__ == "__main__":
    ejecutar_experimento_1_2()