"""
src/regenerar_resultados_oficiales.py
--------------------------------------
Re-ejecuta los modelos oficiales corrigiendo la convergencia técnica de la 
Regresión Logística (max_iter=5000) y congelando los resultados oficiales de Validation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, 
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import load_dataset
from src.preprocessing import aplicar_limpieza_base
from src.features import construir_features_t0
from src.split_data import ejecutar_particion_temporal

def evaluar_oficiales():
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    X_train, X_val, X_test, y_train, y_val, y_test = ejecutar_particion_temporal(df_full)

    # Hot-Encoding alineado para predictores categóricos
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, drop_first=True).reindex(columns=X_train_encoded.columns, fill_value=0)

    print("=" * 80)
    print("TAREA 1: CLASIFICADOR - VERIFICACIÓN DE CONVERGENCIA EN VALIDATION")
    print("=" * 80)

    # 1. CLASIFICADOR: Logistic Regression (max_iter aumentado a 5000 para convergencia estricta)
    clf = LogisticRegression(max_iter=5000, random_state=42)
    clf.fit(X_train_encoded, y_train)
    y_val_pred_proba = clf.predict_proba(X_val_encoded)[:, 1]
    auc_val = roc_auc_score(y_val, y_val_pred_proba)
    
    print(f"Logistic Regression (max_iter=5000):")
    print(f"  - ROC-AUC Validation: {auc_val:.4f}")

    print("\n" + "=" * 80)
    print("TAREA 2: REGRESIÓN CONDICIONAL - MODELOS CONGELADOS EN VALIDATION")
    print("=" * 80)

    # Subconjuntos condicionales a falla (y = 1)
    mask_train_falla = (y_train == 1)
    mask_val_falla = (y_val == 1)

    X_train_reg = X_train_encoded[mask_train_falla]
    y_train_costo = df_full.loc[X_train_reg.index, 'costo_falla_futura_usd']

    X_val_reg = X_val_encoded[mask_val_falla]
    y_val_costo = df_full.loc[X_val_reg.index, 'costo_falla_futura_usd']

    # 2. BASELINE: RF Original
    rf_base = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_base.fit(X_train_reg, y_train_costo)
    pred_rf_base = rf_base.predict(X_val_reg)
    mae_base = mean_absolute_error(y_val_costo, pred_rf_base)
    rmse_base = np.sqrt(mean_squared_error(y_val_costo, pred_rf_base))
    r2_base = r2_score(y_val_costo, pred_rf_base)
    print(f"1. RF Baseline (original):")
    print(f"   MAE: ${mae_base:.2f} | RMSE: ${rmse_base:.2f} | R²: {r2_base:.4f}")

    # 3. COMPAÑERO / CANDIDATO A: RF + log1p (Mejor RMSE y R²)
    rf_log = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_log.fit(X_train_reg, np.log1p(y_train_costo))
    pred_rf_log = np.expm1(rf_log.predict(X_val_reg))
    mae_rf_log = mean_absolute_error(y_val_costo, pred_rf_log)
    rmse_rf_log = np.sqrt(mean_squared_error(y_val_costo, pred_rf_log))
    r2_rf_log = r2_score(y_val_costo, pred_rf_log)
    print(f"\n2. RF + log1p (Seleccionado por menor RMSE y mayor R²):")
    print(f"   MAE: ${mae_rf_log:.2f} | RMSE: ${rmse_rf_log:.2f} | R²: {r2_rf_log:.4f}")

    # 4. CANDIDATO B: ElasticNet (alpha=0.1, l1_ratio=0.5 - Mejor MAE)
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    enet.fit(X_train_reg, np.log1p(y_train_costo))
    pred_enet = np.expm1(enet.predict(X_val_reg))
    mae_enet = mean_absolute_error(y_val_costo, pred_enet)
    rmse_enet = np.sqrt(mean_squared_error(y_val_costo, pred_enet))
    r2_enet = r2_score(y_val_costo, pred_enet)
    print(f"\n3. ElasticNet (alpha=0.1, l1_ratio=0.5) (Seleccionado por menor MAE):")
    print(f"   MAE: ${mae_enet:.2f} | RMSE: ${rmse_enet:.2f} | R²: {r2_enet:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    evaluar_oficiales()