"""
src/explicabilidad_shap.py
--------------------------
Calcula y exporta la explicabilidad global y local (SHAP) sobre los modelos congelados.
Genera explicaciones locales (Top-3 factores de riesgo) por vehículo para el motor generativo.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import load_dataset
from src.preprocessing import aplicar_limpieza_base
from src.features import construir_features_t0
from src.split_data import ejecutar_particion_temporal

def generar_explicabilidad_shap():
    print("=" * 80)
    print("ETAPA 2: EXPLICABILIDAD (XAI / SHAP) - EXPLICACIONES GLOBALES Y LOCALES")
    print("=" * 80)

    # 1. Cargar y procesar datos
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    X_train, X_val, X_test, y_train, y_val, y_test = ejecutar_particion_temporal(df_full)

    # Hot-Encoding alineado
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, drop_first=True).reindex(columns=X_train_encoded.columns, fill_value=0)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True).reindex(columns=X_train_encoded.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_val_scaled = scaler.transform(X_val_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    feature_names = X_train_encoded.columns.tolist()

    # -------------------------------------------------------------------------
    # A. SHAP CLASIFICADOR (Logistic Regression)
    # -------------------------------------------------------------------------
    print("\n[1/2] Calculando SHAP para Clasificador de Riesgo...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # background reducido a 100 para estabilidad
    explainer_clf = shap.LinearExplainer(clf, X_train_scaled[:100], feature_names=feature_names)
    shap_values_clf = explainer_clf(X_test_scaled)

    # Global
    mean_abs_shap_clf = np.abs(shap_values_clf.values).mean(axis=0)
    df_importance_clf = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Absolute_SHAP': mean_abs_shap_clf,
        'Coeficiente': clf.coef_[0]
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False)

    print("\nTop 10 Variables más influyentes en el Riesgo de Falla (Global):")
    print(df_importance_clf.head(10).to_string(index=False))

    # -------------------------------------------------------------------------
    # B. SHAP REGRESIÓN DE COSTO (ElasticNet con alpha=0.01 / RF)
    # -------------------------------------------------------------------------
    print("\n[2/2] Calculando SHAP para Regresión de Costo Condicional...")
    mask_tr = (y_train == 1)
    mask_te = (y_test == 1)

    X_tr_reg_scl = X_train_scaled[mask_tr]
    X_te_reg_scl = X_test_scaled[mask_te]
    y_tr_costo = df_full.loc[X_train_encoded[mask_tr].index, 'costo_falla_futura_usd']

    # Ajustamos alpha=0.01 para evitar que Lasso suprima todas las variables
    enet_cost = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    enet_cost.fit(X_tr_reg_scl, np.log1p(y_tr_costo))

    explainer_reg = shap.LinearExplainer(enet_cost, X_tr_reg_scl[:100], feature_names=feature_names)
    shap_values_reg = explainer_reg(X_te_reg_scl)

    mean_abs_shap_reg = np.abs(shap_values_reg.values).mean(axis=0)
    df_importance_reg = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Absolute_SHAP': mean_abs_shap_reg,
        'Coeficiente': enet_cost.coef_
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False)

    print("\nTop 10 Variables más influyentes en el Costo Condicional (Global):")
    print(df_importance_reg.head(10).to_string(index=False))

    # -------------------------------------------------------------------------
    # C. GENERACIÓN DE EXPLICACIONES LOCALES PARA EL MOTOR GENERATIVO
    # -------------------------------------------------------------------------
    print("\n[3/3] Generando estructura de Explicabilidad Local (Top-3 Factores por Vehículo)...")
    
    # Ejemplo de extracción local sobre los primeros 5 vehículos de TEST
    explicaciones_locales = []
    probabilidades = clf.predict_proba(X_test_scaled)[:, 1]

    for idx in range(min(5, len(X_test_encoded))):
        shaps_instancia = shap_values_clf.values[idx]
        # Ordenar factores que MÁS AUMENTAN el riesgo (valores SHAP positivos)
        pos_indices = np.argsort(shaps_instancia)[::-1]
        
        top_factores = []
        for i in pos_indices:
            if shaps_instancia[i] > 0 and len(top_factores) < 3:
                top_factores.append(f"{feature_names[i]} (+{shaps_instancia[i]:.3f})")
        
        vehiculo_id = df_full.loc[X_test_encoded.index[idx], 'vehiculo_id']
        explicaciones_locales.append({
            'vehiculo_id': vehiculo_id,
            'probabilidad_falla': round(float(probabilidades[idx]), 4),
            'top_factores_riesgo': top_factores
        })

    df_local_sample = pd.DataFrame(explicaciones_locales)
    print("\nMuestra de Explicaciones Locales para el Prompt del LLM:")
    print(df_local_sample.to_string(index=False))

    # Archivar reportes
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    df_importance_clf.to_csv(reports_dir / "shap_importance_clasificacion.csv", index=False)
    df_importance_reg.to_csv(reports_dir / "shap_importance_regresion.csv", index=False)
    df_local_sample.to_csv(reports_dir / "muestra_explicaciones_locales.csv", index=False)

    print("\n" + "=" * 80)
    print("ETAPA 2 FINALIZADA: RESULTADOS DE EXPLICABILIDAD GUARDADOS EN 'reports/'.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    generar_explicabilidad_shap()