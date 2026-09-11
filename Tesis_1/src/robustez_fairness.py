"""
src/robustez_fairness.py
------------------------
Auditoría de Robustez y Fairness sobre los modelos congelados.
Evalúa disparidades de desempeño por subgrupos (Tipo de Vehículo, Combustible, Zona Quito)
y prueba de sensibilidad ante ruido de sensores de telemetría.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import load_dataset
from src.preprocessing import aplicar_limpieza_base
from src.features import construir_features_t0
from src.split_data import ejecutar_particion_temporal

def auditar_robustez_y_fairness():
    print("=" * 80)
    print("ETAPA 3: AUDITORÍA DE ROBUSTEZ Y FAIRNESS (EQUIDAD)")
    print("=" * 80)

    # 1. Cargar y preparar datos
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    X_train, X_val, X_test, y_train, y_val, y_test = ejecutar_particion_temporal(df_full)

    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True).reindex(columns=X_train_encoded.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # Entrenar clasificador congelado
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred_test = clf.predict(X_test_scaled)
    y_prob_test = clf.predict_proba(X_test_scaled)[:, 1]

    # -------------------------------------------------------------------------
    # A. AUDITORÍA DE FAIRNESS / EQUIDAD POR SUBGRUPOS (En TEST)
    # -------------------------------------------------------------------------
    print("\n[1/2] Evaluando Equidad (Fairness) por Subgrupos en TEST...")
    
    # Recuperar atributos sensibles de X_test original
    df_test_meta = X_test.copy()
    df_test_meta['target_falla'] = y_test
    df_test_meta['pred_falla'] = y_pred_test
    df_test_meta['prob_falla'] = y_prob_test

    def evaluar_subgrupo(columna):
        resultados = []
        for grupo, df_g in df_test_meta.groupby(columna):
            if len(df_g) < 5:  # Omitir grupos demasiado pequeños para métricas estables
                continue
            acc = accuracy_score(df_g['target_falla'], df_g['pred_falla'])
            f1 = f1_score(df_g['target_falla'], df_g['pred_falla'], zero_division=0)
            
            # ROC-AUC solo si hay ambas clases presentes en el subgrupo
            if len(df_g['target_falla'].unique()) > 1:
                auc = roc_auc_score(df_g['target_falla'], df_g['prob_falla'])
            else:
                auc = np.nan

            resultados.append({
                'Atributo': columna,
                'Subgrupo': grupo,
                'Muestras': len(df_g),
                'Tasa_Falla_Real': round(df_g['target_falla'].mean(), 3),
                'Accuracy': round(acc, 3),
                'F1_Score': round(f1, 3),
                'ROC_AUC': round(auc, 3) if not np.isnan(auc) else "N/A"
            })
        return pd.DataFrame(resultados)

    df_fairness_tipo = evaluar_subgrupo('tipo_vehiculo')
    df_fairness_comb = evaluar_subgrupo('combustible')
    df_fairness_zona = evaluar_subgrupo('zona_operacion_quito')

    df_fairness_total = pd.concat([df_fairness_tipo, df_fairness_comb, df_fairness_zona], ignore_index=True)
    print("\nReporte de Desempeño por Subgrupos (Fairness Audit):")
    print(df_fairness_total.to_string(index=False))

    # -------------------------------------------------------------------------
    # B. AUDITORÍA DE ROBUSTEZ (Prueba de Estrés con Ruido de Sensores)
    # -------------------------------------------------------------------------
    print("\n[2/2] Evaluando Robustez ante Ruido de Sensores Telemétricos...")
    
    np.random.seed(42)
    # Perturbación: agregar 5% de ruido gaussiano a características continuas en TEST
    X_test_scaled_noisy = X_test_scaled.copy()
    num_cols_indices = [i for i, col in enumerate(X_train_encoded.columns) if not col.startswith(('marca_', 'modelo_', 'tipo_', 'combustible_', 'transmision_', 'zona_'))]
    
    noise = np.random.normal(0, 0.05, size=X_test_scaled_noisy[:, num_cols_indices].shape)
    X_test_scaled_noisy[:, num_cols_indices] += noise

    y_prob_noisy = clf.predict_proba(X_test_scaled_noisy)[:, 1]
    y_pred_noisy = clf.predict(X_test_scaled_noisy)

    auc_orig = roc_auc_score(y_test, y_prob_test)
    auc_noisy = roc_auc_score(y_test, y_prob_noisy)
    f1_orig = f1_score(y_test, y_pred_test)
    f1_noisy = f1_score(y_test, y_pred_noisy)

    df_robustness = pd.DataFrame([{
        'Escenario': 'Original TEST',
        'ROC_AUC': round(auc_orig, 4),
        'F1_Score': round(f1_orig, 4)
    }, {
        'Escenario': 'Con Ruido Telemetría (±5%)',
        'ROC_AUC': round(auc_noisy, 4),
        'F1_Score': round(f1_noisy, 4)
    }, {
        'Escenario': 'Diferencia (Degradación)',
        'ROC_AUC': round(auc_noisy - auc_orig, 4),
        'F1_Score': round(f1_noisy - f1_orig, 4)
    }])

    print("\nResultados de la Prueba de Robustez / Sensibilidad:")
    print(df_robustness.to_string(index=False))

    # Exportar reportes para la Tesis
    reports_dir = ROOT_DIR / "reports"
    df_fairness_total.to_csv(reports_dir / "fairness_subgrupos_audit.csv", index=False)
    df_robustness.to_csv(reports_dir / "robustez_ruido_sensores.csv", index=False)

    print("\n" + "=" * 80)
    print("ETAPA 3 FINALIZADA: AUDITORÍAS REGISTRADAS EN 'reports/'.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    auditar_robustez_y_fairness()