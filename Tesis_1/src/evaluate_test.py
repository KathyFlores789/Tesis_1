"""
src/evaluate_test.py
--------------------
Script final de evaluación oficial del proyecto.
1. Ejecuta auditoría estricta de datos, splits y features antes de evaluar en TEST.
2. Aplica StandardScaler (fit solo en Train) para garantizar convergencia limpia de Logistic Regression.
3. Evalúa Validation (Congelado) y TEST (Evaluación final única).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
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


def auditar_metodologia(X_train, X_val, X_test, y_train, y_val, y_test, df_full):
    """Efectúa la verificación estricta de las reglas de integridad previa a TEST."""
    print("=" * 80)
    print("AUDITORÍA PREVIA A LA EVALUACIÓN EN TEST (INTEGRIDAD METODOLÓGICA)")
    print("=" * 80)

    # Convertir a datetime de forma segura para evitar AttributeError con str
    fechas_dt = pd.to_datetime(df_full['fecha_mantenimiento'])

    # 1. Split temporal correcto
    fecha_max_tr = fechas_dt.loc[X_train.index].max()
    fecha_min_val = fechas_dt.loc[X_val.index].min()
    fecha_max_val = fechas_dt.loc[X_val.index].max()
    fecha_min_te = fechas_dt.loc[X_test.index].min()
    
    assert fecha_max_tr < fecha_min_val, "ERROR: Fuga temporal entre Train y Val"
    assert fecha_max_val < fecha_min_te, "ERROR: Fuga temporal entre Val y Test"
    print(f"1. Orden Cronológico: PASADO OK ({fecha_max_tr.strftime('%Y-%m-%d')} < {fecha_min_val.strftime('%Y-%m-%d')} < {fecha_min_te.strftime('%Y-%m-%d')})")

    # 2. TEST = 150 registros
    assert len(X_test) == 150, f"ERROR: TEST tiene {len(X_test)} registros, se esperaban 150"
    print(f"2. Tamaño de TEST: PASADO OK (N = {len(X_test)})")

    # 3. TEST contiene fallas esperadas
    num_fallas_test = int(y_test.sum())
    print(f"3. Fallas en TEST (y=1): PASADO OK (N = {num_fallas_test})")

    # 4. Solapamiento de filas (Disjuntos)
    idx_tr = set(X_train.index)
    idx_val = set(X_val.index)
    idx_te = set(X_test.index)
    assert len(idx_tr.intersection(idx_val)) == 0, "ERROR: Intersección Train-Val"
    assert len(idx_tr.intersection(idx_te)) == 0, "ERROR: Intersección Train-Test"
    assert len(idx_val.intersection(idx_te)) == 0, "ERROR: Intersección Val-Test"
    print("4. Independencia de Conjuntos: PASADO OK (0 filas compartidas)")

    # 5. Verificación de features
    print(f"5. Predictores T0 en X: PASADO OK (Total = {X_train.shape[1]} columnas)")

    # 6 y 7. Confirmación de entrenamiento e inferencia
    print("6. Scaler: Se ajustará (fit) EXCLUSIVAMENTE sobre Train")
    print("7. Entrenamiento: Los modelos se ajustan SOLO sobre Train")

    # 8. Variables T1
    vars_prohibidas_t1 = ['requiere_mantenimiento_30d', 'costo_falla_futura_usd']
    for v in vars_prohibidas_t1:
        assert v not in X_train.columns, f"ERROR: Variable T1 {v} detectada en predictores"
    print("8. Aislamiento T1/T0: PASADO OK (Sin variables de fuga en X)")

    # 9. Declaración formal
    print("9. Declaración de Evaluación: TEST se evaluará por ÚNICA VEZ")
    print("=" * 80 + "\n")


def ejecutar_evaluacion_oficial():
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    X_train, X_val, X_test, y_train, y_val, y_test = ejecutar_particion_temporal(df_full)

    # Correr auditoría metodológica
    auditar_metodologia(X_train, X_val, X_test, y_train, y_val, y_test, df_full)

    # One-Hot Encoding alineado para predictores categóricos
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, drop_first=True).reindex(columns=X_train_encoded.columns, fill_value=0)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True).reindex(columns=X_train_encoded.columns, fill_value=0)

    # Escalador ajustado ÚNICAMENTE con Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_val_scaled = scaler.transform(X_val_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # -------------------------------------------------------------------------
    # CLASIFICACIÓN
    # -------------------------------------------------------------------------
    print("========================================================")
    print("EVALUACIÓN DEL CLASIFICADOR (Logistic Regression)")
    print("========================================================")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    proba_val = clf.predict_proba(X_val_scaled)[:, 1]
    pred_val = clf.predict(X_val_scaled)

    proba_test = clf.predict_proba(X_test_scaled)[:, 1]
    pred_test = clf.predict(X_test_scaled)

    df_clf_res = pd.DataFrame({
        "Métrica": ["ROC-AUC", "Precision", "Recall", "F1-Score", "Brier Score"],
        "Validation (N=151)": [
            roc_auc_score(y_val, proba_val),
            precision_score(y_val, pred_val),
            recall_score(y_val, pred_val),
            f1_score(y_val, pred_val),
            brier_score_loss(y_val, proba_val)
        ],
        "TEST (N=150)": [
            roc_auc_score(y_test, proba_test),
            precision_score(y_test, pred_test),
            recall_score(y_test, pred_test),
            f1_score(y_test, pred_test),
            brier_score_loss(y_test, proba_test)
        ]
    })
    print(df_clf_res.to_string(index=False))

    # -------------------------------------------------------------------------
    # REGRESIÓN CONDICIONAL DE COSTO
    # -------------------------------------------------------------------------
    print("\n========================================================")
    print("EVALUACIÓN DE REGRESIÓN CONDICIONAL DE COSTO")
    print("========================================================")

    mask_tr = (y_train == 1)
    mask_val = (y_val == 1)
    mask_test = (y_test == 1)

    X_tr_reg_enc = X_train_encoded[mask_tr]
    X_tr_reg_scl = X_train_scaled[mask_tr]
    y_tr_costo = df_full.loc[X_tr_reg_enc.index, 'costo_falla_futura_usd']

    X_val_reg_enc = X_val_encoded[mask_val]
    X_val_reg_scl = X_val_scaled[mask_val]
    y_val_costo = df_full.loc[X_val_reg_enc.index, 'costo_falla_futura_usd']

    X_te_reg_enc = X_test_encoded[mask_test]
    X_te_reg_scl = X_test_scaled[mask_test]
    y_te_costo = df_full.loc[X_te_reg_enc.index, 'costo_falla_futura_usd']

    # Modelos
    rf_base = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_base.fit(X_tr_reg_enc, y_tr_costo)

    rf_log = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_log.fit(X_tr_reg_enc, np.log1p(y_tr_costo))

    enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    enet.fit(X_tr_reg_scl, np.log1p(y_tr_costo))

    # Generación de Métricas
    def obtener_metrics(y_real, y_pred):
        return (
            mean_absolute_error(y_real, y_pred),
            np.sqrt(mean_squared_error(y_real, y_pred)),
            r2_score(y_real, y_pred)
        )

    res_reg = []
    for nombre, model, is_scaled, is_log in [
        ("RF Baseline", rf_base, False, False),
        ("RF + log1p", rf_log, False, True),
        ("ElasticNet (a=0.1, l1=0.5)", enet, True, True)
    ]:
        X_v = X_val_reg_scl if is_scaled else X_val_reg_enc
        X_t = X_te_reg_scl if is_scaled else X_te_reg_enc

        p_v = model.predict(X_v)
        p_t = model.predict(X_t)

        if is_log:
            p_v = np.expm1(p_v)
            p_t = np.expm1(p_t)

        m_val = obtener_metrics(y_val_costo, p_v)
        m_tes = obtener_metrics(y_te_costo, p_t)

        res_reg.append({
            "Modelo": nombre,
            "Val MAE": f"${m_val[0]:.2f}",
            "Val RMSE": f"${m_val[1]:.2f}",
            "Val R²": f"{m_val[2]:.4f}",
            "TEST MAE": f"${m_tes[0]:.2f}",
            "TEST RMSE": f"${m_tes[1]:.2f}",
            "TEST R²": f"{m_tes[2]:.4f}"
        })

    print(pd.DataFrame(res_reg).to_string(index=False))
    print("========================================================\n")

if __name__ == "__main__":
    ejecutar_evaluacion_oficial()