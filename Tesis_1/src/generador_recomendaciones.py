"""
src/generador_recomendaciones.py
--------------------------------
Motor de Recomendaciones Preventivas + IA Generativa Explicable.
Incluye deduplicación de acciones y lenguaje financiero cauteloso.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Asegurar path del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import load_dataset
from src.preprocessing import aplicar_limpieza_base
from src.features import construir_features_t0
from src.split_data import ejecutar_particion_temporal


class GeneradorRecomendaciones:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.reg = RandomForestRegressor(n_estimators=100, random_state=42)
        self.explainer = None
        self.feature_names = []

    def fit_modelos(self, X_train: pd.DataFrame, y_train: pd.Series, df_full: pd.DataFrame):
        X_train_enc = pd.get_dummies(X_train, drop_first=True)
        self.clf.fit(X_train_enc, y_train)

        # Usar y_train directamente para identificar casos de falla
        mask_falla = (y_train == 1)
        if mask_falla.sum() > 5:
            X_falla = X_train_enc[mask_falla]
            y_costo = df_full.loc[X_train.index[mask_falla], 'costo_mantenimiento_actual_usd']
            self.reg.fit(X_falla, y_costo)
        else:
            self.reg.fit(X_train_enc, df_full.loc[X_train.index, 'costo_mantenimiento_actual_usd'])

        self.explainer = shap.TreeExplainer(self.clf)
        self.feature_names = X_train_enc.columns.tolist()

    def generar_prompt_contextual(self, datos_vehiculo: Dict[str, Any], row_encoded: pd.DataFrame) -> Dict[str, Any]:
        prob_falla = float(self.clf.predict_proba(row_encoded)[0, 1])
        
        if prob_falla >= 0.70:
            nivel_riesgo = "CRÍTICO"
        elif prob_falla >= 0.40:
            nivel_riesgo = "ALTO"
        elif prob_falla >= 0.20:
            nivel_riesgo = "MEDIO"
        else:
            nivel_riesgo = "BAJO"

        costo_est = float(self.reg.predict(row_encoded)[0])

        shap_values = self.explainer.shap_values(row_encoded)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        df_shap = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': sv
        }).sort_values('shap_value', ascending=False)

        top_factores = df_shap.head(3).to_dict(orient='records')

        return {
            "vehiculo_id": datos_vehiculo.get("vehiculo_id", "N/A"),
            "marca_modelo": f"{datos_vehiculo.get('marca', '')} {datos_vehiculo.get('modelo', '')}".strip(),
            "anio": datos_vehiculo.get("anio", "N/A"),
            "zona_quito": datos_vehiculo.get("zona_operacion_quito", "N/A"),
            "probabilidad_falla": prob_falla,
            "nivel_riesgo": nivel_riesgo,
            "costo_estimado_falla_usd": round(costo_est, 2),
            "top_factores_riesgo": top_factores
        }

    def construir_recomendacion_explicable(self, contexto: Dict[str, Any]) -> str:
        factores_txt = []
        acciones_raw = []

        for f in contexto["top_factores_riesgo"]:
            feat = f["feature"]
            val_shap = f["shap_value"]

            if "km_desde_ultimo_mantenimiento" in feat:
                factores_txt.append(f"Exceso de recorrido desde el último servicio (+{val_shap:.4f} impacto en riesgo).")
                acciones_raw.append("Agendar mantenimiento preventivo inmediato de fluidos y filtros.")
            elif "tiene_codigo_obd_activo" in feat:
                factores_txt.append(f"Presencia de un código de falla OBD activo (+{val_shap:.4f} impacto en riesgo).")
                acciones_raw.append("Efectuar escaneo telemétrico OBD-II para diagnosticar el código registrado.")
            elif "frenadas_bruscas" in feat:
                factores_txt.append(f"Alta frecuencia de frenadas bruscas detectada (+{val_shap:.4f} impacto).")
                acciones_raw.append("Revisar el estado de pastillas, discos de freno y hábitos de conducción.")
            elif "temperatura_promedio_motor" in feat:
                factores_txt.append(f"Registro de temperatura promedio de motor elevada (+{val_shap:.4f} impacto).")
                acciones_raw.append("Inspeccionar el sistema de refrigeración, líquido refrigerante y termostato.")
            elif "presion_aceite" in feat:
                factores_txt.append(f"Anomalía o variación en la presión de aceite registrada (+{val_shap:.4f} impacto).")
                acciones_raw.append("Comprobar el nivel de aceite del motor y verificar bomba/sensor de presión.")
            else:
                nombre_limpio = feat.replace("_", " ")
                factores_txt.append(f"Variable de desgaste operacional: {nombre_limpio} (+{val_shap:.4f} impacto).")
                acciones_raw.append("Realizar inspección técnica general de seguridad vehicular.")

        # 1. DEDUPLICACIÓN DE ACCIONES
        acciones_unicas = []
        for ac in acciones_raw:
            if ac not in acciones_unicas:
                acciones_unicas.append(ac)

        # 2. DIVERSIFICACIÓN SI QUEDAN ACCIONES REPETIDAS/INSUFICIENTES
        if len(acciones_unicas) < 2:
            if contexto["nivel_riesgo"] in ["CRÍTICO", "ALTO"]:
                acciones_unicas.append("Programar revisión prioritaria en taller antes de la siguiente ruta de operación.")
            else:
                acciones_unicas.append("Mantener el plan de monitoreo rutinario y verificar presiones de neumáticos.")

        if len(acciones_unicas) < 3:
            acciones_unicas.append("Verificar niveles de líquidos operativos (frenos, refrigerante y dirección).")

        # 3. CONSTRUCCIÓN DEL TEXTO DEL REPORTE
        reporte = f"""
================================================================================
REPORTE GENERATIVO DE RECOMENDACIÓN PREVENTIVA EXPLICABLE
VEHÍCULO: {contexto['vehiculo_id']} ({contexto['marca_modelo']} - Año {contexto['anio']}) | OPERACIÓN: {contexto['zona_quito']}
================================================================================

1. DIAGNÓSTICO DE RIESGO PREDICTIVO
   • Nivel de Riesgo Operacional : {contexto['nivel_riesgo']} ({contexto['probabilidad_falla']*100:.1f}% de probabilidad de falla a 30 días)
   • Estimación de Costo Futuro  : ${contexto['costo_estimado_falla_usd']:,.2f} USD (Costo promedio estimado si ocurre la falla)

2. EVIDENCIA Y EXPLICABILIDAD TÉCNICA (SHAP Analysis)
   El análisis de inteligencia artificial identifica que el riesgo se debe principalmente a:
"""
        for f_txt in factores_txt:
            reporte += f"  • {f_txt}\n"

        reporte += "\n3. PLAN DE ACCIÓN PREVENTIVA RECOMENDADO\n"
        for i, ac in enumerate(acciones_unicas[:3], 1):
            reporte += f"   {i}. {ac}\n"

        # 4. LENGUAJE FINANCIERO PRUDENTE
        reporte += f"""
4. EVALUACIÓN FINANCIERA DEL RIESGO
   Se estima un costo contingente de ${contexto['costo_estimado_falla_usd']:,.2f} USD en caso de presentarse 
   una falla no atendida. Un mantenimiento preventivo oportuno (~$80 - $150 USD) permite mitigar 
   la exposición a esta contingencia.
================================================================================
"""
        return reporte


# BLOQUE DE EJECUCIÓN DIRECTA DESDE TERMINAL
if __name__ == "__main__":
    print("\nEjecutando prueba del Motor de Recomendaciones con deduplicación...")
    
    df_raw = load_dataset()
    df_clean = aplicar_limpieza_base(df_raw)
    df_full = construir_features_t0(df_clean)

    X_train, X_val, X_test, y_train, y_val, y_test = ejecutar_particion_temporal(df_full)

    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True).reindex(columns=X_train_encoded.columns, fill_value=0)

    engine = GeneradorRecomendaciones()
    engine.fit_modelos(X_train, y_train, df_full)

    sample_ids = ['UIO-0676', 'UIO-0379']
    for vid in sample_ids:
        idx_test = df_full[df_full['vehiculo_id'] == vid].index[0]
        datos_v = df_full.loc[idx_test].to_dict()
        X_enc_row = X_test_encoded.loc[[idx_test]]

        contexto = engine.generar_prompt_contextual(datos_v, X_enc_row)
        reporte = engine.construir_recomendacion_explicable(contexto)
        print(reporte)