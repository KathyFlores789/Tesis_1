"""
src/api.py
----------
Backend REST con FastAPI optimizado para Dashboard de Tesis.
Inferencia de modelos congelados (Clasificador RF + Regresor ElasticNet) 
e Integración de Explicabilidad Prescriptiva XAI.
"""

import sys
import json
import joblib
from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

app = FastAPI(
    title="API Mantenimiento Predictivo - Vehículos Quito",
    description="Servicio de inferencia y explicabilidad en tiempo real.",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = ROOT_DIR / "models"
RAW_DIR = ROOT_DIR / "data" / "raw"

clf = None
reg = None
feature_names = []
df_test_sample = pd.DataFrame()

# Helper para identificar el nombre real del estimador dentro de un Pipeline
def obtener_nombre_modelo(modelo):
    if modelo is None:
        return "Ninguno"
    if hasattr(modelo, "named_steps") and "model" in modelo.named_steps:
        return modelo.named_steps["model"].__class__.__name__
    elif hasattr(modelo, "named_steps") and "regressor" in modelo.named_steps:
        return modelo.named_steps["regressor"].__class__.__name__
    elif hasattr(modelo, "named_steps") and "classifier" in modelo.named_steps:
        return modelo.named_steps["classifier"].__class__.__name__
    return modelo.__class__.__name__

try:
    # 1. Carga del Clasificador
    if (MODELS_DIR / "best_classifier.joblib").exists():
        clf = joblib.load(MODELS_DIR / "best_classifier.joblib")
    elif (MODELS_DIR / "classifier_rf.joblib").exists():
        clf = joblib.load(MODELS_DIR / "classifier_rf.joblib")

    # 2. Carga del Regresor de Costos (Búsqueda flexible)
    posibles_regresores = [
        "best_regressor.joblib",
        "regressor_elasticnet.joblib",
        "elasticnet_regressor.joblib",
        "regressor_rf.joblib",
        "best_model_reg.joblib"
    ]
    
    for f_reg in posibles_regresores:
        if (MODELS_DIR / f_reg).exists():
            reg = joblib.load(MODELS_DIR / f_reg)
            break

    # Si no lo encontró por nombre exacto, busca cualquier joblib con 'reg' o 'elastic'
    if reg is None and MODELS_DIR.exists():
        for file in MODELS_DIR.glob("*.joblib"):
            fname = file.name.lower()
            if ("reg" in fname or "elastic" in fname or "cost" in fname) and "class" not in fname:
                reg = joblib.load(file)
                break

    # 3. Metadatos del Pipeline
    if (MODELS_DIR / "pipeline_metadata.json").exists():
        with open(MODELS_DIR / "pipeline_metadata.json", "r", encoding="utf-8") as f:
            feature_names = json.load(f).get("feature_names", [])

    # 4. Dataset de Vehículos de Quito
    if (RAW_DIR / "datos_mantenimiento_quito.csv").exists():
        df_test_sample = pd.read_csv(RAW_DIR / "datos_mantenimiento_quito.csv")
    elif (MODELS_DIR / "test_vehicles_sample.csv").exists():
        df_test_sample = pd.read_csv(MODELS_DIR / "test_vehicles_sample.csv")

except Exception as e:
    print(f"Error al cargar artefactos: {e}")

class VehiculoInput(BaseModel):
    vehiculo_id: str

@app.get("/")
def estado_api():
    return {
        "status": "ONLINE",
        "clasificador_cargado": obtener_nombre_modelo(clf),
        "regresor_costo_cargado": obtener_nombre_modelo(reg)
    }

@app.get("/vehiculos")
def obtener_vehiculos():
    if not df_test_sample.empty and "vehiculo_id" in df_test_sample.columns:
        ids = df_test_sample["vehiculo_id"].astype(str).unique().tolist()
    else:
        ids = ["UIO-0001", "UIO-0002"]
    return {"total": len(ids), "vehiculos": ids}

@app.post("/predict")
def predecir_falla(datos: VehiculoInput):
    v_id = datos.vehiculo_id
    raw_dict = {}

    if not df_test_sample.empty and "vehiculo_id" in df_test_sample.columns:
        coincidencia = df_test_sample[df_test_sample["vehiculo_id"].astype(str) == str(v_id)]
        if not coincidencia.empty:
            raw_dict = coincidencia.iloc[0].to_dict()

    marca = raw_dict.get("marca", "Vehículo")
    modelo = raw_dict.get("modelo", "")
    anio = raw_dict.get("anio", "2021")
    combustible = raw_dict.get("combustible", "Gasolina")
    encabezado_detallado = f"{v_id} - {marca} {modelo} {anio} ({combustible})".strip()

    km_actual = float(raw_dict.get("kilometraje_acumulado_km", 0))
    dias_mantenimiento = int(raw_dict.get("dias_desde_ultimo_mantenimiento", 0))
    obd_flag = int(raw_dict.get("tiene_codigo_obd_activo", 0))
    codigo_obd = "ACTIVO ⚠️ (Código de falla detectado)" if obd_flag == 1 else "INACTIVO 🟢 (Sin códigos activos)"

    sistema_afectado = str(raw_dict.get("sistema_intervenido_actual", "Sistema de Motor y Transmisión"))
    repuesto_sugerido = str(raw_dict.get("repuesto_cambiado_actual", "Filtro y Aceite"))
    zona = str(raw_dict.get("zona_operacion_quito", "Quito Centro-Norte"))

    df_single = pd.DataFrame([raw_dict]).drop(columns=["vehiculo_id"], errors="ignore")
    df_encoded = pd.get_dummies(df_single, drop_first=True)
    if feature_names:
        df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)

    # 1. Probabilidad de Falla
    if clf is not None:
        try:
            prob_falla = float(clf.predict_proba(df_encoded)[0, 1])
        except Exception:
            prob_falla = 0.45
    else:
        prob_falla = 0.45

    # 2. Inferencia de Costo
    if reg is not None:
        try:
            costo_est = float(reg.predict(df_encoded)[0])
            if costo_est < 0:
                costo_est = float(raw_dict.get("costo_mantenimiento_actual_usd", 80.0))
        except Exception:
            costo_est = float(raw_dict.get("costo_mantenimiento_actual_usd", 120.0))
    else:
        costo_est = float(raw_dict.get("costo_mantenimiento_actual_usd", 120.0))

    if prob_falla >= 0.70:
        nivel_riesgo = "CRÍTICO"
        dias_accion = "en las próximas 48 horas"
    elif prob_falla >= 0.40:
        nivel_riesgo = "ALTO"
        dias_accion = "en los próximos 5 días"
    elif prob_falla >= 0.20:
        nivel_riesgo = "MEDIO"
        dias_accion = "en los próximos 15 días"
    else:
        nivel_riesgo = "BAJO"
        dias_accion = "en el próximo chequeo rutinario"

    costo_formateado = round(costo_est, 2)

    plan_accion_directo = (
        f"🛠️ PLAN DE ACCIÓN: Programar la revisión e inspección de '{repuesto_sugerido}' "
        f"asociado al {sistema_afectado} {dias_accion} para evitar un gasto contingente "
        f"estimado de ${costo_formateado} USD por parada no programada."
    )

    top_factores = [
        {"factor": "Ficha Técnica", "valor": f"{km_actual:,.0f} km | {dias_mantenimiento} días sin taller"},
        {"factor": "Diagnóstico OBD", "valor": codigo_obd},
        {"factor": "Sistema en Riesgo", "valor": sistema_afectado},
        {"factor": "Repuesto Crítico", "valor": repuesto_sugerido}
    ]

    reporte_texto = (
        f"La unidad **{encabezado_detallado}** que opera en la zona **{zona}** registra un nivel de riesgo "
        f"**{nivel_riesgo}** con una probabilidad de falla del **{round(prob_falla * 100, 1)}%**.\n\n"
        f"**Ficha de Operación:** {km_actual:,.0f} km recorridos, {dias_mantenimiento} días transcurridos desde el último mantenimiento. "
        f"Estado del scanner OBD: {codigo_obd}.\n\n"
        f"**Diagnóstico de Falla:** Amenaza detectada principalmente en el **{sistema_afectado}**.\n\n"
        f"{plan_accion_directo}"
    )

    contexto = {
        "vehiculo_id": encabezado_detallado,
        "marca_modelo": f"{marca} {modelo}",
        "anio": anio,
        "zona_quito": zona,
        "probabilidad_falla": round(prob_falla, 4),
        "nivel_riesgo": nivel_riesgo,
        "costo_estimado_falla_usd": costo_formateado,
        "top_factores_riesgo": top_factores
    }

    return {
        "contexto": contexto,
        "reporte_texto": reporte_texto
    }