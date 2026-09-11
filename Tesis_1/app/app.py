"""
app.py
------
Dashboard Robusto HTML/Streamlit para el Sistema de Mantenimiento Predictivo.
Compatible con Python 3.14 en Windows.
"""

import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

# Configuración básica
st.set_page_config(page_title="Mantenimiento Predictivo", page_icon="🚗")

# Título Principal
st.write("# 🚗 Sistema Predictivo de Mantenimiento Vehicular")
st.write("### Plataforma de Diagnóstico para Flotas en Quito")
st.write("---")

# Verificar comunicación con FastAPI
api_online = False
try:
    res_status = requests.get(f"{API_URL}/", timeout=3)
    if res_status.status_code == 200:
        st.sidebar.write("🟢 **API Conectada**")
        api_online = True
    else:
        st.sidebar.write("🔴 **Error en API**")
except Exception:
    st.sidebar.write("🔴 **API Offline** (Inicia Uvicorn en el puerto 8000)")

st.sidebar.write("---")
st.sidebar.write("### Selección de Vehículo")

# Cargar lista de vehículos
vehiculos_list = []
if api_online:
    try:
        res_v = requests.get(f"{API_URL}/vehiculos", timeout=3)
        if res_v.status_code == 200:
            vehiculos_list = res_v.json().get("vehiculos", [])
    except Exception:
        pass

if vehiculos_list:
    selected_id = st.sidebar.selectbox("ID de Vehículo:", vehiculos_list)
else:
    selected_id = st.sidebar.text_input("Ingrese ID de Vehículo:", value="UIO-0676")

analizar = st.sidebar.button("🔍 Analizar Riesgo")

if analizar:
    if not api_online:
        st.error("La API de FastAPI no está corriendo en el puerto 8000.")
    else:
        with st.spinner("Consultando modelos predictivos..."):
            try:
                payload = {"vehiculo_id": selected_id}
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    ctx = data["contexto"]
                    reporte = data["reporte_texto"]

                    # Tarjetas visuales en HTML directo (A prueba de pantalla negra)
                    html_cards = f"""
                    <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                        <div style="background-color: #1e293b; color: white; padding: 15px; border-radius: 8px; flex: 1; text-align: center;">
                            <h4 style="margin:0; color: #94a3b8;">Nivel de Riesgo</h4>
                            <h2 style="margin:5px 0; color: #f87171;">{ctx['nivel_riesgo']}</h2>
                            <p style="margin:0; font-size: 0.9em;">Probabilidad: {ctx['probabilidad_falla']*100:.1f}%</p>
                        </div>
                        <div style="background-color: #1e293b; color: white; padding: 15px; border-radius: 8px; flex: 1; text-align: center;">
                            <h4 style="margin:0; color: #94a3b8;">Costo Estimado</h4>
                            <h2 style="margin:5px 0; color: #38bdf8;">${ctx['costo_estimado_falla_usd']:,.2f} USD</h2>
                            <p style="margin:0; font-size: 0.9em;">Contingencia</p>
                        </div>
                        <div style="background-color: #1e293b; color: white; padding: 15px; border-radius: 8px; flex: 1; text-align: center;">
                            <h4 style="margin:0; color: #94a3b8;">Zona Operativa</h4>
                            <h2 style="margin:5px 0; color: #4ade80;">{ctx['zona_quito']}</h2>
                            <p style="margin:0; font-size: 0.9em;">Sector Quito</p>
                        </div>
                    </div>
                    """
                    st.markdown(html_cards, unsafe_allow_html=True)

                    st.write("### 📄 Reporte Explicable del Vehículo")
                    st.code(reporte, language="text")

                    st.write("### 📊 Top Factores de Riesgo (SHAP)")
                    df_factors = pd.DataFrame(ctx["top_factores_riesgo"])
                    st.table(df_factors)

                else:
                    st.error("Error en la respuesta de la API.")
            except Exception as e:
                st.error(f"Error de conexión: {e}")
else:
    st.info("👈 Selecciona un vehículo en el menú lateral y haz clic en **Analizar Riesgo**.")