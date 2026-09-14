from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Sistema Predictivo de Mantenimiento Vehicular",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.explicabilidad_shap_v2 import explicar_prediccion
from src.generador_ia_v2 import generar_recomendacion_ia


DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "datos_mantenimiento_quito.csv"
)

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "classifier_v2.joblib"
)

REGRESSOR_PATH = (
    ROOT_DIR
    / "models"
    / "regressor_v2.joblib"
)


# ============================================================
# CARGA DE DATOS Y MODELO
# ============================================================

@st.cache_data
def cargar_datos():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def cargar_regresor():
    return joblib.load(REGRESSOR_PATH)


df = cargar_datos()
modelo = cargar_modelo()
regresor = cargar_regresor()


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

FEATURES = [
    "marca",
    "modelo",
    "anio",
    "tipo_vehiculo",
    "combustible",
    "cilindraje_l",
    "transmision",
    "kilometraje_acumulado_km",
    "km_desde_ultimo_mantenimiento",
    "dias_desde_ultimo_mantenimiento",
    "temperatura_promedio_motor_c",
    "presion_aceite_kpa",
    "frenadas_bruscas_por_100km",
    "tiempo_ralenti_min_dia",
    "altitud_operacion_m",
    "zona_operacion_quito",
    "tiene_codigo_obd_activo",
    "tipo_mantenimiento_actual",
    "sistema_intervenido_actual",
    "repuesto_cambiado_actual",
    "costo_mantenimiento_actual_usd"
]

UMBRAL = 0.50


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def generar_siguiente_id(dataframe):
    ids = (
        dataframe["vehiculo_id"]
        .astype(str)
        .str.extract(r"UIO-(\d+)", expand=False)
        .dropna()
        .astype(int)
    )

    if ids.empty:
        siguiente = 1
    else:
        siguiente = ids.max() + 1

    return f"UIO-{siguiente:04d}"


def predecir_riesgo(datos_entrada):
    probabilidad = modelo.predict_proba(datos_entrada)[0][1]
    prediccion = int(probabilidad >= UMBRAL)
    nivel_riesgo = "ALTO" if prediccion == 1 else "BAJO"

    return probabilidad, prediccion, nivel_riesgo


def estimar_costo(datos_entrada):
    """
    El regresor fue entrenado sobre log1p(costo), por lo que
    su salida se convierte nuevamente a USD mediante expm1().
    """
    pred_log = float(regresor.predict(datos_entrada)[0])
    costo_usd = float(np.expm1(pred_log))
    return max(costo_usd, 0.0)


def mostrar_resultado(probabilidad, prediccion, nivel_riesgo, costo_estimado):
    st.divider()
    st.subheader("📊 Resultado de la predicción")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Probabilidad de falla",
            f"{probabilidad * 100:.1f}%"
        )

    with col2:
        st.metric(
            "Nivel de riesgo",
            nivel_riesgo
        )

    with col3:
        st.metric(
            "Costo aproximado",
            f"${costo_estimado:,.2f} USD"
        )

    with col4:
        st.metric(
            "Umbral de decisión",
            f"{UMBRAL * 100:.0f}%"
        )

    st.caption(
        "💰 El costo es una estimación orientativa y condicional a que "
        "ocurra una falla o sea necesaria una intervención. El valor real "
        "puede variar según el diagnóstico técnico, repuestos y trabajo requerido."
    )

    if prediccion == 1:
        st.error(
            "⚠️ El modelo identifica riesgo de falla "
            "en los próximos 30 días."
        )
    else:
        st.success(
            "✅ El modelo no identifica un riesgo elevado "
            "de falla en los próximos 30 días."
        )


def mostrar_factores_shap(aumentan, reducen):
    """
    Muestra los factores SHAP con colores intuitivos:
    - Rojo: aumenta la estimación de riesgo
    - Verde: reduce la estimación de riesgo
    """

    st.divider()
    st.subheader("🧠 ¿Por qué el modelo obtuvo este resultado?")

    st.info(
        "SHAP identifica las variables que más influyeron en esta "
        "predicción. Los factores muestran cómo aumentan o reducen "
        "la estimación de riesgo del modelo y no representan, por sí "
        "solos, causas directas de una falla."
    )

    col_shap_1, col_shap_2 = st.columns(2)

    with col_shap_1:
        st.markdown("### ⬆️ Factores que aumentan la estimación")

        if aumentan is None or aumentan.empty:
            st.write("No se identificaron factores positivos relevantes.")
        else:
            for _, fila in aumentan.iterrows():
                factor = fila["factor"]
                valor = fila["valor"]
                shap_val = float(fila["shap"])

                st.markdown(
                    f"""
                    <div style="margin-bottom:18px;">
                        <b>{factor}:</b> {valor}
                        — impacto SHAP:
                        <span style="
                            background-color: rgba(255, 75, 75, 0.15);
                            color: #ff4b4b;
                            padding: 3px 7px;
                            border-radius: 5px;
                            font-family: monospace;
                            font-weight: bold;
                        ">
                            +{abs(shap_val):.4f}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with col_shap_2:
        st.markdown("### ⬇️ Factores que reducen la estimación")

        if reducen is None or reducen.empty:
            st.write("No se identificaron factores negativos relevantes.")
        else:
            for _, fila in reducen.iterrows():
                factor = fila["factor"]
                valor = fila["valor"]
                shap_val = float(fila["shap"])

                st.markdown(
                    f"""
                    <div style="margin-bottom:18px;">
                        <b>{factor}:</b> {valor}
                        — impacto SHAP:
                        <span style="
                            background-color: rgba(0, 200, 120, 0.15);
                            color: #32d583;
                            padding: 3px 7px;
                            border-radius: 5px;
                            font-family: monospace;
                            font-weight: bold;
                        ">
                            -{abs(shap_val):.4f}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def calcular_shap_seguro(datos_vehiculo, top_n=3):
    try:
        return explicar_prediccion(
            datos_vehiculo,
            top_n=top_n
        )
    except Exception as error:
        st.warning(
            "No fue posible calcular SHAP para esta predicción. "
            f"Detalle técnico: {error}"
        )
        return pd.DataFrame(), pd.DataFrame()


def mostrar_recomendacion_ia(texto):
    st.divider()
    st.subheader("🤖 Análisis y recomendación generada por IA")

    st.caption(
        "La IA Generativa utiliza la predicción del modelo y los factores "
        "SHAP como contexto. No realiza un diagnóstico mecánico autónomo."
    )

    # Mejorar la presentación de las secciones generadas.
    texto_formateado = texto
    reemplazos = {
        "RESUMEN:": "### Resumen",
        "INTERPRETACIÓN:": "### Interpretación",
        "RECOMENDACIÓN PREVENTIVA:": "### Recomendación preventiva",
        "ADVERTENCIA:": "### Advertencia",
    }

    for origen, destino in reemplazos.items():
        texto_formateado = texto_formateado.replace(origen, destino)

    st.markdown(texto_formateado)


# ============================================================
# ESTADO DE SESIÓN
# ============================================================

# Vehículo existente
if "existente_id_analizado" not in st.session_state:
    st.session_state.existente_id_analizado = None

if "existente_analizado" not in st.session_state:
    st.session_state.existente_analizado = None

if "resultado_existente" not in st.session_state:
    st.session_state.resultado_existente = None

if "shap_aumentan_existente" not in st.session_state:
    st.session_state.shap_aumentan_existente = None

if "shap_reducen_existente" not in st.session_state:
    st.session_state.shap_reducen_existente = None

if "ia_existente" not in st.session_state:
    st.session_state.ia_existente = None


# Vehículo nuevo
if "nuevo_analizado" not in st.session_state:
    st.session_state.nuevo_analizado = None

if "resultado_nuevo" not in st.session_state:
    st.session_state.resultado_nuevo = None

if "shap_aumentan_nuevo" not in st.session_state:
    st.session_state.shap_aumentan_nuevo = None

if "shap_reducen_nuevo" not in st.session_state:
    st.session_state.shap_reducen_nuevo = None

if "ia_nuevo" not in st.session_state:
    st.session_state.ia_nuevo = None


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

st.title("🚗 Sistema Predictivo de Mantenimiento Vehicular")

st.subheader(
    "Predicción de riesgo de falla vehicular en los próximos 30 días"
)

st.caption(
    "Modelo de aprendizaje automático aplicado a vehículos "
    "registrados en un taller mecánico de Quito."
)


# ============================================================
# MENÚ
# ============================================================

st.sidebar.title("Menú")

opcion = st.sidebar.radio(
    "Seleccione una opción:",
    [
        "🔍 Consultar vehículo",
        "➕ Añadir nuevo vehículo"
    ]
)

st.sidebar.divider()
st.sidebar.success("Modelo predictivo cargado")
st.sidebar.write(f"Vehículos disponibles: **{len(df)}**")


# ============================================================
# CONSULTAR VEHÍCULO EXISTENTE
# ============================================================

if opcion == "🔍 Consultar vehículo":

    st.header("🔍 Consultar vehículo existente")

    vehiculos = sorted(
        df["vehiculo_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    vehiculo_id = st.selectbox(
        "Seleccione el ID del vehículo:",
        vehiculos
    )

    registro = df[
        df["vehiculo_id"].astype(str) == str(vehiculo_id)
    ].iloc[0]

    st.write(
        f"**Vehículo seleccionado:** "
        f"{registro['marca']} {registro['modelo']} "
        f"{registro['anio']}"
    )

    if st.button(
        "🔍 Analizar riesgo",
        type="primary",
        key="btn_analizar_existente"
    ):

        X_existente = pd.DataFrame(
            [registro[FEATURES].to_dict()]
        )

        probabilidad, prediccion, nivel_riesgo = predecir_riesgo(
            X_existente
        )

        costo_estimado = estimar_costo(
            X_existente
        )

        aumentan, reducen = calcular_shap_seguro(
            X_existente,
            top_n=3
        )

        st.session_state.existente_id_analizado = vehiculo_id
        st.session_state.existente_analizado = X_existente
        st.session_state.resultado_existente = {
            "probabilidad": probabilidad,
            "prediccion": prediccion,
            "nivel_riesgo": nivel_riesgo,
            "costo_estimado": costo_estimado
        }
        st.session_state.shap_aumentan_existente = aumentan
        st.session_state.shap_reducen_existente = reducen

        # Una nueva predicción invalida la recomendación anterior.
        st.session_state.ia_existente = None

    # --------------------------------------------------------
    # MOSTRAR ANÁLISIS DEL VEHÍCULO EXISTENTE
    # --------------------------------------------------------

    if (
        st.session_state.existente_id_analizado == vehiculo_id
        and st.session_state.existente_analizado is not None
        and st.session_state.resultado_existente is not None
    ):

        resultado = st.session_state.resultado_existente

        mostrar_resultado(
            resultado["probabilidad"],
            resultado["prediccion"],
            resultado["nivel_riesgo"],
            resultado["costo_estimado"]
        )

        mostrar_factores_shap(
            st.session_state.shap_aumentan_existente,
            st.session_state.shap_reducen_existente
        )

        st.divider()
        st.subheader("✨ Inteligencia Artificial Generativa")

        st.write(
            "Genera una explicación en lenguaje natural a partir de la "
            "predicción y de los factores SHAP mostrados anteriormente."
        )

        if st.button(
            "🤖 Generar análisis y recomendación",
            type="primary",
            key=f"btn_ia_existente_{vehiculo_id}"
        ):
            try:
                with st.spinner(
                    "Generando explicación y recomendación con IA..."
                ):
                    st.session_state.ia_existente = generar_recomendacion_ia(
                        datos_vehiculo=st.session_state.existente_analizado,
                        probabilidad=resultado["probabilidad"],
                        nivel_riesgo=resultado["nivel_riesgo"],
                        costo_estimado=resultado["costo_estimado"],
                        factores_aumentan=st.session_state.shap_aumentan_existente,
                        factores_reducen=st.session_state.shap_reducen_existente
                    )

            except Exception as error:
                st.error(
                    "No fue posible generar la recomendación con IA en este "
                    f"momento. Detalle técnico: {error}"
                )

        if st.session_state.ia_existente:
            mostrar_recomendacion_ia(
                st.session_state.ia_existente
            )


# ============================================================
# AÑADIR NUEVO VEHÍCULO
# ============================================================

elif opcion == "➕ Añadir nuevo vehículo":

    st.header("➕ Registrar y analizar nuevo vehículo")

    st.info(
        "Ingrese los datos disponibles del vehículo. "
        "Si la marca o el modelo no aparecen en la lista, "
        "seleccione la opción para ingresarlos manualmente."
    )

    # --------------------------------------------------------
    # DATOS DEL VEHÍCULO
    # --------------------------------------------------------

    st.subheader("🚗 Datos del vehículo")

    col1, col2, col3 = st.columns(3)

    with col1:

        marcas_conocidas = sorted(
            df["marca"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        marca_opcion = st.selectbox(
            "Marca",
            marcas_conocidas + ["➕ Ingresar otra marca"],
            key="marca_opcion_nuevo"
        )

        if marca_opcion == "➕ Ingresar otra marca":
            marca = st.text_input(
                "Escriba la marca del vehículo",
                placeholder="Ej. Volkswagen",
                key="marca_texto_nuevo"
            ).strip()
        else:
            marca = marca_opcion

        modelos_conocidos = sorted(
            df["modelo"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        modelo_opcion = st.selectbox(
            "Modelo",
            modelos_conocidos + ["➕ Ingresar otro modelo"],
            key="modelo_opcion_nuevo"
        )

        if modelo_opcion == "➕ Ingresar otro modelo":
            modelo_nuevo = st.text_input(
                "Escriba el modelo del vehículo",
                placeholder="Ej. Tiguan",
                key="modelo_texto_nuevo"
            ).strip()
        else:
            modelo_nuevo = modelo_opcion

        anio = st.number_input(
            "Año",
            min_value=1990,
            max_value=2035,
            value=2020,
            step=1,
            key="anio_nuevo"
        )

    with col2:

        tipo_vehiculo = st.selectbox(
            "Tipo de vehículo",
            sorted(
                df["tipo_vehiculo"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
            key="tipo_vehiculo_nuevo"
        )

        combustible = st.selectbox(
            "Combustible",
            sorted(
                df["combustible"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
            key="combustible_nuevo"
        )

        cilindraje_l = st.number_input(
            "Cilindraje (L)",
            min_value=0.5,
            max_value=8.0,
            value=2.0,
            step=0.1,
            key="cilindraje_nuevo"
        )

    with col3:

        transmision = st.selectbox(
            "Transmisión",
            sorted(
                df["transmision"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
            key="transmision_nuevo"
        )

        zona_operacion_quito = st.selectbox(
            "Zona de operación",
            sorted(
                df["zona_operacion_quito"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
            key="zona_nuevo"
        )

        altitud_operacion_m = st.number_input(
            "Altitud de operación (m)",
            min_value=1500,
            max_value=3500,
            value=2800,
            step=10,
            key="altitud_nuevo"
        )

    st.divider()

    # --------------------------------------------------------
    # ESTADO Y MANTENIMIENTO
    # --------------------------------------------------------

    st.subheader("🔧 Estado y mantenimiento")

    col4, col5, col6 = st.columns(3)

    with col4:

        kilometraje_acumulado_km = st.number_input(
            "Kilometraje acumulado (km)",
            min_value=0,
            value=50000,
            step=1000,
            key="kilometraje_nuevo"
        )

        km_desde_ultimo_mantenimiento = st.number_input(
            "Km desde último mantenimiento",
            min_value=0,
            value=5000,
            step=500,
            key="km_ultimo_nuevo"
        )

        dias_desde_ultimo_mantenimiento = st.number_input(
            "Días desde último mantenimiento",
            min_value=0,
            value=120,
            step=1,
            key="dias_ultimo_nuevo"
        )

    with col5:

        temperatura_promedio_motor_c = st.number_input(
            "Temperatura promedio del motor (°C)",
            min_value=50.0,
            max_value=130.0,
            value=90.0,
            step=1.0,
            key="temperatura_nuevo"
        )

        presion_aceite_kpa = st.number_input(
            "Presión de aceite (kPa)",
            min_value=0.0,
            max_value=1000.0,
            value=300.0,
            step=10.0,
            key="presion_nuevo"
        )

        frenadas_bruscas_por_100km = st.number_input(
            "Frenadas bruscas por 100 km",
            min_value=0.0,
            value=3.0,
            step=0.5,
            key="frenadas_nuevo"
        )

    with col6:

        tiempo_ralenti_min_dia = st.number_input(
            "Tiempo en ralentí (min/día)",
            min_value=0.0,
            value=20.0,
            step=1.0,
            key="ralenti_nuevo"
        )

        tiene_codigo_obd_activo = st.selectbox(
            "¿Tiene código OBD activo?",
            ["No", "Sí"],
            key="obd_nuevo"
        )

        tipo_mantenimiento_actual = st.selectbox(
            "Tipo de mantenimiento actual",
            sorted(
                df["tipo_mantenimiento_actual"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
            key="mantenimiento_nuevo"
        )

    st.divider()

    # --------------------------------------------------------
    # INTERVENCIÓN ACTUAL
    # --------------------------------------------------------

    st.subheader("🛠️ Intervención actual")

    col7, col8, col9 = st.columns(3)

    with col7:

        sistema_intervenido_actual = st.selectbox(
            "Sistema intervenido",
            sorted(
                df["sistema_intervenido_actual"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
            key="sistema_nuevo"
        )

    with col8:

        repuesto_cambiado_actual = st.selectbox(
            "Repuesto cambiado",
            sorted(
                df["repuesto_cambiado_actual"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
            key="repuesto_nuevo"
        )

    with col9:

        costo_mantenimiento_actual_usd = st.number_input(
            "Costo mantenimiento actual (USD)",
            min_value=0.0,
            value=150.0,
            step=10.0,
            key="costo_nuevo"
        )

    st.divider()

    # --------------------------------------------------------
    # ANALIZAR NUEVO VEHÍCULO
    # --------------------------------------------------------

    enviar = st.button(
        "🤖 Analizar nuevo vehículo",
        type="primary",
        key="btn_analizar_nuevo"
    )

    if enviar:

        if not marca:
            st.error(
                "Debe ingresar una marca válida."
            )
            st.stop()

        if not modelo_nuevo:
            st.error(
                "Debe ingresar un modelo válido."
            )
            st.stop()

        nuevo = pd.DataFrame([{
            "marca": marca,
            "modelo": modelo_nuevo,
            "anio": anio,
            "tipo_vehiculo": tipo_vehiculo,
            "combustible": combustible,
            "cilindraje_l": cilindraje_l,
            "transmision": transmision,
            "kilometraje_acumulado_km": kilometraje_acumulado_km,
            "km_desde_ultimo_mantenimiento": km_desde_ultimo_mantenimiento,
            "dias_desde_ultimo_mantenimiento": dias_desde_ultimo_mantenimiento,
            "temperatura_promedio_motor_c": temperatura_promedio_motor_c,
            "presion_aceite_kpa": presion_aceite_kpa,
            "frenadas_bruscas_por_100km": frenadas_bruscas_por_100km,
            "tiempo_ralenti_min_dia": tiempo_ralenti_min_dia,
            "altitud_operacion_m": altitud_operacion_m,
            "zona_operacion_quito": zona_operacion_quito,
            "tiene_codigo_obd_activo": (
                1 if tiene_codigo_obd_activo == "Sí" else 0
            ),
            "tipo_mantenimiento_actual": tipo_mantenimiento_actual,
            "sistema_intervenido_actual": sistema_intervenido_actual,
            "repuesto_cambiado_actual": repuesto_cambiado_actual,
            "costo_mantenimiento_actual_usd": costo_mantenimiento_actual_usd
        }])

        probabilidad, prediccion, nivel_riesgo = predecir_riesgo(
            nuevo
        )

        costo_estimado = estimar_costo(
            nuevo
        )

        aumentan, reducen = calcular_shap_seguro(
            nuevo,
            top_n=3
        )

        st.session_state.nuevo_analizado = nuevo

        st.session_state.resultado_nuevo = {
            "probabilidad": probabilidad,
            "prediccion": prediccion,
            "nivel_riesgo": nivel_riesgo,
            "costo_estimado": costo_estimado
        }

        st.session_state.shap_aumentan_nuevo = aumentan
        st.session_state.shap_reducen_nuevo = reducen

        # Una nueva predicción invalida la recomendación anterior.
        st.session_state.ia_nuevo = None

    # --------------------------------------------------------
    # MOSTRAR RESULTADO GUARDADO EN SESIÓN
    # --------------------------------------------------------

    if (
        st.session_state.nuevo_analizado is not None
        and st.session_state.resultado_nuevo is not None
    ):

        resultado = st.session_state.resultado_nuevo

        mostrar_resultado(
            resultado["probabilidad"],
            resultado["prediccion"],
            resultado["nivel_riesgo"],
            resultado["costo_estimado"]
        )

        mostrar_factores_shap(
            st.session_state.shap_aumentan_nuevo,
            st.session_state.shap_reducen_nuevo
        )

        # ----------------------------------------------------
        # IA GENERATIVA
        # ----------------------------------------------------

        st.divider()
        st.subheader("✨ Inteligencia Artificial Generativa")

        st.write(
            "Genera una explicación y recomendación preventiva en lenguaje "
            "natural a partir de la predicción y de los factores SHAP."
        )

        if st.button(
            "🤖 Generar análisis y recomendación",
            type="primary",
            key="btn_ia_nuevo"
        ):
            try:
                with st.spinner(
                    "Generando explicación y recomendación con IA..."
                ):
                    st.session_state.ia_nuevo = generar_recomendacion_ia(
                        datos_vehiculo=st.session_state.nuevo_analizado,
                        probabilidad=resultado["probabilidad"],
                        nivel_riesgo=resultado["nivel_riesgo"],
                        costo_estimado=resultado["costo_estimado"],
                        factores_aumentan=st.session_state.shap_aumentan_nuevo,
                        factores_reducen=st.session_state.shap_reducen_nuevo
                    )

            except Exception as error:
                st.error(
                    "No fue posible generar la recomendación con IA en este "
                    f"momento. Detalle técnico: {error}"
                )

        if st.session_state.ia_nuevo:
            mostrar_recomendacion_ia(
                st.session_state.ia_nuevo
            )

        st.divider()

        # ----------------------------------------------------
        # GUARDAR VEHÍCULO
        # ----------------------------------------------------

        st.subheader("💾 Guardar vehículo")

        nuevo_id = generar_siguiente_id(df)

        st.write(
            f"El vehículo será registrado con el ID: "
            f"**{nuevo_id}**"
        )

        guardar = st.button(
            "💾 Guardar vehículo",
            type="secondary",
            key="btn_guardar_nuevo"
        )

        if guardar:

            nuevo_registro = (
                st.session_state
                .nuevo_analizado
                .copy()
            )

            nuevo_registro.insert(
                0,
                "vehiculo_id",
                nuevo_id
            )

            # Las columnas futuras se dejan vacías porque
            # aún no se conoce si ocurrirá realmente una falla.
            for columna in df.columns:
                if columna not in nuevo_registro.columns:
                    nuevo_registro[columna] = pd.NA

            nuevo_registro = nuevo_registro[
                df.columns
            ]

            df_actualizado = pd.concat(
                [
                    df,
                    nuevo_registro
                ],
                ignore_index=True
            )

            df_actualizado.to_csv(
                DATA_PATH,
                index=False
            )

            st.cache_data.clear()

            st.session_state.nuevo_analizado = None
            st.session_state.resultado_nuevo = None
            st.session_state.shap_aumentan_nuevo = None
            st.session_state.shap_reducen_nuevo = None
            st.session_state.ia_nuevo = None

            st.success(
                f"✅ Vehículo {nuevo_id} "
                f"guardado correctamente."
            )

            st.info(
                "El vehículo ya está registrado. "
                "Puede consultarlo desde la opción "
                "'Consultar vehículo'."
            )
