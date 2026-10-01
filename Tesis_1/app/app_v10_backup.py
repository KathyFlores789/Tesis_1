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
    page_title="Mantenimiento Inteligente | V10",
    page_icon="🚗",
    layout="wide"
)

# ============================================================
# ESTILO VISUAL V3
# ============================================================

st.markdown("""
<style>
.block-container {max-width:1320px; padding-top:2rem; padding-bottom:3rem;}
.hero {padding:1.55rem 1.7rem; border:1px solid rgba(120,140,170,.22);
border-radius:18px; background:linear-gradient(135deg,rgba(30,58,95,.32),rgba(18,24,38,.20));
margin-bottom:1.35rem;}
.hero h1 {margin:0; font-size:2.15rem; line-height:1.15;}
.hero p {margin:.55rem 0 0 0; opacity:.78; font-size:1rem;}
.hero-badge {display:inline-block; margin-top:.9rem; padding:.28rem .7rem; border-radius:999px;
border:1px solid rgba(91,156,255,.38); background:rgba(91,156,255,.10); font-size:.82rem; font-weight:600;}
.vehicle-card {border:1px solid rgba(120,140,170,.22); border-radius:16px; padding:1rem 1.15rem;
background:rgba(255,255,255,.025); margin:.45rem 0 1rem 0;}
.vehicle-card .eyebrow {opacity:.62; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.3rem;}
.vehicle-card .vehicle-title {font-size:1.18rem; font-weight:700; margin-bottom:.2rem;}
.vehicle-card .vehicle-meta {opacity:.76; font-size:.92rem;}
[data-testid="stMetric"] {border:1px solid rgba(120,140,170,.20); border-radius:14px;
padding:.85rem 1rem; background:rgba(255,255,255,.025); min-height:112px;}
[data-testid="stMetricLabel"] {opacity:.72;}
[data-testid="stMetricValue"] {font-weight:750;}
.stButton > button {border-radius:10px; min-height:2.65rem; font-weight:650;}
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {border-radius:9px !important;}
[data-testid="stSidebar"] {border-right:1px solid rgba(120,140,170,.16);}
.sidebar-brand {padding:.5rem .1rem .9rem .1rem;}
.sidebar-brand .name {font-weight:750; font-size:1.08rem;}
.sidebar-brand .version {opacity:.62; font-size:.8rem; margin-top:.2rem;}
h2,h3 {letter-spacing:-.015em;}
.app-footer {margin-top:2.5rem; padding-top:1rem; border-top:1px solid rgba(120,140,170,.16);
text-align:center; opacity:.58; font-size:.82rem;}

/* ===== V3.1 ===== */
.risk-banner-high,.risk-banner-low{border-radius:14px;padding:1rem 1.1rem;margin:.9rem 0 1rem;font-weight:650;border:1px solid;}
.risk-banner-high{background:rgba(255,75,75,.12);border-color:rgba(255,75,75,.30);color:#ff6b6b;}
.risk-banner-low{background:rgba(50,213,131,.10);border-color:rgba(50,213,131,.28);color:#45d995;}
.prob-wrap{margin:.35rem 0 1rem}.prob-track{height:9px;width:100%;border-radius:999px;background:rgba(130,145,170,.18);overflow:hidden}
.prob-high{height:100%;border-radius:999px;background:#ff4b4b}.prob-low{height:100%;border-radius:999px;background:#32d583}
.shap-panel{border:1px solid rgba(120,140,170,.20);border-radius:16px;padding:1rem 1.15rem .6rem;background:rgba(255,255,255,.025);min-height:250px;margin:.35rem 0 .8rem}
.shap-up{border-top:3px solid rgba(255,75,75,.75)}.shap-down{border-top:3px solid rgba(50,213,131,.75)}
.ai-card{border:1px solid rgba(120,140,170,.20);border-radius:16px;padding:1rem 1.2rem;background:rgba(255,255,255,.025);margin:.8rem 0}
.ai-card h3{font-size:1.2rem;margin:0 0 .65rem}.ai-card p{line-height:1.65;margin:0;overflow-wrap:anywhere}
.ai-summary{border-left:4px solid #5b9cff}.ai-interpretation{border-left:4px solid #9b7cff}.ai-recommendation{border-left:4px solid #32d583}.ai-warning{border-left:4px solid #f2b84b}


/* ===== V3.3: SHAP compacto + IA horizontal ===== */
.shap-box{border:1px solid;border-radius:16px;overflow:hidden;background:rgba(255,255,255,.018);margin:.35rem 0 1rem}
.shap-box-up{border-color:rgba(255,75,75,.55)}.shap-box-down{border-color:rgba(50,213,131,.55)}
.shap-title{display:flex;align-items:center;justify-content:space-between;padding:.85rem 1rem;font-weight:750;font-size:1.03rem}
.shap-title-up{background:rgba(255,75,75,.13)}.shap-title-down{background:rgba(50,213,131,.11)}
.shap-badge{display:inline-flex;align-items:center;justify-content:center;min-width:28px;height:28px;padding:0 8px;border-radius:999px;font-size:.78rem;font-weight:800}
.shap-badge-up{background:#ff4b4b;color:#fff}.shap-badge-down{background:#32d583;color:#07130d}
.shap-items{padding:.55rem .7rem .7rem}
.shap-item{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.58rem .72rem;margin:.35rem 0;border-radius:9px;background:rgba(120,140,170,.09);border:1px solid rgba(120,140,170,.13)}
.shap-impact{flex:0 0 auto;padding:.2rem .55rem;border-radius:7px;font-family:monospace;font-weight:800}
.shap-impact-up{background:rgba(255,75,75,.16);color:#ff6565}.shap-impact-down{background:rgba(50,213,131,.15);color:#39dc8b}
.ai-grid-v33{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1rem;margin:.8rem 0 1rem}
.ai-grid-v33 .ai-card{height:auto;margin:0}
.ai-grid-v33 .ai-card h3{font-size:1.03rem;margin:0 0 .65rem}
.ai-grid-v33 .ai-card p{font-size:.94rem;line-height:1.55;margin:0}
@media(max-width:1150px){.ai-grid-v33{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:720px){.ai-grid-v33{grid-template-columns:1fr}.shap-item{align-items:flex-start;flex-direction:column;gap:.45rem}}


/* ===== V3.4: IA en tarjetas horizontales ===== */
.ai-stack-v34{
    display:flex;
    flex-direction:column;
    gap:1rem;
    margin:.9rem 0 1rem;
    width:100%;
}
.ai-stack-v34 .ai-card{
    width:100%;
    height:auto;
    min-height:0;
    margin:0;
    padding:1.15rem 1.3rem;
    box-sizing:border-box;
}
.ai-stack-v34 .ai-card h3{
    font-size:1.12rem;
    margin:0 0 .7rem 0;
}
.ai-stack-v34 .ai-card p{
    font-size:1rem;
    line-height:1.65;
    margin:0;
    max-width:none;
}


/* ===== V5: aclaración de costos sin intervalos no validados ===== */
.cost-note{border:1px solid rgba(242,184,75,.32);border-left:4px solid #f2b84b;border-radius:12px;background:rgba(242,184,75,.07);padding:.85rem 1rem;margin:.8rem 0 1rem;line-height:1.6;font-size:.94rem}

/* ===== V3.5: fondos sutiles para tarjetas de IA ===== */
.ai-stack-v34 .ai-summary{
    background:linear-gradient(90deg, rgba(91,156,255,.10), rgba(91,156,255,.035));
    border-color:rgba(91,156,255,.28);
    border-left:4px solid #5b9cff;
}
.ai-stack-v34 .ai-interpretation{
    background:linear-gradient(90deg, rgba(155,124,255,.10), rgba(155,124,255,.035));
    border-color:rgba(155,124,255,.28);
    border-left:4px solid #9b7cff;
}
.ai-stack-v34 .ai-recommendation{
    background:linear-gradient(90deg, rgba(50,213,131,.09), rgba(50,213,131,.03));
    border-color:rgba(50,213,131,.26);
    border-left:4px solid #32d583;
}
.ai-stack-v34 .ai-warning{
    background:linear-gradient(90deg, rgba(242,184,75,.09), rgba(242,184,75,.03));
    border-color:rgba(242,184,75,.26);
    border-left:4px solid #f2b84b;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.explicabilidad_shap_v2 import explicar_prediccion
from src.generador_ia_v3 import generar_recomendacion_ia


DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "datos_mantenimiento_quito.csv"
)

SEGUIMIENTO_PATH = (
    ROOT_DIR
    / "data"
    / "seguimiento"
    / "vehiculos_seguimiento.csv"
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
def cargar_datos_historicos():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def cargar_datos_seguimiento():
    if SEGUIMIENTO_PATH.is_file():
        return pd.read_csv(SEGUIMIENTO_PATH)

    # Si todavía no existe el archivo de seguimiento, se crea una
    # estructura vacía compatible con el dataset histórico.
    columnas = pd.read_csv(DATA_PATH, nrows=0).columns
    return pd.DataFrame(columns=columnas)


@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def cargar_regresor():
    return joblib.load(REGRESSOR_PATH)


# V8: el clasificador de sistemas es opcional y EXCLUSIVAMENTE experimental.
SYSTEM_MODEL_PATH = ROOT_DIR / "models" / "system_classifier_v1.joblib"
SYSTEM_FEATURES = [
    "marca", "modelo", "tipo_vehiculo", "combustible", "transmision",
    "zona_operacion_quito", "tiene_codigo_obd_activo",
    "tipo_mantenimiento_actual", "sistema_intervenido_actual",
    "repuesto_cambiado_actual", "anio", "cilindraje_l",
    "kilometraje_acumulado_km", "km_desde_ultimo_mantenimiento",
    "dias_desde_ultimo_mantenimiento", "temperatura_promedio_motor_c",
    "presion_aceite_kpa", "frenadas_bruscas_por_100km",
    "tiempo_ralenti_min_dia", "altitud_operacion_m",
    "costo_mantenimiento_actual_usd"
]

@st.cache_resource
def cargar_clasificador_sistemas():
    if not SYSTEM_MODEL_PATH.is_file():
        return None
    return joblib.load(SYSTEM_MODEL_PATH)


def mostrar_sistemas_experimentales(datos_entrada):
    """Visualización condicional, no calibrada y no diagnóstica."""
    st.divider()
    st.subheader("🔬 Exploración de posibles sistemas de falla")
    with st.expander("Ver distribución experimental entre los cinco sistemas", expanded=False):
        st.caption(
            "Distribución estimada de los sistemas que podrían verse afectados "
            "si el vehículo presenta una falla en los próximos 30 días. "
            "Los resultados son orientativos y deben complementarse con "
            "una revisión técnica del vehículo."
        )
        try:
            clasificador = cargar_clasificador_sistemas()
            if clasificador is None:
                st.info(
                    "No se encontró models/system_classifier_v1.joblib. "
                    "Copia el modelo V1 generado en tu computadora a la carpeta models. "
                    "El resto de la aplicación continúa funcionando."
                )
                return
            faltantes = [c for c in SYSTEM_FEATURES if c not in datos_entrada.columns]
            if faltantes:
                st.error("Faltan variables para el módulo experimental: " + ", ".join(faltantes))
                return
            entrada = datos_entrada.loc[:, SYSTEM_FEATURES].copy()
            salida = clasificador.predict_proba(entrada)[0]
            nombres = clasificador.classes_ if hasattr(clasificador, "classes_") else clasificador.named_steps["classifier"].classes_
            distribucion = sorted(zip(nombres, salida), key=lambda par: par[1], reverse=True)
            for nombre, valor in distribucion:
                st.write(f"**{nombre}** — {valor * 100:.1f} %")
                st.progress(float(valor))
            st.caption(
                "La distribución es una salida matemática de un modelo de bajo "
                "rendimiento; no indica una avería confirmada ni reemplaza la "
                "inspección física."
            )
        except Exception as error:
            st.error("No fue posible calcular el módulo experimental: " + str(error))


# Los históricos y los vehículos en seguimiento se mantienen separados
# físicamente, pero se combinan en memoria para consulta y predicción.
df_historico = cargar_datos_historicos()
df_seguimiento = cargar_datos_seguimiento()
df = pd.concat([df_historico, df_seguimiento], ignore_index=True)

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
    st.subheader("📊 Resultado del análisis predictivo")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probabilidad de falla", f"{probabilidad * 100:.1f}%")
    with col2:
        st.metric("Nivel de riesgo", nivel_riesgo)
    with col3:
        st.metric("Costo estimado si ocurre una falla", f"${costo_estimado:,.2f} USD")
    porcentaje = max(0.0, min(probabilidad * 100, 100.0))
    clase = "prob-high" if prediccion == 1 else "prob-low"
    st.markdown(f"""<div class="prob-wrap"><div class="prob-track"><div class="{clase}" style="width:{porcentaje:.1f}%"></div></div></div>""", unsafe_allow_html=True)
    # Aviso solo para resultados a cinco puntos porcentuales o menos del umbral.
    if abs(probabilidad - UMBRAL) <= 0.05:
        st.warning(
            "⚠️ Resultado cercano al límite de clasificación. "
            "Aunque el nivel mostrado sea " + nivel_riesgo.lower() + ", "
            "la probabilidad estimada está próxima al umbral; "
            "interprete el resultado con precaución."
        )
    with st.expander("ℹ️ Información técnica de la clasificación"):
        st.write(
            f"El umbral de decisión es del {UMBRAL * 100:.0f} %. "
            "Las probabilidades iguales o superiores se clasifican como riesgo ALTO; "
            "las inferiores, como riesgo BAJO. "
            "Este criterio de clasificación no garantiza que las probabilidades "
            "estimadas estén calibradas."
        )
    st.markdown("""<div class="cost-note"><strong>💰 ¿Qué significa este costo?</strong><br>Es una estimación orientativa del costo de una <strong>posible falla futura</strong>, condicionada a que esta ocurra. <strong>No es el costo del mantenimiento actual ni una cotización del taller.</strong> El importe real dependerá de la avería, los repuestos y la mano de obra. El modelo tiene mayor incertidumbre en reparaciones costosas y puede subestimar su importe. No se muestra un intervalo numérico porque todavía no está calibrado para el modelo desplegado.</div>""", unsafe_allow_html=True)
    if prediccion == 1:
        st.markdown('<div class="risk-banner-high">⚠️ <b>Riesgo elevado:</b> el modelo identifica riesgo de falla en los próximos 30 días.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-banner-low">✓ <b>Riesgo bajo:</b> el modelo no identifica un riesgo elevado de falla en los próximos 30 días.</div>', unsafe_allow_html=True)


def mostrar_factores_shap(aumentan, reducen):
    st.divider()
    st.subheader("🧠 Explicación de la predicción")
    st.info(
        "SHAP identifica las variables que más influyeron en esta predicción. "
        "Los valores positivos aumentan la estimación de riesgo y los negativos la reducen. "
        "Estos impactos no representan, por sí solos, causas directas de una falla."
    )

    def grupo_shap(factores, aumenta=True):
        if aumenta:
            titulo, box, head, badge, impact, signo = (
                "⬆ Factores que aumentan la estimación",
                "shap-box-up", "shap-title-up", "shap-badge-up", "shap-impact-up", "+"
            )
        else:
            titulo, box, head, badge, impact, signo = (
                "⬇ Factores que reducen la estimación",
                "shap-box-down", "shap-title-down", "shap-badge-down", "shap-impact-down", "-"
            )

        cantidad = 0 if factores is None else len(factores)
        items = []

        if factores is None or factores.empty:
            items.append('<div class="shap-item"><span>No se identificaron factores relevantes.</span></div>')
        else:
            for _, fila in factores.iterrows():
                factor = str(fila["factor"])
                valor = str(fila["valor"])
                shap_val = abs(float(fila["shap"]))
                items.append(
                    f'<div class="shap-item"><span><b>{factor}:</b> {valor}</span>'
                    f'<span class="shap-impact {impact}">{signo}{shap_val:.4f}</span></div>'
                )

        return (
            f'<div class="shap-box {box}">'
            f'<div class="shap-title {head}"><span>{titulo}</span>'
            f'<span class="shap-badge {badge}">{cantidad}</span></div>'
            f'<div class="shap-items">{"".join(items)}</div></div>'
        )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(grupo_shap(aumentan, True), unsafe_allow_html=True)
    with col2:
        st.markdown(grupo_shap(reducen, False), unsafe_allow_html=True)

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
    st.subheader("🤖 Análisis preventivo con IA Generativa")
    st.caption(
        "La IA Generativa utiliza la predicción del modelo y los factores SHAP como contexto. "
        "No realiza un diagnóstico mecánico autónomo."
    )

    secciones = [
        ("RESUMEN:", "📋 Resumen", "ai-summary"),
        ("INTERPRETACIÓN:", "🧠 Interpretación", "ai-interpretation"),
        ("RECOMENDACIÓN PREVENTIVA:", "💡 Recomendación preventiva", "ai-recommendation"),
        ("ADVERTENCIA:", "⚠️ Advertencia", "ai-warning"),
    ]

    texto_upper = texto.upper()
    encontrados = []

    for marcador, titulo, clase in secciones:
        pos = texto_upper.find(marcador)
        if pos >= 0:
            encontrados.append((pos, marcador, titulo, clase))

    encontrados.sort(key=lambda x: x[0])

    if not encontrados:
        st.markdown(texto)
        return

    tarjetas = []

    for i, (pos, marcador, titulo, clase) in enumerate(encontrados):
        inicio = pos + len(marcador)
        fin = encontrados[i + 1][0] if i + 1 < len(encontrados) else len(texto)
        contenido = texto[inicio:fin].strip().replace("\n", "<br>")

        tarjetas.append(
            f'<div class="ai-card {clase}">'
            f'<h3>{titulo}</h3>'
            f'<p>{contenido}</p>'
            f'</div>'
        )

    st.markdown(
        f'<div class="ai-stack-v34">{"".join(tarjetas)}</div>',
        unsafe_allow_html=True
    )

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

st.markdown("""
<div class="hero">
<h1>🚗 Sistema Inteligente de Mantenimiento Vehicular</h1>
<p>Predicción de mantenimiento vehicular mediante Machine Learning,<br>explicabilidad SHAP e Inteligencia Artificial Generativa.</p>
<span class="hero-badge">ML · SHAP · IA Generativa</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MENÚ
# ============================================================

st.sidebar.markdown("""
<div class="sidebar-brand">
<div class="name">🚘 Mantenimiento Inteligente</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Navegación")

opcion = st.sidebar.radio(
    "Seleccione una opción:",
    [
        "🔍 Consultar vehículo",
        "➕ Añadir nuevo vehículo"
    ]
)

st.sidebar.divider()
st.sidebar.success("● Modelos cargados correctamente")
st.sidebar.metric("Vehículos registrados", f"{len(df):,}")
st.sidebar.caption("Sistema de apoyo a la decisión. Las predicciones no sustituyen el diagnóstico técnico.")


# ============================================================
# CONSULTAR VEHÍCULO EXISTENTE
# ============================================================

if opcion == "🔍 Consultar vehículo":

    st.header("🔍 Consulta y análisis de vehículo")

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

    st.markdown(
        f"""
        <div class="vehicle-card">
            <div class="eyebrow">Vehículo seleccionado · {vehiculo_id}</div>
            <div class="vehicle-title">{registro['marca']} {registro['modelo']} · {registro['anio']}</div>
            <div class="vehicle-meta">
                {registro['tipo_vehiculo']} · {registro['combustible']} ·
                {float(registro['kilometraje_acumulado_km']):,.0f} km acumulados
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Ficha completa del registro seleccionado: solo datos conocidos al analizar.
    # Las columnas de fallas futuras son resultados históricos, no estado actual.
    with st.expander("📋 Ver ficha completa del vehículo", expanded=False):
        st.caption("Datos registrados del vehículo y de su mantenimiento actual.")

        def mostrar_dato(etiqueta, columna, unidad="", decimales=None):
            if columna not in registro.index or pd.isna(registro[columna]):
                valor = "No registrado"
            else:
                dato = registro[columna]
                if isinstance(dato, str) and not dato.strip():
                    valor = "No registrado"
                elif decimales is not None:
                    try:
                        valor = f"{float(dato):,.{decimales}f}{unidad}"
                    except (TypeError, ValueError):
                        valor = f"{dato}{unidad}"
                else:
                    valor = f"{dato}{unidad}"
            st.markdown(f"**{etiqueta}:** {valor}")

        st.markdown("#### 🚗 Identificación y características")
        c1, c2, c3 = st.columns(3)
        with c1:
            mostrar_dato("ID", "vehiculo_id")
            mostrar_dato("Marca", "marca")
            mostrar_dato("Modelo", "modelo")
        with c2:
            mostrar_dato("Año", "anio", decimales=0)
            mostrar_dato("Tipo de vehículo", "tipo_vehiculo")
            mostrar_dato("Combustible", "combustible")
        with c3:
            mostrar_dato("Cilindraje", "cilindraje_l", " L", 1)
            mostrar_dato("Transmisión", "transmision")
            mostrar_dato("Zona de operación", "zona_operacion_quito")
            mostrar_dato("Altitud de operación", "altitud_operacion_m", " m", 0)

        st.markdown("#### 🛣️ Uso y condiciones de funcionamiento")
        c1, c2, c3 = st.columns(3)
        with c1:
            mostrar_dato("Kilometraje acumulado", "kilometraje_acumulado_km", " km", 0)
            mostrar_dato("Km desde el último mantenimiento", "km_desde_ultimo_mantenimiento", " km", 0)
        with c2:
            mostrar_dato("Temperatura promedio del motor", "temperatura_promedio_motor_c", " °C", 1)
            mostrar_dato("Presión de aceite", "presion_aceite_kpa", " kPa", 1)
        with c3:
            mostrar_dato("Frenadas bruscas por 100 km", "frenadas_bruscas_por_100km", decimales=1)
            mostrar_dato("Tiempo en ralentí", "tiempo_ralenti_min_dia", " min/día", 1)
            mostrar_dato("Código OBD activo", "tiene_codigo_obd_activo")

        st.markdown("#### 🔧 Mantenimiento registrado")
        c1, c2, c3 = st.columns(3)
        with c1:
            mostrar_dato("Fecha de mantenimiento", "fecha_mantenimiento")
            mostrar_dato("Días desde el último mantenimiento", "dias_desde_ultimo_mantenimiento", " días", 0)
        with c2:
            mostrar_dato("Tipo de mantenimiento", "tipo_mantenimiento_actual")
            mostrar_dato("Sistema intervenido", "sistema_intervenido_actual")
        with c3:
            mostrar_dato("Repuesto cambiado", "repuesto_cambiado_actual")
            mostrar_dato("Costo del mantenimiento", "costo_mantenimiento_actual_usd", " USD", 2)

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

        mostrar_sistemas_experimentales(st.session_state.existente_analizado)

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

    st.header("➕ Registro y análisis de nuevo vehículo")

    st.info(
        "Complete la información real del vehículo. Los campos obligatorios "
        "comienzan vacíos; los ejemplos no se utilizan para predecir. "
        "Si la marca o el modelo no aparecen, puede ingresarlos manualmente."
    )

    # Los valores vacíos evitan confundir datos de ejemplo con datos reales.
    # index=None y value=None requieren una versión reciente de Streamlit.
    def opciones_columna(columna):
        return sorted(df[columna].dropna().astype(str).str.strip().loc[
            lambda serie: serie.ne("")
        ].unique().tolist(), key=str.casefold)

    def seleccionar(etiqueta, columna, clave, texto=None):
        return st.selectbox(
            etiqueta, opciones_columna(columna), index=None,
            placeholder=texto or f"Seleccione {etiqueta.lower()}", key=clave
        )

    st.subheader("🚗 Datos del vehículo")
    col1, col2, col3 = st.columns(3)
    with col1:
        marca_opcion = st.selectbox(
            "Marca *", opciones_columna("marca") + ["➕ Ingresar otra marca"],
            index=None, placeholder="Seleccione una marca", key="marca_opcion_nuevo"
        )
        if marca_opcion == "➕ Ingresar otra marca":
            marca = st.text_input(
                "Nueva marca *", placeholder="Ej. Volkswagen", key="marca_texto_nuevo"
            ).strip()
        else:
            marca = marca_opcion

        # Solo se muestran modelos asociados a la marca seleccionada.
        if marca and marca_opcion != "➕ Ingresar otra marca":
            modelos_marca = sorted(
                df.loc[
                    df["marca"].astype(str).str.strip().str.casefold() == marca.casefold(),
                    "modelo"
                ].dropna().astype(str).str.strip().loc[lambda serie: serie.ne("")]
                .unique().tolist(), key=str.casefold
            )
        else:
            modelos_marca = []
        modelo_opcion = st.selectbox(
            "Modelo *", modelos_marca + ["➕ Ingresar otro modelo"],
            index=None, placeholder="Primero seleccione la marca" if not marca else "Seleccione un modelo",
            disabled=not bool(marca), key=f"modelo_opcion_{marca or 'sin_marca'}"
        )
        if modelo_opcion == "➕ Ingresar otro modelo":
            modelo_nuevo = st.text_input(
                "Nuevo modelo *", placeholder="Ej. Tiguan", key="modelo_texto_nuevo"
            ).strip()
        else:
            modelo_nuevo = modelo_opcion

        anio = st.number_input(
            "Año *", min_value=1990, max_value=2035, value=None,
            step=1, placeholder="Ej. 2020", key="anio_nuevo"
        )

    with col2:
        tipo_vehiculo = seleccionar("Tipo de vehículo *", "tipo_vehiculo", "tipo_vehiculo_nuevo")
        combustible = seleccionar("Combustible *", "combustible", "combustible_nuevo")
        cilindraje_l = st.number_input(
            "Cilindraje (L) *", min_value=0.5, max_value=8.0, value=None,
            step=0.1, placeholder="Ej. 2.0", key="cilindraje_nuevo"
        )

    with col3:
        transmision = seleccionar("Transmisión *", "transmision", "transmision_nuevo")
        zona_operacion_quito = seleccionar(
            "Zona de operación *", "zona_operacion_quito", "zona_nuevo"
        )
        altitud_operacion_m = st.number_input(
            "Altitud de operación (m) *", min_value=1500, max_value=3500,
            value=None, step=10, placeholder="Ej. 2800", key="altitud_nuevo"
        )

    st.divider()
    st.subheader("🔧 Estado y mantenimiento")
    col4, col5, col6 = st.columns(3)
    with col4:
        kilometraje_acumulado_km = st.number_input(
            "Kilometraje acumulado (km) *", min_value=0, value=None,
            step=1000, placeholder="Ej. 50000", key="kilometraje_nuevo"
        )
        km_desde_ultimo_mantenimiento = st.number_input(
            "Km desde último mantenimiento *", min_value=0, value=None,
            step=500, placeholder="Ej. 5000", key="km_ultimo_nuevo"
        )
        dias_desde_ultimo_mantenimiento = st.number_input(
            "Días desde último mantenimiento *", min_value=0, value=None,
            step=1, placeholder="Ej. 120", key="dias_ultimo_nuevo"
        )
    with col5:
        temperatura_promedio_motor_c = st.number_input(
            "Temperatura promedio del motor (°C) *", min_value=50.0,
            max_value=130.0, value=None, step=1.0,
            placeholder="Ej. 90", key="temperatura_nuevo"
        )
        presion_aceite_kpa = st.number_input(
            "Presión de aceite (kPa) *", min_value=0.0, max_value=1000.0,
            value=None, step=10.0, placeholder="Ej. 300", key="presion_nuevo"
        )
        frenadas_bruscas_por_100km = st.number_input(
            "Frenadas bruscas por 100 km *", min_value=0.0, value=None,
            step=0.5, placeholder="Ej. 3", key="frenadas_nuevo"
        )
    with col6:
        tiempo_ralenti_min_dia = st.number_input(
            "Tiempo en ralentí (min/día) *", min_value=0.0, value=None,
            step=1.0, placeholder="Ej. 20", key="ralenti_nuevo"
        )
        tiene_codigo_obd_activo = st.selectbox(
            "¿Tiene código OBD activo? *", ["No", "Sí"], index=None,
            placeholder="Seleccione una opción", key="obd_nuevo"
        )
        tipo_mantenimiento_actual = seleccionar(
            "Tipo de mantenimiento actual *", "tipo_mantenimiento_actual", "mantenimiento_nuevo"
        )

    st.divider()
    st.subheader("🛠️ Intervención actual")
    col7, col8, col9 = st.columns(3)
    with col7:
        sistema_intervenido_actual = seleccionar(
            "Sistema intervenido *", "sistema_intervenido_actual", "sistema_nuevo"
        )
    with col8:
        repuesto_cambiado_actual = seleccionar(
            "Repuesto cambiado *", "repuesto_cambiado_actual", "repuesto_nuevo"
        )
    with col9:
        costo_mantenimiento_actual_usd = st.number_input(
            "Costo mantenimiento actual (USD) *", min_value=0.0,
            value=None, step=10.0, placeholder="Ej. 150", key="costo_nuevo"
        )
    st.caption("* Campos obligatorios. Un valor de 0 es distinto de un campo sin completar.")
    st.divider()

    # Validación de los campos ANTES de invocar los modelos.
    campos_obligatorios = {
        "Marca": marca, "Modelo": modelo_nuevo, "Año": anio,
        "Tipo de vehículo": tipo_vehiculo, "Combustible": combustible,
        "Cilindraje": cilindraje_l, "Transmisión": transmision,
        "Kilometraje acumulado": kilometraje_acumulado_km,
        "Km desde último mantenimiento": km_desde_ultimo_mantenimiento,
        "Días desde último mantenimiento": dias_desde_ultimo_mantenimiento,
        "Temperatura promedio del motor": temperatura_promedio_motor_c,
        "Presión de aceite": presion_aceite_kpa,
        "Frenadas bruscas": frenadas_bruscas_por_100km,
        "Tiempo en ralentí": tiempo_ralenti_min_dia,
        "Altitud de operación": altitud_operacion_m,
        "Zona de operación": zona_operacion_quito,
        "Código OBD activo": tiene_codigo_obd_activo,
        "Tipo de mantenimiento actual": tipo_mantenimiento_actual,
        "Sistema intervenido": sistema_intervenido_actual,
        "Repuesto cambiado": repuesto_cambiado_actual,
        "Costo mantenimiento actual": costo_mantenimiento_actual_usd,
    }

    enviar = st.button(
        "🤖 Analizar nuevo vehículo",
        type="primary",
        key="btn_analizar_nuevo"
    )

    if enviar:
        faltantes = [
            nombre for nombre, valor in campos_obligatorios.items()
            if valor is None or (isinstance(valor, str) and not valor.strip())
        ]
        if faltantes:
            st.error("Complete los siguientes campos antes de analizar: " + ", ".join(faltantes) + ".")
            st.stop()
        if km_desde_ultimo_mantenimiento > kilometraje_acumulado_km:
            st.error("Los kilómetros desde el último mantenimiento no pueden superar el kilometraje acumulado.")
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

        mostrar_sistemas_experimentales(st.session_state.nuevo_analizado)

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

            # Los vehículos nuevos se guardan únicamente en seguimiento.
            # El dataset histórico de entrenamiento permanece intacto.
            SEGUIMIENTO_PATH.parent.mkdir(parents=True, exist_ok=True)

            seguimiento_actualizado = pd.concat(
                [
                    df_seguimiento,
                    nuevo_registro
                ],
                ignore_index=True
            )

            seguimiento_actualizado.to_csv(
                SEGUIMIENTO_PATH,
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


# ============================================================
# PIE DE PÁGINA V3
# ============================================================
st.markdown("""
<div class="app-footer">
Sistema Inteligente de Mantenimiento Vehicular · Modelo predictivo + SHAP + IA Generativa<br>
Resultados orientativos para apoyo a la toma de decisiones.
</div>
""", unsafe_allow_html=True)
