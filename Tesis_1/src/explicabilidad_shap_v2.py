"""
src/explicabilidad_shap_v2.py
-----------------------------
Explicabilidad local SHAP para classifier_v2.joblib.

Explica exactamente el mismo modelo que utiliza Streamlit:
Pipeline = preprocesamiento + LogisticRegression.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "classifier_v2.joblib"
)

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "datos_mantenimiento_quito.csv"
)


# ============================================================
# VARIABLES UTILIZADAS POR EL MODELO V2
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

# ============================================================
# NOMBRES AMIGABLES PARA MOSTRAR EN STREAMLIT
# ============================================================

NOMBRES_AMIGABLES = {
    "marca": "Marca",
    "modelo": "Modelo",
    "anio": "Año",
    "tipo_vehiculo": "Tipo de vehículo",
    "combustible": "Combustible",
    "cilindraje_l": "Cilindraje",
    "transmision": "Transmisión",
    "kilometraje_acumulado_km": "Kilometraje acumulado",
    "km_desde_ultimo_mantenimiento": "Km desde último mantenimiento",
    "dias_desde_ultimo_mantenimiento": "Días desde último mantenimiento",
    "temperatura_promedio_motor_c": "Temperatura promedio del motor",
    "presion_aceite_kpa": "Presión de aceite",
    "frenadas_bruscas_por_100km": "Frenadas bruscas por 100 km",
    "tiempo_ralenti_min_dia": "Tiempo en ralentí por día",
    "altitud_operacion_m": "Altitud de operación",
    "zona_operacion_quito": "Zona de operación",
    "tiene_codigo_obd_activo": "Código OBD activo",
    "tipo_mantenimiento_actual": "Tipo de mantenimiento actual",
    "sistema_intervenido_actual": "Sistema intervenido",
    "repuesto_cambiado_actual": "Repuesto cambiado",
    "costo_mantenimiento_actual_usd": "Costo del mantenimiento actual"
}


UNIDADES = {
    "cilindraje_l": " L",
    "kilometraje_acumulado_km": " km",
    "km_desde_ultimo_mantenimiento": " km",
    "dias_desde_ultimo_mantenimiento": " días",
    "temperatura_promedio_motor_c": " °C",
    "presion_aceite_kpa": " kPa",
    "frenadas_bruscas_por_100km": "",
    "tiempo_ralenti_min_dia": " min/día",
    "altitud_operacion_m": " m",
    "costo_mantenimiento_actual_usd": " USD"
}


# ============================================================
# CARGA DEL MODELO REAL
# ============================================================

modelo_pipeline = joblib.load(MODEL_PATH)

preprocessor = modelo_pipeline.named_steps["preprocessor"]
clasificador = modelo_pipeline.named_steps["model"]


# ============================================================
# DATASET DE REFERENCIA
# ============================================================

df = pd.read_csv(DATA_PATH)

X_referencia = df[FEATURES].copy()


# ============================================================
# PREPARAR BACKGROUND PARA SHAP
# ============================================================

# Utilizamos solamente registros válidos como referencia.
# No estamos reentrenando el modelo.

X_background_raw = X_referencia.iloc[:100].copy()

X_background = preprocessor.transform(
    X_background_raw
)

# Algunos preprocesadores generan matrices sparse.
if hasattr(X_background, "toarray"):
    X_background = X_background.toarray()


# ============================================================
# NOMBRES DE VARIABLES DESPUÉS DEL PREPROCESAMIENTO
# ============================================================

feature_names = preprocessor.get_feature_names_out()

# Limpiar prefijos "num__" y "cat__" para presentación.
feature_names_limpios = [
    nombre
    .replace("num__", "")
    .replace("cat__", "")
    for nombre in feature_names
]


# ============================================================
# EXPLAINER SHAP
# ============================================================

explainer = shap.LinearExplainer(
    clasificador,
    X_background,
    feature_names=feature_names_limpios
)


# ============================================================
# FUNCIÓN PARA EXPLICAR UNA PREDICCIÓN
# ============================================================

def explicar_prediccion(datos_vehiculo, top_n=5):
    """
    Explica una predicción del classifier_v2 agrupando las
    variables transformadas nuevamente a las 21 variables
    originales del vehículo.

    Retorna:
        - factores que aumentan la estimación de riesgo
        - factores que reducen la estimación de riesgo
    """

    X = datos_vehiculo[FEATURES].copy()

    # --------------------------------------------------------
    # TRANSFORMAR CON EL MISMO PREPROCESAMIENTO DEL MODELO
    # --------------------------------------------------------

    X_transformado = preprocessor.transform(X)

    if hasattr(X_transformado, "toarray"):
        X_transformado = X_transformado.toarray()

    # --------------------------------------------------------
    # CALCULAR SHAP
    # --------------------------------------------------------

    shap_values = explainer(X_transformado)

    valores_shap = shap_values.values[0]

    # Nombres generados por ColumnTransformer
    nombres_transformados = preprocessor.get_feature_names_out()

    # --------------------------------------------------------
    # IDENTIFICAR VARIABLES NUMÉRICAS Y CATEGÓRICAS
    # --------------------------------------------------------

    numeric_features = [
        "anio",
        "cilindraje_l",
        "kilometraje_acumulado_km",
        "km_desde_ultimo_mantenimiento",
        "dias_desde_ultimo_mantenimiento",
        "temperatura_promedio_motor_c",
        "presion_aceite_kpa",
        "frenadas_bruscas_por_100km",
        "tiempo_ralenti_min_dia",
        "altitud_operacion_m",
        "tiene_codigo_obd_activo",
        "costo_mantenimiento_actual_usd"
    ]

    categorical_features = [
        "marca",
        "modelo",
        "tipo_vehiculo",
        "combustible",
        "transmision",
        "zona_operacion_quito",
        "tipo_mantenimiento_actual",
        "sistema_intervenido_actual",
        "repuesto_cambiado_actual"
    ]

    # --------------------------------------------------------
    # AGRUPAR SHAP POR VARIABLE ORIGINAL
    # --------------------------------------------------------

    impactos = {}

    for variable in numeric_features:

        nombre_busqueda = f"num__{variable}"

        indices = [
            i
            for i, nombre in enumerate(nombres_transformados)
            if nombre == nombre_busqueda
        ]

        if indices:
            impactos[variable] = float(
                sum(valores_shap[i] for i in indices)
            )

    for variable in categorical_features:

        prefijo = f"cat__{variable}_"

        indices = [
            i
            for i, nombre in enumerate(nombres_transformados)
            if nombre.startswith(prefijo)
        ]

        if indices:
            impactos[variable] = float(
                sum(valores_shap[i] for i in indices)
            )

    # --------------------------------------------------------
    # CREAR RESULTADOS CON VALORES REALES
    # --------------------------------------------------------

    filas = []

    registro_original = X.iloc[0]

    for variable, impacto in impactos.items():

        valor = registro_original[variable]

        # Código OBD más comprensible
        if variable == "tiene_codigo_obd_activo":

            valor_mostrado = (
                "Sí"
                if int(valor) == 1
                else "No"
            )

        else:

            unidad = UNIDADES.get(variable, "")

            if isinstance(valor, (int, float, np.integer, np.floating)):

                if variable in [
                    "kilometraje_acumulado_km",
                    "km_desde_ultimo_mantenimiento"
                ]:
                    valor_mostrado = (
                        f"{float(valor):,.0f}{unidad}"
                    )

                elif variable == "costo_mantenimiento_actual_usd":
                    valor_mostrado = (
                        f"${float(valor):,.2f}"
                    )

                else:
                    valor_mostrado = (
                        f"{valor}{unidad}"
                    )

            else:
                valor_mostrado = str(valor)

        filas.append({
            "variable": variable,
            "factor": NOMBRES_AMIGABLES.get(
                variable,
                variable
            ),
            "valor": valor_mostrado,
            "shap": impacto,
            "direccion": (
                "Aumenta"
                if impacto > 0
                else "Reduce"
            )
        })

    resultado = pd.DataFrame(filas)

    # --------------------------------------------------------
    # FACTORES QUE AUMENTAN EL RIESGO
    # --------------------------------------------------------

    aumentan = (
        resultado[
            resultado["shap"] > 0
        ]
        .sort_values(
            "shap",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    # --------------------------------------------------------
    # FACTORES QUE REDUCEN EL RIESGO
    # --------------------------------------------------------

    reducen = (
        resultado[
            resultado["shap"] < 0
        ]
        .sort_values(
            "shap",
            ascending=True
        )
        .head(top_n)
        .copy()
    )

    return aumentan, reducen

# ============================================================
# PRUEBA LOCAL CON UIO-1001
# ============================================================

if __name__ == "__main__":

    vehiculo_prueba = df[
        df["vehiculo_id"].astype(str) == "UIO-1001"
    ].copy()

    if vehiculo_prueba.empty:
        print("No se encontró el vehículo UIO-1001.")

    else:
        X_prueba = vehiculo_prueba[FEATURES].copy()

        aumentan, reducen = explicar_prediccion(
            X_prueba,
            top_n=5
        )

        print("\n" + "=" * 70)
        print("EXPLICABILIDAD SHAP - UIO-1001")
        print("=" * 70)

        print("\nFACTORES QUE AUMENTAN LA ESTIMACIÓN DE RIESGO:")
        print(
        aumentan[
        ["factor", "valor", "shap"]
        ].to_string(index=False)
        )       

        print("\nFACTORES QUE REDUCEN LA ESTIMACIÓN DE RIESGO:")
        print(
        reducen[
        ["factor", "valor", "shap"]
        ].to_string(index=False)
        )