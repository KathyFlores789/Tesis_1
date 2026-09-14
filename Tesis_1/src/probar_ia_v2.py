import pandas as pd

from generador_ia_v2 import generar_recomendacion_ia


# ============================================================
# VEHÍCULO DE PRUEBA: UIO-1001
# ============================================================

datos_vehiculo = pd.DataFrame([{
    "marca": "Volkswagen",
    "modelo": "Gol",
    "anio": 2013,
    "tipo_vehiculo": "Automóvil",
    "combustible": "Gasolina",
    "kilometraje_acumulado_km": 50000,
    "km_desde_ultimo_mantenimiento": 5000,
    "dias_desde_ultimo_mantenimiento": 120,
    "tiene_codigo_obd_activo": 0
}])


# ============================================================
# RESULTADO DEL MODELO
# ============================================================

probabilidad = 0.634
nivel_riesgo = "ALTO"


# ============================================================
# RESULTADOS SHAP
# ============================================================

factores_aumentan = pd.DataFrame([
    {
        "factor": "Año",
        "valor": "2013",
        "shap": 0.512792
    },
    {
        "factor": "Km desde último mantenimiento",
        "valor": "5,000 km",
        "shap": 0.489191
    },
    {
        "factor": "Repuesto cambiado",
        "valor": "Alineación y balanceo",
        "shap": 0.206410
    }
])


factores_reducen = pd.DataFrame([
    {
        "factor": "Código OBD activo",
        "valor": "No",
        "shap": -0.260795
    },
    {
        "factor": "Tiempo en ralentí por día",
        "valor": "20.0 min/día",
        "shap": -0.118573
    },
    {
        "factor": "Zona de operación",
        "valor": "Centro/Hipercentro",
        "shap": -0.114188
    }
])


# ============================================================
# LLAMADA REAL A GEMINI
# ============================================================

print("=" * 70)
print("PRUEBA DE IA GENERATIVA")
print("=" * 70)

try:

    respuesta = generar_recomendacion_ia(
        datos_vehiculo=datos_vehiculo,
        probabilidad=probabilidad,
        nivel_riesgo=nivel_riesgo,
        factores_aumentan=factores_aumentan,
        factores_reducen=factores_reducen
    )

    print("\nRESPUESTA GENERADA:\n")
    print(respuesta)

except Exception as error:

    print("\nERROR AL GENERAR LA RECOMENDACIÓN:")
    print(error)