from pathlib import Path
import os
import time

from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "No se encontró GEMINI_API_KEY en el archivo .env"
    )

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# GENERACIÓN DE RECOMENDACIÓN
# ============================================================

def generar_recomendacion_ia(
    datos_vehiculo,
    probabilidad,
    nivel_riesgo,
    costo_estimado,
    factores_aumentan,
    factores_reducen
):
    """
    Genera una explicación en lenguaje natural utilizando:
    - datos del vehículo
    - probabilidad del clasificador
    - nivel de riesgo
    - costo aproximado del regresor
    - resultados SHAP

    La IA no debe inventar diagnósticos, averías, repuestos
    futuros ni costos distintos al valor entregado por el modelo.
    """

    registro = datos_vehiculo.iloc[0]

    # ========================================================
    # PREPARAR FACTORES SHAP POSITIVOS
    # ========================================================

    factores_pos = []

    for _, fila in factores_aumentan.iterrows():
        factores_pos.append(
            f"- {fila['factor']}: {fila['valor']} "
            f"(impacto SHAP {fila['shap']:+.4f})"
        )

    if factores_pos:
        texto_pos = "\n".join(factores_pos)
    else:
        texto_pos = "No se identificaron factores positivos relevantes."

    # ========================================================
    # PREPARAR FACTORES SHAP NEGATIVOS
    # ========================================================

    factores_neg = []

    for _, fila in factores_reducen.iterrows():
        factores_neg.append(
            f"- {fila['factor']}: {fila['valor']} "
            f"(impacto SHAP {fila['shap']:+.4f})"
        )

    if factores_neg:
        texto_neg = "\n".join(factores_neg)
    else:
        texto_neg = "No se identificaron factores negativos relevantes."

    # ========================================================
    # ESTADO DEL CÓDIGO OBD
    # ========================================================

    try:
        obd = (
            "Sí"
            if int(registro["tiene_codigo_obd_activo"]) == 1
            else "No"
        )
    except Exception:
        obd = str(
            registro.get(
                "tiene_codigo_obd_activo",
                "No disponible"
            )
        )

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
Eres un asistente técnico de apoyo para mantenimiento vehicular.

Debes explicar de manera clara, prudente y comprensible
el resultado de un modelo predictivo de riesgo de falla.

REGLAS OBLIGATORIAS:

1. No inventes una falla específica.
2. No afirmes que un componente está dañado.
3. No inventes repuestos futuros.
4. No inventes costos de reparación.
5. Utiliza exactamente la probabilidad entregada por el clasificador.
6. Utiliza exactamente el costo aproximado entregado por el regresor.
7. Presenta el costo como una estimación orientativa y condicional
   a que ocurra una falla o se requiera una intervención.
8. No afirmes que SHAP identifica causas directas.
9. Explica que SHAP muestra cómo las variables influyeron
   en la estimación realizada por el modelo.
10. Las recomendaciones deben ser preventivas y generales.
11. El resultado debe presentarse como apoyo para la toma
    de decisiones y no como un diagnóstico mecánico definitivo.
12. Indica que se recomienda complementar el resultado
    con una inspección técnica profesional.

DATOS DEL VEHÍCULO:

Marca:
{registro["marca"]}

Modelo:
{registro["modelo"]}

Año:
{registro["anio"]}

Tipo de vehículo:
{registro["tipo_vehiculo"]}

Combustible:
{registro["combustible"]}

Kilometraje acumulado:
{registro["kilometraje_acumulado_km"]} km

Km desde último mantenimiento:
{registro["km_desde_ultimo_mantenimiento"]} km

Días desde último mantenimiento:
{registro["dias_desde_ultimo_mantenimiento"]}

Código OBD activo:
{obd}

RESULTADO DEL MODELO:

Probabilidad estimada de falla en los próximos 30 días:
{probabilidad * 100:.1f} %

Nivel de riesgo:
{nivel_riesgo}

Costo aproximado de una eventual intervención:
${costo_estimado:,.2f} USD

NOTA SOBRE EL COSTO:
El costo es una estimación orientativa del regresor y no representa
un presupuesto, una factura ni un valor garantizado.

FACTORES SHAP QUE AUMENTAN LA ESTIMACIÓN DE RIESGO:

{texto_pos}

FACTORES SHAP QUE REDUCEN LA ESTIMACIÓN DE RIESGO:

{texto_neg}

GENERA LA RESPUESTA EXACTAMENTE CON ESTAS CUATRO SECCIONES:

RESUMEN:
Explica el resultado del modelo en 2 o 3 frases.
Incluye exactamente la probabilidad y el costo aproximado.
Aclara que el costo es orientativo.

INTERPRETACIÓN:
Explica los factores SHAP más relevantes.
No afirmes causalidad directa.

RECOMENDACIÓN PREVENTIVA:
Propón acciones prudentes de revisión preventiva.
No diagnostiques una avería específica.

ADVERTENCIA:
Aclara que tanto el riesgo como el costo son estimaciones predictivas,
sirven como apoyo para la toma de decisiones y deben complementarse
con una inspección técnica profesional.

Escribe en español claro y profesional.
"""

    # ========================================================
    # MODELOS DE RESPALDO
    # ========================================================

    modelos = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash"
    ]

    ultimo_error = None

    # ========================================================
    # GENERACIÓN CON REINTENTOS
    # ========================================================

    for modelo_nombre in modelos:

        for intento in range(3):

            try:
                response = client.models.generate_content(
                    model=modelo_nombre,
                    contents=prompt
                )

                if response.text:
                    return response.text.strip()

                ultimo_error = RuntimeError(
                    "Gemini respondió sin contenido."
                )

            except Exception as error:
                ultimo_error = error
                mensaje = str(error)

                if (
                    "503" in mensaje
                    or "UNAVAILABLE" in mensaje
                    or "high demand" in mensaje.lower()
                ):
                    espera = 2 ** intento

                    print(
                        f"{modelo_nombre} no disponible. "
                        f"Reintentando en {espera} segundos..."
                    )

                    time.sleep(espera)
                    continue

                if (
                    "404" in mensaje
                    or "NOT_FOUND" in mensaje
                    or "not found" in mensaje.lower()
                ):
                    print(
                        f"{modelo_nombre} no está disponible. "
                        f"Probando otro modelo..."
                    )
                    break

                raise RuntimeError(
                    "Error durante la generación con Gemini: "
                    f"{error}"
                ) from error

    raise RuntimeError(
        "No fue posible generar la recomendación "
        "con ninguno de los modelos configurados. "
        f"Último error: {ultimo_error}"
    )


# ============================================================
# PRUEBA DE CARGA
# ============================================================

if __name__ == "__main__":
    print(
        "Componente de IA Generativa "
        "cargado correctamente."
    )
