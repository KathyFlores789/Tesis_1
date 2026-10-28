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

    # Contexto ampliado: datos existentes, sin inferir diagnósticos.
    campos_extra = {
        "Temperatura promedio del motor (°C)": "temperatura_promedio_motor_c",
        "Presión de aceite (kPa)": "presion_aceite_kpa",
        "Frenadas bruscas por 100 km": "frenadas_bruscas_por_100km",
        "Tiempo en ralentí (min/día)": "tiempo_ralenti_min_dia",
        "Altitud de operación (m)": "altitud_operacion_m",
        "Zona de operación": "zona_operacion_quito",
        "Tipo de mantenimiento actual": "tipo_mantenimiento_actual",
        "Sistema intervenido actualmente": "sistema_intervenido_actual",
        "Repuesto cambiado actualmente": "repuesto_cambiado_actual",
    }
    datos_extra = "\n".join(
        f"- {etiqueta}: {registro.get(columna, 'No disponible')}"
        for etiqueta, columna in campos_extra.items()
    )

    prompt = f"""
Eres un asistente técnico de apoyo para mantenimiento preventivo vehicular.
Redacta una respuesta personalizada en español, basada EXCLUSIVAMENTE en
los datos proporcionados y en el resultado del modelo. No tienes acceso al
manual del fabricante ni has inspeccionado físicamente el vehículo.

REGLAS OBLIGATORIAS:
- No diagnostiques averías ni afirmes que un componente está dañado.
- No predigas un sistema o repuesto futuro: ese modelo aún no está implementado.
- No inventes síntomas, códigos OBD específicos, valores de referencia del
  fabricante, periodicidades exactas ni costos adicionales.
- Conserva exactamente la probabilidad y el costo estimado indicados.
- El costo es orientativo y condicional a una eventual intervención; no es
  un presupuesto ni un gasto de mantenimiento preventivo confirmado.
- SHAP describe contribuciones a la predicción, NO causalidad ni evidencia
  de una avería concreta.
- Distingue los datos observados de las verificaciones que propones.
- Cuando un dato no permita decidir si un parámetro es normal, recomienda
  contrastarlo con el manual y las especificaciones del fabricante.
- Si no hay código OBD activo, no afirmes que existe un código de falla.
- No recomiendes reemplazar componentes sin inspección técnica.
- Adapta las recomendaciones al riesgo estimado, sin tratar un riesgo bajo
  como garantía de ausencia de averías.

DATOS DEL VEHÍCULO:
- Marca: {registro['marca']}
- Modelo: {registro['modelo']}
- Año: {registro['anio']}
- Tipo: {registro['tipo_vehiculo']}
- Combustible: {registro['combustible']}
- Kilometraje acumulado: {registro['kilometraje_acumulado_km']} km
- Km desde último mantenimiento: {registro['km_desde_ultimo_mantenimiento']} km
- Días desde último mantenimiento: {registro['dias_desde_ultimo_mantenimiento']}
- Código OBD activo: {obd}
{datos_extra}

RESULTADO DEL MODELO:
- Probabilidad de falla en los próximos 30 días: {probabilidad * 100:.1f}%
- Nivel de riesgo: {nivel_riesgo}
- Costo aproximado condicional: ${costo_estimado:,.2f} USD

FACTORES SHAP QUE AUMENTAN LA ESTIMACIÓN:
{texto_pos}

FACTORES SHAP QUE REDUCEN LA ESTIMACIÓN:
{texto_neg}

GENERA EXACTAMENTE ESTAS CUATRO SECCIONES, CON ESTOS ENCABEZADOS
EN MAYÚSCULAS Y SIN OTROS ENCABEZADOS DE PRIMER NIVEL:

RESUMEN:
En 2-3 frases indica vehículo, probabilidad EXACTA, nivel de riesgo y costo
EXACTO, aclarando que el costo es orientativo y condicional.

INTERPRETACIÓN:
Explica en un párrafo los principales factores SHAP, mencionando valores
reales cuando estén disponibles y sin atribuirles causalidad.

RECOMENDACIÓN PREVENTIVA:
Escribe entre 4 y 6 recomendaciones NUMERADAS, cada una en su propio párrafo.
En cada recomendación incluye: (a) componente o aspecto a inspeccionar,
(b) acción concreta y verificable por un mecánico, y (c) justificación
relacionada con un dato real del vehículo, cuando exista.
Prioriza los aspectos relacionados con los factores SHAP relevantes, pero
no interpretes automáticamente una contribución SHAP como anomalía física.
Considera cuando corresponda: historial e intervalo de mantenimiento,
lectura OBD si hay código activo, lubricación y presión de aceite,
refrigeración y temperatura, frenos y hábitos de conducción, neumáticos
y suspensión. No incluyas todos por rutina: selecciona lo pertinente.
Indica si la atención debe programarse o priorizarse para inspección
según los datos y el riesgo, sin inventar plazos ni urgencias clínicas.
No repitas recomendaciones genéricas ni propongas sustituir repuestos
sin diagnóstico. Extensión total aproximada: 180-260 palabras.

ADVERTENCIA:
En 2-3 frases aclara límites del modelo y de SHAP, y la necesidad de
inspección técnica y consulta del manual del fabricante.

Usa texto simple, sin tablas ni formato Markdown complejo.
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
                    print(f"MODELO GEMINI UTILIZADO: {modelo_nombre}")
                    print(f"MODELO REPORTADO POR API: {getattr(response, 'model_version', 'No disponible')}")
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
