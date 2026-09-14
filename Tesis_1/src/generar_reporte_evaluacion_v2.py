from pathlib import Path

import pandas as pd

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import (
    WD_TABLE_ALIGNMENT,
    WD_CELL_VERTICAL_ALIGNMENT
)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "datos_mantenimiento_quito.csv"
)

REPORTS_DIR = (
    ROOT_DIR
    / "reports"
)

OUTPUT_PATH = (
    REPORTS_DIR
    / "Reporte_Evaluacion_Modelos_V2.docx"
)


# ============================================================
# ARCHIVOS DE ENTRADA
# ============================================================

CLASSIFIER_METRICS = (
    REPORTS_DIR
    / "metricas_clasificador_v2_test.csv"
)

CONFUSION_IMAGE = (
    REPORTS_DIR
    / "matriz_confusion_clasificador_v2.png"
)

REGRESSOR_METRICS = (
    REPORTS_DIR
    / "metricas_regresor_v2_test.csv"
)

REGRESSOR_PREDICTIONS = (
    REPORTS_DIR
    / "predicciones_regresor_v2_test.csv"
)

REAL_VS_ESTIMADO_IMAGE = (
    REPORTS_DIR
    / "real_vs_estimado_regresor_v2.png"
)

ERRORS_IMAGE = (
    REPORTS_DIR
    / "errores_regresor_v2.png"
)

SHAP_GLOBAL = (
    REPORTS_DIR
    / "shap_importancia_global_v2.csv"
)

SHAP_GLOBAL_IMAGE = (
    REPORTS_DIR
    / "shap_importancia_global_v2.png"
)

SHAP_LOCAL = (
    REPORTS_DIR
    / "shap_explicaciones_locales_v2.csv"
)

SHAP_METRICS = (
    REPORTS_DIR
    / "metricas_shap_v2.csv"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def configurar_documento(documento):

    estilos = documento.styles

    estilo_normal = estilos["Normal"]
    estilo_normal.font.name = "Arial"
    estilo_normal.font.size = Pt(11)

    for nombre in [
        "Title",
        "Heading 1",
        "Heading 2",
        "Heading 3"
    ]:
        estilos[nombre].font.name = "Arial"

    estilos["Title"].font.size = Pt(18)
    estilos["Heading 1"].font.size = Pt(15)
    estilos["Heading 2"].font.size = Pt(13)


def sombrear_celda(cell, fill):

    tc_pr = cell._tc.get_or_add_tcPr()

    shd = OxmlElement("w:shd")

    shd.set(
        qn("w:fill"),
        fill
    )

    tc_pr.append(shd)


def agregar_tabla(
    documento,
    columnas,
    filas,
    encabezado_color="D9EAF7"
):

    tabla = documento.add_table(
        rows=1,
        cols=len(columnas)
    )

    tabla.alignment = (
        WD_TABLE_ALIGNMENT.CENTER
    )

    tabla.style = "Table Grid"

    encabezado = tabla.rows[0].cells

    for i, columna in enumerate(columnas):

        encabezado[i].text = str(columna)

        sombrear_celda(
            encabezado[i],
            encabezado_color
        )

        encabezado[i].vertical_alignment = (
            WD_CELL_VERTICAL_ALIGNMENT.CENTER
        )

        for parrafo in encabezado[i].paragraphs:

            for run in parrafo.runs:

                run.bold = True
                run.font.name = "Arial"
                run.font.size = Pt(10)

    for fila in filas:

        celdas = tabla.add_row().cells

        for i, valor in enumerate(fila):

            celdas[i].text = str(valor)

            celdas[i].vertical_alignment = (
                WD_CELL_VERTICAL_ALIGNMENT.CENTER
            )

            for parrafo in celdas[i].paragraphs:

                for run in parrafo.runs:

                    run.font.name = "Arial"
                    run.font.size = Pt(10)

    documento.add_paragraph()

    return tabla


def agregar_imagen(
    documento,
    ruta,
    titulo,
    ancho=6.2
):

    if not ruta.exists():

        documento.add_paragraph(
            f"[Imagen no disponible: {ruta.name}]"
        )

        return

    p = documento.add_paragraph()

    p.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    run = p.add_run()

    run.add_picture(
        str(ruta),
        width=Inches(ancho)
    )

    caption = documento.add_paragraph(
        titulo
    )

    caption.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    for run in caption.runs:

        run.italic = True
        run.font.size = Pt(9)

    documento.add_paragraph()


def agregar_texto_justificado(
    documento,
    texto
):

    p = documento.add_paragraph(
        texto
    )

    p.alignment = (
        WD_ALIGN_PARAGRAPH.JUSTIFY
    )

    return p


# ============================================================
# VALIDACIÓN DE ARCHIVOS
# ============================================================

archivos_necesarios = [
    DATA_PATH,
    CLASSIFIER_METRICS,
    REGRESSOR_METRICS,
    SHAP_GLOBAL,
    SHAP_METRICS
]

faltantes = [
    archivo
    for archivo in archivos_necesarios
    if not archivo.exists()
]

if faltantes:

    print(
        "No es posible generar el reporte."
    )

    print(
        "Faltan los siguientes archivos:"
    )

    for archivo in faltantes:

        print(
            "-",
            archivo
        )

    raise SystemExit


# ============================================================
# CARGA DE RESULTADOS
# ============================================================

df_classifier = pd.read_csv(
    CLASSIFIER_METRICS
)

df_regressor = pd.read_csv(
    REGRESSOR_METRICS
)

df_shap = pd.read_csv(
    SHAP_GLOBAL
)

df_shap_metrics = pd.read_csv(
    SHAP_METRICS
)


# ============================================================
# CARGA DEL DATASET ORIGINAL
# ============================================================

df_datos = pd.read_csv(
    DATA_PATH
)

df_datos["fecha_mantenimiento"] = pd.to_datetime(
    df_datos["fecha_mantenimiento"],
    errors="coerce"
)

df_datos = (
    df_datos
    .sort_values(
        "fecha_mantenimiento"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# EXTRAER MÉTRICAS
# ============================================================

clf = df_classifier.iloc[0]

reg = df_regressor.iloc[0]

shap_metric = (
    df_shap_metrics.iloc[0]
)


# ============================================================
# CREAR DOCUMENTO
# ============================================================

doc = Document()

configurar_documento(
    doc
)


# ============================================================
# PORTADA
# ============================================================

titulo = doc.add_paragraph()

titulo.alignment = (
    WD_ALIGN_PARAGRAPH.CENTER
)

run = titulo.add_run(
    "REPORTE DE EVALUACIÓN DEL MODELO PREDICTIVO V2"
)

run.bold = True
run.font.name = "Arial"
run.font.size = Pt(18)


subtitulo = doc.add_paragraph()

subtitulo.alignment = (
    WD_ALIGN_PARAGRAPH.CENTER
)

run = subtitulo.add_run(
    "Sistema Predictivo de Mantenimiento Vehicular"
)

run.bold = True
run.font.size = Pt(14)


doc.add_paragraph()


p = doc.add_paragraph()

p.alignment = (
    WD_ALIGN_PARAGRAPH.CENTER
)

p.add_run(
    "Evaluación del clasificador de riesgo, "
    "estimación aproximada de costos y "
    "explicabilidad mediante SHAP."
)


doc.add_page_break()


# ============================================================
# 1. INTRODUCCIÓN
# ============================================================

doc.add_heading(
    "1. Introducción",
    level=1
)

agregar_texto_justificado(
    doc,
    "El presente documento consolida los resultados de evaluación "
    "de los componentes principales del sistema predictivo de "
    "mantenimiento vehicular. La evaluación comprende el modelo "
    "de clasificación utilizado para estimar la probabilidad de "
    "falla vehicular en los próximos 30 días, el modelo de regresión "
    "empleado para obtener una referencia aproximada del costo de "
    "una eventual intervención y el componente de explicabilidad "
    "basado en SHAP."
)

agregar_texto_justificado(
    doc,
    "Las métricas corresponden a conjuntos de prueba separados "
    "del proceso de entrenamiento. Los vehículos añadidos "
    "posteriormente desde la aplicación no fueron utilizados "
    "para la evaluación supervisada cuando no disponían de una "
    "etiqueta real de falla futura."
)


# ============================================================
# 2. RESUMEN GENERAL
# ============================================================

doc.add_heading(
    "2. Resumen de resultados",
    level=1
)


filas_resumen = [
    [
        "Clasificador",
        "Accuracy",
        f"{clf['accuracy']:.4f}"
    ],
    [
        "Clasificador",
        "Precision",
        f"{clf['precision']:.4f}"
    ],
    [
        "Clasificador",
        "Recall",
        f"{clf['recall']:.4f}"
    ],
    [
        "Clasificador",
        "F1-score",
        f"{clf['f1_score']:.4f}"
    ],
    [
        "Clasificador",
        "ROC-AUC",
        f"{clf['roc_auc']:.4f}"
    ],
    [
        "Regresor",
        "MAE",
        f"${reg['mae']:.2f}"
    ],
    [
        "Regresor",
        "RMSE",
        f"${reg['rmse']:.2f}"
    ],
    [
        "Regresor",
        "R²",
        f"{reg['r2']:.4f}"
    ],
    [
        "Regresor",
        "Error absoluto mediano",
        f"${reg['error_absoluto_mediano']:.2f}"
    ],
    [
        "SHAP",
        "Error medio de reconstrucción",
        f"{shap_metric['error_reconstruccion_medio']:.10f}"
    ],
    [
        "SHAP",
        "Error máximo de reconstrucción",
        f"{shap_metric['error_reconstruccion_max']:.10f}"
    ]
]


agregar_tabla(
    doc,
    [
        "Componente",
        "Métrica",
        "Resultado"
    ],
    filas_resumen
)


# ============================================================
# 3. CLASIFICADOR
# ============================================================

doc.add_heading(
    "3. Evaluación del clasificador de riesgo",
    level=1
)

agregar_texto_justificado(
    doc,
    "El clasificador estima la probabilidad de que un vehículo "
    "presente una falla durante los próximos 30 días. Para la "
    "clasificación final se utilizó un umbral de decisión de 0.50."
)


filas_clasificador = [
    [
        "Registros de prueba",
        int(clf["n_test"])
    ],
    [
        "Umbral",
        f"{clf['umbral']:.2f}"
    ],
    [
        "Accuracy",
        f"{clf['accuracy']:.4f}"
    ],
    [
        "Precision",
        f"{clf['precision']:.4f}"
    ],
    [
        "Recall",
        f"{clf['recall']:.4f}"
    ],
    [
        "F1-score",
        f"{clf['f1_score']:.4f}"
    ],
    [
        "ROC-AUC",
        f"{clf['roc_auc']:.4f}"
    ]
]


agregar_tabla(
    doc,
    [
        "Métrica",
        "Resultado"
    ],
    filas_clasificador
)


# ============================================================
# 3.1 MATRIZ DE CONFUSIÓN
# ============================================================

doc.add_heading(
    "3.1 Matriz de confusión",
    level=2
)


filas_confusion = [
    [
        "Verdadero negativo (TN)",
        int(clf["tn"])
    ],
    [
        "Falso positivo (FP)",
        int(clf["fp"])
    ],
    [
        "Falso negativo (FN)",
        int(clf["fn"])
    ],
    [
        "Verdadero positivo (TP)",
        int(clf["tp"])
    ]
]


agregar_tabla(
    doc,
    [
        "Resultado",
        "Cantidad"
    ],
    filas_confusion
)


agregar_imagen(
    doc,
    CONFUSION_IMAGE,
    "Figura 1. Matriz de confusión del clasificador V2."
)


doc.add_heading(
    "3.2 Interpretación",
    level=2
)


agregar_texto_justificado(
    doc,
    f"El modelo alcanzó una exactitud de {clf['accuracy']:.4f}, "
    f"una precisión de {clf['precision']:.4f}, un recall de "
    f"{clf['recall']:.4f} y un F1-score de {clf['f1_score']:.4f}. "
    f"El ROC-AUC de {clf['roc_auc']:.4f} evidencia una capacidad "
    "de discriminación moderada entre vehículos con y sin falla."
)


agregar_texto_justificado(
    doc,
    f"La matriz de confusión registró {int(clf['tp'])} verdaderos "
    f"positivos y {int(clf['tn'])} verdaderos negativos. Se "
    f"observaron además {int(clf['fp'])} falsos positivos y "
    f"{int(clf['fn'])} falsos negativos. Los falsos negativos "
    "son especialmente relevantes en mantenimiento predictivo, "
    "porque representan fallas reales que no fueron identificadas "
    "anticipadamente."
)


# ============================================================
# 4. REGRESOR
# ============================================================

doc.add_heading(
    "4. Evaluación de la estimación aproximada de costos",
    level=1
)


agregar_texto_justificado(
    doc,
    "El componente de regresión estima de manera aproximada el "
    "costo asociado a una eventual intervención en registros "
    "donde existe una falla futura. El modelo fue entrenado "
    "mediante una transformación logarítmica del costo y las "
    "predicciones fueron transformadas nuevamente a dólares."
)


filas_regresor = [
    [
        "Registros de prueba",
        int(reg["n_test"])
    ],
    [
        "MAE",
        f"${reg['mae']:.2f}"
    ],
    [
        "RMSE",
        f"${reg['rmse']:.2f}"
    ],
    [
        "R²",
        f"{reg['r2']:.4f}"
    ],
    [
        "Error absoluto mediano",
        f"${reg['error_absoluto_mediano']:.2f}"
    ]
]


agregar_tabla(
    doc,
    [
        "Métrica",
        "Resultado"
    ],
    filas_regresor
)


agregar_imagen(
    doc,
    REAL_VS_ESTIMADO_IMAGE,
    "Figura 2. Comparación entre costo real y costo estimado."
)


agregar_imagen(
    doc,
    ERRORS_IMAGE,
    "Figura 3. Distribución del error absoluto del regresor."
)


doc.add_heading(
    "4.1 Interpretación",
    level=2
)


agregar_texto_justificado(
    doc,
    f"El error absoluto medio obtenido fue de ${reg['mae']:.2f}, "
    f"mientras que el RMSE alcanzó ${reg['rmse']:.2f}. "
    f"El error absoluto mediano fue de "
    f"${reg['error_absoluto_mediano']:.2f}."
)


agregar_texto_justificado(
    doc,
    f"El coeficiente de determinación R² fue de {reg['r2']:.4f}, "
    "lo que evidencia una capacidad limitada para explicar la "
    "variabilidad total de los costos observados. Por esta razón, "
    "el resultado económico se presenta en la aplicación como "
    "un costo aproximado y orientativo."
)


# ============================================================
# 5. SHAP
# ============================================================

doc.add_heading(
    "5. Evaluación de explicabilidad mediante SHAP",
    level=1
)


agregar_texto_justificado(
    doc,
    "SHAP se empleó para identificar la contribución de las "
    "variables utilizadas por el clasificador sobre cada "
    "predicción. Los valores positivos indican factores que "
    "incrementan la estimación del riesgo, mientras que los "
    "valores negativos indican factores que reducen dicha "
    "estimación. Estas contribuciones explican el comportamiento "
    "del modelo y no deben interpretarse como relaciones causales."
)


doc.add_heading(
    "5.1 Variables con mayor importancia global",
    level=2
)


top_shap = (
    df_shap
    .head(15)
    .copy()
)


filas_shap = []


for _, row in top_shap.iterrows():

    filas_shap.append([
        row["feature"],
        f"{row['mean_abs_shap']:.6f}"
    ])


agregar_tabla(
    doc,
    [
        "Variable",
        "Impacto SHAP medio absoluto"
    ],
    filas_shap
)


agregar_imagen(
    doc,
    SHAP_GLOBAL_IMAGE,
    "Figura 4. Importancia global de variables mediante SHAP."
)


# ============================================================
# 5.2 CONSISTENCIA SHAP
# ============================================================

doc.add_heading(
    "5.2 Consistencia de la explicación",
    level=2
)


filas_consistencia = [
    [
        "Registros de prueba",
        int(shap_metric["n_test"])
    ],
    [
        "Variables transformadas",
        int(
            shap_metric[
                "n_features_transformadas"
            ]
        )
    ],
    [
        "Error medio de reconstrucción",
        f"{shap_metric['error_reconstruccion_medio']:.10f}"
    ],
    [
        "Error máximo de reconstrucción",
        f"{shap_metric['error_reconstruccion_max']:.10f}"
    ]
]


agregar_tabla(
    doc,
    [
        "Indicador",
        "Resultado"
    ],
    filas_consistencia
)


agregar_texto_justificado(
    doc,
    "El error medio y el error máximo de reconstrucción fueron "
    "iguales a cero dentro de la precisión numérica utilizada. "
    "Esto evidencia que las contribuciones SHAP junto con el "
    "valor base reconstruyen correctamente la salida del "
    "clasificador en su espacio de decisión."
)


# ============================================================
# 6. EJEMPLOS LOCALES SHAP
# ============================================================

if SHAP_LOCAL.exists():

    df_local = pd.read_csv(
        SHAP_LOCAL
    )

    doc.add_heading(
        "6. Ejemplos de explicaciones locales SHAP",
        level=1
    )

    agregar_texto_justificado(
        doc,
        "A continuación se presentan ejemplos de explicaciones "
        "locales correspondientes a registros reales del conjunto "
        "de prueba. Cada caso identifica el vehículo evaluado, "
        "el resultado real, la predicción realizada, la probabilidad "
        "estimada de falla y los factores que tuvieron mayor "
        "influencia sobre la decisión del modelo."
    )


    nombres_amigables = {

        "km_desde_ultimo_mantenimiento":
            "Km desde último mantenimiento",

        "tiene_codigo_obd_activo":
            "Código OBD activo",

        "anio":
            "Año",

        "presion_aceite_kpa":
            "Presión de aceite",

        "frenadas_bruscas_por_100km":
            "Frenadas bruscas por 100 km",

        "temperatura_promedio_motor_c":
            "Temperatura promedio del motor",

        "kilometraje_acumulado_km":
            "Kilometraje acumulado",

        "tiempo_ralenti_min_dia":
            "Tiempo en ralentí",

        "altitud_operacion_m":
            "Altitud de operación",

        "cilindraje_l":
            "Cilindraje",

        "costo_mantenimiento_actual_usd":
            "Costo del mantenimiento actual"
    }


    n_ejemplos = min(
        5,
        len(df_local)
    )


    for i in range(n_ejemplos):

        fila = df_local.iloc[i]

        indice_original = int(
            fila["indice_test"]
        )

        registro = df_datos.iloc[
            indice_original
        ]


        vehiculo_id = registro.get(
            "vehiculo_id",
            "No disponible"
        )

        marca = registro.get(
            "marca",
            "No disponible"
        )

        modelo = registro.get(
            "modelo",
            "No disponible"
        )

        anio = registro.get(
            "anio",
            "No disponible"
        )


        doc.add_heading(
            f"6.{i + 1} Caso de prueba {i + 1} — "
            f"{vehiculo_id} | "
            f"{marca} {modelo} {anio}",
            level=2
        )


        valor_real = (
            "Falla"
            if int(fila["valor_real"]) == 1
            else "Sin falla"
        )

        prediccion = (
            "Falla"
            if int(fila["prediccion"]) == 1
            else "Sin falla"
        )


        obd = registro.get(
            "tiene_codigo_obd_activo",
            "No disponible"
        )


        if pd.notna(obd):

            try:

                obd = (
                    "Sí"
                    if int(obd) == 1
                    else "No"
                )

            except Exception:
                pass


        datos_caso = [
            [
                "ID del vehículo",
                vehiculo_id
            ],
            [
                "Marca",
                marca
            ],
            [
                "Modelo",
                modelo
            ],
            [
                "Año",
                anio
            ],
            [
                "Tipo de vehículo",
                registro.get(
                    "tipo_vehiculo",
                    "No disponible"
                )
            ],
            [
                "Combustible",
                registro.get(
                    "combustible",
                    "No disponible"
                )
            ],
            [
                "Kilometraje acumulado",
                f"{registro.get('kilometraje_acumulado_km', 'N/D')} km"
            ],
            [
                "Km desde último mantenimiento",
                f"{registro.get('km_desde_ultimo_mantenimiento', 'N/D')} km"
            ],
            [
                "Días desde último mantenimiento",
                registro.get(
                    "dias_desde_ultimo_mantenimiento",
                    "N/D"
                )
            ],
            [
                "Código OBD activo",
                obd
            ],
            [
                "Valor real",
                valor_real
            ],
            [
                "Predicción del modelo",
                prediccion
            ],
            [
                "Probabilidad de falla",
                f"{fila['probabilidad_falla'] * 100:.2f}%"
            ]
        ]


        agregar_tabla(
            doc,
            [
                "Elemento",
                "Valor"
            ],
            datos_caso
        )


        factores = []


        for j in range(
            1,
            6
        ):

            factor_col = (
                f"factor_{j}"
            )

            shap_col = (
                f"shap_{j}"
            )


            if (
                factor_col in fila.index
                and shap_col in fila.index
            ):

                factor_original = str(
                    fila[factor_col]
                )

                valor_shap = float(
                    fila[shap_col]
                )


                nombre_factor = (
                    nombres_amigables.get(
                        factor_original,
                        factor_original
                        .replace("_", " ")
                        .capitalize()
                    )
                )


                direccion = (
                    "Aumenta la estimación"
                    if valor_shap > 0
                    else "Reduce la estimación"
                )


                factores.append([
                    nombre_factor,
                    f"{valor_shap:+.6f}",
                    direccion
                ])


        agregar_tabla(
            doc,
            [
                "Factor explicativo",
                "Valor SHAP",
                "Efecto sobre la estimación"
            ],
            factores
        )


        agregar_texto_justificado(
            doc,
            f"El vehículo {marca} {modelo}, año {anio}, obtuvo "
            f"una probabilidad estimada de falla del "
            f"{fila['probabilidad_falla'] * 100:.2f} %. "
            f"El valor real registrado fue '{valor_real}' y el "
            f"modelo predijo '{prediccion}'. Los factores SHAP "
            f"presentados muestran cuáles variables incrementaron "
            f"o redujeron la estimación de riesgo para esta "
            f"observación particular. Estos valores explican la "
            f"decisión matemática del modelo y no representan "
            f"causas directas de una falla."
        )


# ============================================================
# 7. HALLAZGOS
# ============================================================

doc.add_heading(
    "7. Hallazgos principales",
    level=1
)


hallazgos = [
    (
        "El clasificador presenta una capacidad predictiva "
        "moderada, con ROC-AUC superior a 0.75."
    ),
    (
        "La precisión es superior al recall, por lo que el "
        "modelo presenta mayor confiabilidad cuando identifica "
        "un caso positivo que capacidad para detectar la totalidad "
        "de fallas reales."
    ),
    (
        "La estimación de costos presenta un error moderado en "
        "términos absolutos, pero un R² bajo, lo cual limita su "
        "uso como estimación económica precisa."
    ),
    (
        "SHAP proporciona explicaciones matemáticamente "
        "consistentes con la salida del clasificador."
    ),
    (
        "Los kilómetros desde el último mantenimiento, la "
        "presencia de códigos OBD activos, las frenadas bruscas, "
        "la presión de aceite y el año del vehículo aparecen "
        "entre las variables con mayor influencia global."
    )
]


for hallazgo in hallazgos:

    p = doc.add_paragraph(
        style="List Bullet"
    )

    p.add_run(
        hallazgo
    )


# ============================================================
# 8. LIMITACIONES
# ============================================================

doc.add_heading(
    "8. Limitaciones",
    level=1
)


limitaciones = [
    (
        "El desempeño del clasificador no elimina la presencia "
        "de falsos positivos ni falsos negativos."
    ),
    (
        "El recall obtenido muestra que una proporción de las "
        "fallas reales no es identificada por el modelo."
    ),
    (
        "El bajo R² del regresor limita la precisión de la "
        "estimación de costos."
    ),
    (
        "Los valores SHAP representan contribuciones sobre la "
        "predicción y no relaciones causales con una falla."
    ),
    (
        "Las estimaciones deben utilizarse como apoyo para la "
        "toma de decisiones y complementarse con revisión técnica."
    )
]


for limitacion in limitaciones:

    p = doc.add_paragraph(
        style="List Bullet"
    )

    p.add_run(
        limitacion
    )


# ============================================================
# 9. CONCLUSIONES
# ============================================================

doc.add_heading(
    "9. Conclusiones de la evaluación",
    level=1
)


agregar_texto_justificado(
    doc,
    "La evaluación permite concluir que el clasificador V2 "
    "presenta capacidad útil para discriminar entre vehículos "
    "con y sin riesgo de falla, aunque mantiene oportunidades "
    "de mejora principalmente relacionadas con la detección de "
    "falsos negativos."
)


agregar_texto_justificado(
    doc,
    "El componente de estimación económica proporciona una "
    "referencia aproximada que puede complementar la información "
    "del sistema, pero su bajo coeficiente R² impide interpretarlo "
    "como una predicción precisa del costo final."
)


agregar_texto_justificado(
    doc,
    "El componente SHAP presenta consistencia matemática y permite "
    "identificar tanto factores globales como locales asociados "
    "a las predicciones. Esto facilita que los resultados del "
    "modelo puedan presentarse de manera transparente y "
    "comprensible para los usuarios del sistema."
)


# ============================================================
# GUARDAR DOCUMENTO
# ============================================================

doc.save(
    OUTPUT_PATH
)


print("=" * 70)
print("REPORTE CONSOLIDADO GENERADO")
print("=" * 70)

print(
    "\nDocumento guardado en:"
)

print(
    OUTPUT_PATH
)

print("\n" + "=" * 70)