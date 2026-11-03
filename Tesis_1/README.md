# LenoX Garage — Sistema Inteligente de Mantenimiento Vehicular

## 1. Descripción del proyecto

LenoX Garage es un prototipo académico de apoyo a la toma de decisiones para el mantenimiento vehicular, desarrollado mediante técnicas de aprendizaje automático (Machine Learning), Inteligencia Artificial Explicable (XAI) e Inteligencia Artificial Generativa.

El sistema permite estimar la probabilidad de que un vehículo presente una falla en los próximos 30 días, calcular un costo orientativo condicionado a la ocurrencia de una falla, interpretar las predicciones mediante SHAP y generar recomendaciones preventivas utilizando Gemini.

La aplicación cuenta con una interfaz web desarrollada en Streamlit.

## 2. Funcionalidades principales

- Consulta de vehículos registrados y análisis predictivo.
- Registro y análisis de nuevos vehículos.
- Estimación del nivel de riesgo de falla.
- Estimación orientativa de costos asociados a posibles fallas.
- Explicación de predicciones mediante SHAP.
- Generación de recomendaciones preventivas con Gemini.
- Seguimiento de vehículos mediante registros separados del conjunto histórico.
- Funciones de evaluación y actualización controlada de modelos.
- Exploración experimental de cinco sistemas de falla mediante un clasificador de bajo rendimiento, cuyos resultados son orientativos y no constituyen diagnósticos validados.

## 3. Tecnologías utilizadas

- Python
- Streamlit
- Pandas y NumPy
- Scikit-learn
- SHAP
- Google Gemini API
- Joblib

## 4. Requisitos

- Python 3.14.6 (versión utilizada durante las pruebas locales).
- Acceso a internet para utilizar Gemini.
- Clave API de Gemini para habilitar las recomendaciones generativas.

Las dependencias de Python se encuentran en `requirements.txt`.

## 5. Instalación

Clonar el repositorio:

```bash
git clone https://github.com/KathyFlores789/Tesis_1.git
```

Acceder a la carpeta que contiene `requirements.txt`:

```bash
cd Tesis_1/Tesis_1
```

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activar el entorno virtual en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

## 6. Configuración de Gemini

Crear un archivo `.env` en la carpeta principal de la aplicación y configurar la variable de entorno correspondiente a la clave API de Gemini:

```dotenv
GEMINI_API_KEY=TU_CLAVE_API
```

Por seguridad, el archivo `.env` no debe subirse al repositorio.

## 7. Ejecución de la aplicación

Desde la carpeta que contiene `requirements.txt`, ejecutar:

```bash
python -m streamlit run app/app_v11.py
```

La aplicación se abrirá en el navegador, normalmente en:

`http://localhost:8501`

## 8. Estructura del proyecto

```text
app/          Aplicación web Streamlit
data/         Datos históricos y de seguimiento
models/       Modelos entrenados y metadatos
reports/      Resultados y reportes de evaluación
scripts/      Scripts auxiliares
src/          Procesamiento, predicción, SHAP e IA generativa
requirements.txt
README.md
```
## 9. Modelos utilizados

La aplicación Streamlit V11 integra tres modelos de aprendizaje automático almacenados en la carpeta `models/`, cada uno con una finalidad específica:

- **`classifier_v2.joblib`:** modelo principal de clasificación que estima la probabilidad de que un vehículo presente una falla durante los próximos 30 días.
- **`regressor_v2.joblib`:** modelo de regresión utilizado para calcular un costo orientativo condicionado a la ocurrencia de una falla.
- **`system_classifier_v1.joblib`:** modelo experimental de Regresión Logística que presenta una distribución estimada entre cinco posibles sistemas de falla.

Los modelos de clasificación principal y regresión son necesarios para el funcionamiento del análisis predictivo. El clasificador de sistemas de falla es opcional y se mantiene únicamente con fines exploratorios, debido a su rendimiento limitado.

Este componente experimental no modifica las predicciones principales ni se utiliza como fundamento de las recomendaciones generadas mediante Inteligencia Artificial Generativa. Sus resultados requieren validación adicional y no sustituyen la inspección técnica del vehículo.

## 10. Consideraciones y limitaciones

Las predicciones son estimaciones estadísticas y no constituyen un diagnóstico mecánico confirmado.

Las explicaciones SHAP representan contribuciones de las variables a la predicción del modelo y no demuestran causalidad.

Los costos estimados son orientativos y no representan cotizaciones del taller.

Las recomendaciones generadas por Gemini deben ser contrastadas con una inspección técnica realizada por personal calificado.

El módulo experimental de identificación de sistemas de falla utiliza un clasificador de Regresión Logística entrenado sobre registros con falla futura. En su evaluación final obtuvo una exactitud del 22,86 % y un F1-macro de 0,2282 sobre 70 casos de prueba. Debido a su rendimiento limitado y a la ausencia de calibración de las probabilidades, sus resultados se muestran únicamente con fines exploratorios. Este componente no modifica la predicción principal de riesgo ni la estimación de costos, y sus porcentajes no se transmiten directamente al generador de recomendaciones de Gemini desde la aplicación.

## 11. Finalidad académica

Este repositorio contiene el desarrollo de un proyecto de titulación de la Maestría en Inteligencia Artificial Aplicada, orientado al análisis predictivo y al apoyo de decisiones de mantenimiento vehicular.

## 12. Registro de versiones y control de cambios

El desarrollo de LenoX Garage se gestionó mediante Git y GitHub, utilizando la rama principal `main` para mantener un historial de modificaciones del código fuente, la interfaz y la documentación.

Entre los principales hitos registrados se encuentran:

| Identificador | Descripción del cambio |
|---|---|
| `09cec4a` | Registro de la versión inicial del proyecto. |
| `2eba2b4` | Desarrollo de la versión V2 de Streamlit e incorporación del ingreso de nuevos vehículos. |
| `e09a623` | Modificaciones relacionadas con Streamlit y el registro de vehículos. |
| `f79a02e` | Respaldo del proyecto previo a la incorporación de la versión V9. |
| `2f94035` y `e5eec1e` | Ajustes en el despliegue de información y porcentajes. |
| `3aa4729` | Ajustes de formato y simulación. |
| `8110189` y `5cb3af1` | Incorporación de figuras del proyecto. |
| `e4a2db9` y `6971ece` | Exclusión y eliminación de archivos que contenían credenciales. |
| `66f2970` | Actualización de documentación y eliminación de archivos temporales del seguimiento de Git. |

La aplicación utilizada en la evaluación del prototipo corresponde al archivo `app/app_v11.py`.

El historial completo de cambios puede consultarse en el repositorio:

https://github.com/KathyFlores789/Tesis_1/commits/main

Los identificadores anteriores corresponden a commits de Git y no necesariamente a versiones funcionales independientes de la aplicación.