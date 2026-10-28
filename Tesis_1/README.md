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

## 9. Consideraciones y limitaciones

Las predicciones son estimaciones estadísticas y no constituyen un diagnóstico mecánico confirmado.

Las explicaciones SHAP representan contribuciones de las variables a la predicción del modelo y no demuestran causalidad.

Los costos estimados son orientativos y no representan cotizaciones del taller.

Las recomendaciones generadas por Gemini deben ser contrastadas con una inspección técnica realizada por personal calificado.

## 10. Finalidad académica

Este repositorio contiene el desarrollo de un proyecto de titulación de la Maestría en Inteligencia Artificial Aplicada, orientado al análisis predictivo y al apoyo de decisiones de mantenimiento vehicular.