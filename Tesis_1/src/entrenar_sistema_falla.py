from pathlib import Path
import sys
import pandas as pd


# ============================================================
# RUTA RAÍZ DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================



DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "datos_mantenimiento_quito.csv"
)


# ============================================================
# CARGAR DATASET
# ============================================================

print("=" * 70)
print("PREDICCIÓN DE SISTEMA DE FALLA FUTURA")
print("=" * 70)

print(f"\nRuta del dataset:\n{DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\nDataset cargado correctamente.")
print(f"Registros totales: {len(df)}")
print(f"Columnas totales: {len(df.columns)}")


# ============================================================
# VERIFICAR VARIABLES NECESARIAS
# ============================================================

columnas_necesarias = [
    "vehiculo_id",
    "falla_futura_30d",
    "sistema_falla_futura",
    "repuesto_falla_futura",
]

print("\n" + "=" * 70)
print("VERIFICACIÓN DE COLUMNAS")
print("=" * 70)

for columna in columnas_necesarias:
    if columna in df.columns:
        print(f"[OK] {columna}")
    else:
        print(f"[FALTA] {columna}")


# ============================================================
# REVISAR RESULTADOS FUTUROS CONOCIDOS
# ============================================================

print("\n" + "=" * 70)
print("RESULTADOS FUTUROS")
print("=" * 70)

print("\nDistribución de falla_futura_30d:")
print(
    df["falla_futura_30d"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# CASOS CON FALLA FUTURA
# ============================================================

df_fallas = df[
    df["falla_futura_30d"] == 1
].copy()

print(f"\nRegistros con falla futura: {len(df_fallas)}")


# ============================================================
# DISTRIBUCIÓN DEL SISTEMA DE FALLA
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUCIÓN DE SISTEMA DE FALLA FUTURA")
print("=" * 70)

print(
    df_fallas["sistema_falla_futura"]
    .value_counts(dropna=False)
)


# ============================================================
# VERIFICAR VALORES NULOS
# ============================================================

nulos_sistema = df_fallas[
    "sistema_falla_futura"
].isna().sum()

print(
    f"\nValores nulos en sistema_falla_futura "
    f"dentro de los casos con falla: {nulos_sistema}"
)

print("\nVerificación finalizada correctamente.")
print("Todavía NO se ha entrenado ningún modelo.")


# ============================================================
# DEFINIR VARIABLES PARA PREDICCIÓN DEL SISTEMA
# ============================================================

print("\n" + "=" * 70)
print("VARIABLES PARA EL MODELO DE SISTEMA")
print("=" * 70)

FEATURES_SISTEMA = [
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
    "costo_mantenimiento_actual_usd",
]

TARGET_SISTEMA = "sistema_falla_futura"


# ============================================================
# VERIFICAR QUE LAS VARIABLES EXISTAN
# ============================================================

faltantes = [
    columna
    for columna in FEATURES_SISTEMA + [TARGET_SISTEMA]
    if columna not in df_fallas.columns
]

if faltantes:
    print("\n[ERROR] Faltan columnas:")
    for columna in faltantes:
        print(f" - {columna}")
else:
    print("\n[OK] Todas las variables necesarias existen.")


# ============================================================
# CREAR X E y
# ============================================================

X = df_fallas[FEATURES_SISTEMA].copy()
y = df_fallas[TARGET_SISTEMA].copy()

print(f"\nRegistros disponibles para modelado: {len(X)}")
print(f"Variables predictoras: {X.shape[1]}")
print(f"Clases objetivo: {y.nunique()}")


# ============================================================
# MOSTRAR VARIABLES DE ENTRADA
# ============================================================

print("\nVariables utilizadas como entrada:")

for numero, columna in enumerate(FEATURES_SISTEMA, start=1):
    print(f"{numero:02d}. {columna}")


# ============================================================
# COMPROBAR QUE NO EXISTA FUGA DE INFORMACIÓN
# ============================================================

columnas_futuras = [
    columna
    for columna in FEATURES_SISTEMA
    if "futura" in columna.lower()
]

if columnas_futuras:
    print("\n[ADVERTENCIA] Se encontraron variables futuras:")
    for columna in columnas_futuras:
        print(f" - {columna}")
else:
    print(
        "\n[OK] No se están utilizando variables futuras "
        "como predictores."
    )


print("\n" + "=" * 70)
print("PASO 3 FINALIZADO")
print("=" * 70)

print("X:", X.shape)
print("y:", y.shape)

print("\nTodavía NO se ha entrenado ningún modelo.")



# ============================================================
# PASO 4: PREPARACIÓN DE TRAIN Y TEST
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


print("\n" + "=" * 70)
print("PASO 4 - PREPARACIÓN DE TRAIN Y TEST")
print("=" * 70)


# ============================================================
# IDENTIFICAR VARIABLES NUMÉRICAS Y CATEGÓRICAS
# ============================================================

columnas_numericas = X.select_dtypes(
    include=["number"]
).columns.tolist()

columnas_categoricas = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


print("\nVariables numéricas:")
for columna in columnas_numericas:
    print(f" - {columna}")

print("\nVariables categóricas:")
for columna in columnas_categoricas:
    print(f" - {columna}")


# ============================================================
# DIVISIÓN TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("DIVISIÓN DE DATOS")
print("=" * 70)

print(f"\nRegistros totales: {len(X)}")
print(f"Train: {len(X_train)}")
print(f"Test: {len(X_test)}")

print(
    f"\nPorcentaje Train: "
    f"{len(X_train) / len(X) * 100:.1f}%"
)

print(
    f"Porcentaje Test: "
    f"{len(X_test) / len(X) * 100:.1f}%"
)


# ============================================================
# COMPROBAR DISTRIBUCIÓN DE CLASES
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUCIÓN DE CLASES EN TRAIN")
print("=" * 70)

print(
    y_train
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 70)
print("DISTRIBUCIÓN DE CLASES EN TEST")
print("=" * 70)

print(
    y_test
    .value_counts()
    .sort_index()
)


# ============================================================
# PREPROCESAMIENTO DE VARIABLES NUMÉRICAS
# ============================================================

pipeline_numerico = Pipeline(
    steps=[
        (
            "imputador",
            SimpleImputer(strategy="median")
        )
    ]
)


# ============================================================
# PREPROCESAMIENTO DE VARIABLES CATEGÓRICAS
# ============================================================

pipeline_categorico = Pipeline(
    steps=[
        (
            "imputador",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# COMBINAR PREPROCESAMIENTO
# ============================================================

preprocesador = ColumnTransformer(
    transformers=[
        (
            "numericas",
            pipeline_numerico,
            columnas_numericas
        ),
        (
            "categoricas",
            pipeline_categorico,
            columnas_categoricas
        )
    ]
)


print("\n" + "=" * 70)
print("PREPROCESAMIENTO")
print("=" * 70)

print(
    f"\nCantidad de variables numéricas: "
    f"{len(columnas_numericas)}"
)

print(
    f"Cantidad de variables categóricas: "
    f"{len(columnas_categoricas)}"
)

print(
    "\n[OK] Preprocesador creado correctamente."
)

print(
    "[OK] OneHotEncoder configurado para categorías nuevas."
)

print(
    "[OK] Valores numéricos faltantes se imputarán con la mediana."
)

print(
    "[OK] Valores categóricos faltantes se imputarán "
    "con la categoría más frecuente."
)


print("\n" + "=" * 70)
print("PASO 4 FINALIZADO")
print("=" * 70)

print("\nTodavía NO se ha entrenado ningún modelo.")


# ============================================================
# PASO 5: ENTRENAMIENTO Y COMPARACIÓN DE MODELOS
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


print("\n" + "=" * 70)
print("PASO 5 - ENTRENAMIENTO Y COMPARACIÓN DE MODELOS")
print("=" * 70)


# ============================================================
# MODELOS A EVALUAR
# ============================================================

modelos = {
    "Regresión Logística": LogisticRegression(
        max_iter=3000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
}


# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

resultados = []

for nombre, clasificador in modelos.items():

    print("\n" + "-" * 70)
    print(f"Entrenando: {nombre}")
    print("-" * 70)

    pipeline_modelo = Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            ("clasificador", clasificador)
        ]
    )

    pipeline_modelo.fit(
        X_train,
        y_train
    )

    predicciones = pipeline_modelo.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predicciones
    )

    precision_macro = precision_score(
        y_test,
        predicciones,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test,
        predicciones,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        predicciones,
        average="macro",
        zero_division=0
    )

    resultados.append({
        "Modelo": nombre,
        "Accuracy": accuracy,
        "Precision_macro": precision_macro,
        "Recall_macro": recall_macro,
        "F1_macro": f1_macro
    })

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Precision macro: {precision_macro:.4f}")
    print(f"Recall macro:    {recall_macro:.4f}")
    print(f"F1 macro:        {f1_macro:.4f}")


# ============================================================
# TABLA COMPARATIVA
# ============================================================

resultados_df = pd.DataFrame(
    resultados
).sort_values(
    by="F1_macro",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("COMPARACIÓN FINAL")
print("=" * 70)

print(
    resultados_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# IDENTIFICAR MEJOR RESULTADO DEL EXPERIMENTO
# ============================================================

mejor_nombre = resultados_df.iloc[0]["Modelo"]
mejor_f1 = resultados_df.iloc[0]["F1_macro"]

print("\n" + "=" * 70)
print("MEJOR RESULTADO DE ESTA PRUEBA")
print("=" * 70)

print(f"Modelo: {mejor_nombre}")
print(f"F1 macro: {mejor_f1:.4f}")


print("\n" + "=" * 70)
print("PASO 5 FINALIZADO")
print("=" * 70)

print(
    "\nLos modelos fueron evaluados, pero todavía "
    "NO se ha guardado ningún modelo."
)


# ============================================================
# PASO 6: DIAGNÓSTICO DE LA CAPACIDAD PREDICTIVA
# ============================================================

print("\n" + "=" * 70)
print("PASO 6 - DIAGNÓSTICO DE LOS DATOS")
print("=" * 70)


# ============================================================
# 1. SISTEMA ACTUAL VS SISTEMA DE FALLA FUTURA
# ============================================================

print("\n" + "=" * 70)
print("1. SISTEMA INTERVENIDO ACTUAL VS SISTEMA FUTURO")
print("=" * 70)

tabla_sistema = pd.crosstab(
    df_fallas["sistema_intervenido_actual"],
    df_fallas["sistema_falla_futura"],
    normalize="index"
) * 100

print("\nPorcentajes por sistema intervenido actual:")
print(
    tabla_sistema.round(1).to_string()
)


# ============================================================
# 2. REPUESTO ACTUAL VS SISTEMA DE FALLA FUTURA
# ============================================================

print("\n" + "=" * 70)
print("2. REPUESTO CAMBIADO ACTUAL VS SISTEMA FUTURO")
print("=" * 70)

tabla_repuesto = pd.crosstab(
    df_fallas["repuesto_cambiado_actual"],
    df_fallas["sistema_falla_futura"],
    normalize="index"
) * 100

print("\nPorcentajes por repuesto cambiado actualmente:")
print(
    tabla_repuesto.round(1).to_string()
)


# ============================================================
# 3. PROMEDIOS NUMÉRICOS POR SISTEMA FUTURO
# ============================================================

print("\n" + "=" * 70)
print("3. PROMEDIOS DE VARIABLES NUMÉRICAS POR SISTEMA FUTURO")
print("=" * 70)

variables_diagnostico = [
    "anio",
    "kilometraje_acumulado_km",
    "km_desde_ultimo_mantenimiento",
    "dias_desde_ultimo_mantenimiento",
    "temperatura_promedio_motor_c",
    "presion_aceite_kpa",
    "frenadas_bruscas_por_100km",
    "tiempo_ralenti_min_dia",
    "altitud_operacion_m",
    "costo_mantenimiento_actual_usd",
]

promedios_sistema = (
    df_fallas
    .groupby("sistema_falla_futura")[variables_diagnostico]
    .mean()
    .round(2)
)

print(
    "\nPromedios por sistema de falla futura:"
)

print(
    promedios_sistema.to_string()
)


# ============================================================
# 4. DIFERENCIA RELATIVA ENTRE SISTEMAS
# ============================================================

print("\n" + "=" * 70)
print("4. VARIACIÓN DE VARIABLES ENTRE SISTEMAS")
print("=" * 70)

diagnostico_variacion = []

for columna in variables_diagnostico:

    medias = (
        df_fallas
        .groupby("sistema_falla_futura")[columna]
        .mean()
    )

    media_general = df_fallas[columna].mean()

    if pd.notna(media_general) and media_general != 0:

        diferencia_relativa = (
            (medias.max() - medias.min())
            / abs(media_general)
            * 100
        )

    else:
        diferencia_relativa = 0

    diagnostico_variacion.append({
        "Variable": columna,
        "Media_minima": medias.min(),
        "Media_maxima": medias.max(),
        "Diferencia_relativa_pct": diferencia_relativa
    })


variacion_df = pd.DataFrame(
    diagnostico_variacion
).sort_values(
    "Diferencia_relativa_pct",
    ascending=False
)

print(
    variacion_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 5. CREAR FEATURES ADICIONALES DE DIAGNÓSTICO
# ============================================================

print("\n" + "=" * 70)
print("5. FEATURES ADICIONALES DE FEATURES.PY")
print("=" * 70)

try:

    from src.features import construir_features_t0

    df_features_diagnostico = construir_features_t0(
        df_fallas.copy()
    )

    features_nuevas = [
        "edad_vehiculo_años",
        "km_promedio_estimado_diario",
        "indice_estres_termico_altitud",
        "ratio_recorrido_reciente",
    ]

    features_disponibles = [
        columna
        for columna in features_nuevas
        if columna in df_features_diagnostico.columns
    ]

    print("\nFeatures creadas correctamente:")

    for columna in features_disponibles:
        print(f" [OK] {columna}")

    if features_disponibles:

        print(
            "\nPromedios de las nuevas features "
            "por sistema futuro:"
        )

        resumen_features = (
            df_features_diagnostico
            .groupby("sistema_falla_futura")[
                features_disponibles
            ]
            .mean()
            .round(4)
        )

        print(
            resumen_features.to_string()
        )

except Exception as error:

    print(
        "\n[ADVERTENCIA] No fue posible generar "
        "las features adicionales."
    )

    print(
        f"Detalle técnico: {error}"
    )


# ============================================================
# FIN DEL DIAGNÓSTICO
# ============================================================

print("\n" + "=" * 70)
print("PASO 6 FINALIZADO")
print("=" * 70)

print(
    "\nEste paso únicamente analiza relaciones en los datos."
)

print(
    "NO se ha guardado ni modificado ningún modelo."
)


# ============================================================
# PASO 8: SEGUNDO EXPERIMENTO
# FEATURES T0 + ESCALADO DE VARIABLES NUMÉRICAS
# ============================================================

from sklearn.preprocessing import StandardScaler


print("\n" + "=" * 70)
print("PASO 8 - SEGUNDO EXPERIMENTO")
print("FEATURES T0 + ESCALADO")
print("=" * 70)


# ============================================================
# 1. GENERAR FEATURES T0
# ============================================================

from src.features import construir_features_t0

df_experimento2 = construir_features_t0(
    df_fallas.copy()
)


# ============================================================
# 2. DEFINIR VARIABLES DEL SEGUNDO EXPERIMENTO
# ============================================================

FEATURES_T0 = [
    "edad_vehiculo_años",
    "km_promedio_estimado_diario",
    "indice_estres_termico_altitud",
    "ratio_recorrido_reciente",
]

FEATURES_SISTEMA_V2 = (
    FEATURES_SISTEMA
    + FEATURES_T0
)


print("\nVariables originales:")
print(len(FEATURES_SISTEMA))

print("Features T0 adicionales:")
print(len(FEATURES_T0))

print("Total de variables del experimento 2:")
print(len(FEATURES_SISTEMA_V2))


# ============================================================
# 3. CREAR X E y
# ============================================================

X_v2 = df_experimento2[
    FEATURES_SISTEMA_V2
].copy()

y_v2 = df_experimento2[
    TARGET_SISTEMA
].copy()


print(f"\nX_v2: {X_v2.shape}")
print(f"y_v2: {y_v2.shape}")


# ============================================================
# 4. IDENTIFICAR NUMÉRICAS Y CATEGÓRICAS
# ============================================================

columnas_numericas_v2 = X_v2.select_dtypes(
    include=["number"]
).columns.tolist()

columnas_categoricas_v2 = X_v2.select_dtypes(
    exclude=["number"]
).columns.tolist()


print(
    f"\nVariables numéricas: "
    f"{len(columnas_numericas_v2)}"
)

print(
    f"Variables categóricas: "
    f"{len(columnas_categoricas_v2)}"
)


# ============================================================
# 5. TRAIN / TEST
# MISMA SEMILLA Y MISMO PORCENTAJE DEL EXPERIMENTO ANTERIOR
# ============================================================

X_train_v2, X_test_v2, y_train_v2, y_test_v2 = (
    train_test_split(
        X_v2,
        y_v2,
        test_size=0.20,
        random_state=42,
        stratify=y_v2
    )
)


print("\nDivisión Train/Test:")

print(
    f"Train: {len(X_train_v2)}"
)

print(
    f"Test: {len(X_test_v2)}"
)


# ============================================================
# 6. PIPELINE NUMÉRICO CON ESCALADO
# ============================================================

pipeline_numerico_v2 = Pipeline(
    steps=[
        (
            "imputador",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "escalador",
            StandardScaler()
        )
    ]
)


# ============================================================
# 7. PIPELINE CATEGÓRICO
# ============================================================

pipeline_categorico_v2 = Pipeline(
    steps=[
        (
            "imputador",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# 8. PREPROCESADOR V2
# ============================================================

preprocesador_v2 = ColumnTransformer(
    transformers=[
        (
            "numericas",
            pipeline_numerico_v2,
            columnas_numericas_v2
        ),
        (
            "categoricas",
            pipeline_categorico_v2,
            columnas_categoricas_v2
        )
    ]
)


print(
    "\n[OK] Preprocesamiento V2 creado."
)

print(
    "[OK] Variables numéricas con StandardScaler."
)

print(
    "[OK] Variables categóricas con OneHotEncoder."
)


# ============================================================
# 9. MODELOS DEL SEGUNDO EXPERIMENTO
# ============================================================

modelos_v2 = {

    "Regresión Logística V2": LogisticRegression(
        max_iter=5000,
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest V2": RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Extra Trees V2": ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
}


# ============================================================
# 10. ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

resultados_v2 = []

for nombre, clasificador in modelos_v2.items():

    print("\n" + "-" * 70)
    print(f"Entrenando: {nombre}")
    print("-" * 70)

    pipeline_v2 = Pipeline(
        steps=[
            (
                "preprocesamiento",
                preprocesador_v2
            ),
            (
                "clasificador",
                clasificador
            )
        ]
    )

    pipeline_v2.fit(
        X_train_v2,
        y_train_v2
    )

    predicciones_v2 = pipeline_v2.predict(
        X_test_v2
    )

    accuracy_v2 = accuracy_score(
        y_test_v2,
        predicciones_v2
    )

    precision_v2 = precision_score(
        y_test_v2,
        predicciones_v2,
        average="macro",
        zero_division=0
    )

    recall_v2 = recall_score(
        y_test_v2,
        predicciones_v2,
        average="macro",
        zero_division=0
    )

    f1_v2 = f1_score(
        y_test_v2,
        predicciones_v2,
        average="macro",
        zero_division=0
    )

    resultados_v2.append({
        "Modelo": nombre,
        "Accuracy": accuracy_v2,
        "Precision_macro": precision_v2,
        "Recall_macro": recall_v2,
        "F1_macro": f1_v2
    })

    print(
        f"Accuracy:        {accuracy_v2:.4f}"
    )

    print(
        f"Precision macro: {precision_v2:.4f}"
    )

    print(
        f"Recall macro:    {recall_v2:.4f}"
    )

    print(
        f"F1 macro:        {f1_v2:.4f}"
    )


# ============================================================
# 11. TABLA COMPARATIVA DEL EXPERIMENTO 2
# ============================================================

resultados_v2_df = pd.DataFrame(
    resultados_v2
).sort_values(
    by="F1_macro",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("RESULTADOS DEL SEGUNDO EXPERIMENTO")
print("=" * 70)

print(
    resultados_v2_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 12. COMPARACIÓN CONTRA EL EXPERIMENTO ORIGINAL
# ============================================================

print("\n" + "=" * 70)
print("COMPARACIÓN EXPERIMENTO 1 VS EXPERIMENTO 2")
print("=" * 70)

mejor_f1_v1 = float(
    resultados_df.iloc[0]["F1_macro"]
)

mejor_modelo_v1 = (
    resultados_df.iloc[0]["Modelo"]
)

mejor_f1_v2 = float(
    resultados_v2_df.iloc[0]["F1_macro"]
)

mejor_modelo_v2 = (
    resultados_v2_df.iloc[0]["Modelo"]
)


print("\nExperimento 1:")
print(
    f"Modelo: {mejor_modelo_v1}"
)
print(
    f"F1 macro: {mejor_f1_v1:.4f}"
)


print("\nExperimento 2:")
print(
    f"Modelo: {mejor_modelo_v2}"
)
print(
    f"F1 macro: {mejor_f1_v2:.4f}"
)


diferencia_f1 = (
    mejor_f1_v2
    - mejor_f1_v1
)


print(
    f"\nDiferencia F1 macro: "
    f"{diferencia_f1:+.4f}"
)


# ============================================================
# 13. REFERENCIA ALEATORIA APROXIMADA
# ============================================================

numero_clases = y_v2.nunique()

referencia_aleatoria = (
    1 / numero_clases
)


print("\n" + "=" * 70)
print("REFERENCIA BASE")
print("=" * 70)

print(
    f"Número de clases: {numero_clases}"
)

print(
    f"Referencia aleatoria aproximada: "
    f"{referencia_aleatoria:.4f} "
    f"({referencia_aleatoria * 100:.1f}%)"
)


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 70)
print("PASO 8 FINALIZADO")
print("=" * 70)

print(
    "\nSegundo experimento finalizado."
)

print(
    "Todavía NO se ha guardado ningún modelo."
)


# ============================================================
# PASO 9: VALIDACIÓN CRUZADA ESTRATIFICADA
# ============================================================

from sklearn.model_selection import StratifiedKFold, cross_validate


print("\n" + "=" * 70)
print("PASO 9 - VALIDACIÓN CRUZADA ESTRATIFICADA")
print("=" * 70)


# ============================================================
# 1. CONFIGURAR VALIDACIÓN CRUZADA
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 2. MODELOS A EVALUAR
# ============================================================

modelos_cv = {

    "Regresión Logística V2": LogisticRegression(
        max_iter=5000,
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest V2": RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Extra Trees V2": ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
}


# ============================================================
# 3. MÉTRICAS
# ============================================================

metricas_cv = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}


# ============================================================
# 4. EVALUAR CADA MODELO
# ============================================================

resultados_cv = []

for nombre, clasificador in modelos_cv.items():

    print("\n" + "-" * 70)
    print(f"Evaluando: {nombre}")
    print("-" * 70)

    pipeline_cv = Pipeline(
        steps=[
            (
                "preprocesamiento",
                preprocesador_v2
            ),
            (
                "clasificador",
                clasificador
            )
        ]
    )

    scores = cross_validate(
        pipeline_cv,
        X_v2,
        y_v2,
        cv=cv,
        scoring=metricas_cv,
        n_jobs=-1,
        return_train_score=False
    )

    accuracy_media = scores[
        "test_accuracy"
    ].mean()

    accuracy_std = scores[
        "test_accuracy"
    ].std()

    precision_media = scores[
        "test_precision_macro"
    ].mean()

    recall_media = scores[
        "test_recall_macro"
    ].mean()

    f1_media = scores[
        "test_f1_macro"
    ].mean()

    f1_std = scores[
        "test_f1_macro"
    ].std()

    resultados_cv.append({
        "Modelo": nombre,
        "Accuracy_media": accuracy_media,
        "Accuracy_std": accuracy_std,
        "Precision_macro_media": precision_media,
        "Recall_macro_media": recall_media,
        "F1_macro_media": f1_media,
        "F1_macro_std": f1_std
    })

    print("\nAccuracy por fold:")

    for numero, valor in enumerate(
        scores["test_accuracy"],
        start=1
    ):
        print(
            f" Fold {numero}: {valor:.4f}"
        )

    print("\nF1 macro por fold:")

    for numero, valor in enumerate(
        scores["test_f1_macro"],
        start=1
    ):
        print(
            f" Fold {numero}: {valor:.4f}"
        )

    print(
        f"\nAccuracy promedio: "
        f"{accuracy_media:.4f} "
        f"± {accuracy_std:.4f}"
    )

    print(
        f"F1 macro promedio: "
        f"{f1_media:.4f} "
        f"± {f1_std:.4f}"
    )


# ============================================================
# 5. TABLA FINAL
# ============================================================

resultados_cv_df = pd.DataFrame(
    resultados_cv
).sort_values(
    by="F1_macro_media",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("RESULTADOS VALIDACIÓN CRUZADA")
print("=" * 70)

print(
    resultados_cv_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 6. MEJOR RESULTADO OBSERVADO
# ============================================================

mejor_cv = resultados_cv_df.iloc[0]

print("\n" + "=" * 70)
print("MEJOR RESULTADO DE VALIDACIÓN CRUZADA")
print("=" * 70)

print(
    f"Modelo: "
    f"{mejor_cv['Modelo']}"
)

print(
    f"Accuracy promedio: "
    f"{mejor_cv['Accuracy_media']:.4f}"
)

print(
    f"F1 macro promedio: "
    f"{mejor_cv['F1_macro_media']:.4f}"
)

print(
    f"Desviación F1: "
    f"{mejor_cv['F1_macro_std']:.4f}"
)


# ============================================================
# 7. REFERENCIA BASE
# ============================================================

print("\nReferencia aproximada para 5 clases equilibradas:")
print("Accuracy aleatoria ≈ 0.2000 (20.0%)")


print("\n" + "=" * 70)
print("PASO 9 FINALIZADO")
print("=" * 70)

print(
    "\nLa validación cruzada ha finalizado."
)

print(
    "Todavía NO se ha guardado ningún modelo."
)


# ============================================================
# PASO 10: EXPERIMENTO CON 3 MACROGRUPOS
# ============================================================

print("\n" + "=" * 70)
print("PASO 10 - PREDICCIÓN CON 3 MACROGRUPOS")
print("=" * 70)


# ============================================================
# 1. DEFINIR AGRUPACIÓN
# ============================================================

mapa_macrogrupos = {
    "Sistema de Enfriamiento":
        "Motor y térmico",

    "Sistema de Lubricación y Motor":
        "Motor y térmico",

    "Sistema de Frenos":
        "Seguridad y rodaje",

    "Suspensión, Dirección y Neumáticos":
        "Seguridad y rodaje",

    "Sistema Eléctrico y Batería":
        "Eléctrico"
}


# ============================================================
# 2. CREAR NUEVA VARIABLE OBJETIVO
# ============================================================

y_macro = df_experimento2[
    "sistema_falla_futura"
].map(
    mapa_macrogrupos
)


print("\nDistribución de los 3 macrogrupos:")
print(
    y_macro.value_counts()
)


# Comprobar que ningún sistema haya quedado sin clasificar

nulos_macro = y_macro.isna().sum()

print(
    f"\nRegistros sin macrogrupo: {nulos_macro}"
)

if nulos_macro > 0:
    raise ValueError(
        "Existen sistemas que no fueron asignados "
        "a ningún macrogrupo."
    )


# ============================================================
# 3. VARIABLES PREDICTORAS
# ============================================================

X_macro = df_experimento2[
    FEATURES_SISTEMA_V2
].copy()


print(
    f"\nX_macro: {X_macro.shape}"
)

print(
    f"y_macro: {y_macro.shape}"
)

print(
    f"Número de clases: {y_macro.nunique()}"
)


# ============================================================
# 4. VALIDACIÓN CRUZADA
# ============================================================

cv_macro = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 5. MODELOS
# ============================================================

modelos_macro = {

    "Regresión Logística": LogisticRegression(
        max_iter=5000,
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
}


# ============================================================
# 6. EVALUACIÓN
# ============================================================

resultados_macro = []

for nombre, clasificador in modelos_macro.items():

    print("\n" + "-" * 70)
    print(f"Evaluando: {nombre}")
    print("-" * 70)

    pipeline_macro = Pipeline(
        steps=[
            (
                "preprocesamiento",
                preprocesador_v2
            ),
            (
                "clasificador",
                clasificador
            )
        ]
    )

    scores_macro = cross_validate(
        pipeline_macro,
        X_macro,
        y_macro,
        cv=cv_macro,
        scoring=metricas_cv,
        n_jobs=-1,
        return_train_score=False
    )

    accuracy_media = scores_macro[
        "test_accuracy"
    ].mean()

    accuracy_std = scores_macro[
        "test_accuracy"
    ].std()

    f1_media = scores_macro[
        "test_f1_macro"
    ].mean()

    f1_std = scores_macro[
        "test_f1_macro"
    ].std()

    precision_media = scores_macro[
        "test_precision_macro"
    ].mean()

    recall_media = scores_macro[
        "test_recall_macro"
    ].mean()

    resultados_macro.append({
        "Modelo": nombre,
        "Accuracy_media": accuracy_media,
        "Accuracy_std": accuracy_std,
        "Precision_macro_media": precision_media,
        "Recall_macro_media": recall_media,
        "F1_macro_media": f1_media,
        "F1_macro_std": f1_std
    })

    print("\nAccuracy por fold:")

    for numero, valor in enumerate(
        scores_macro["test_accuracy"],
        start=1
    ):
        print(
            f" Fold {numero}: {valor:.4f}"
        )

    print("\nF1 macro por fold:")

    for numero, valor in enumerate(
        scores_macro["test_f1_macro"],
        start=1
    ):
        print(
            f" Fold {numero}: {valor:.4f}"
        )

    print(
        f"\nAccuracy promedio: "
        f"{accuracy_media:.4f} ± {accuracy_std:.4f}"
    )

    print(
        f"F1 macro promedio: "
        f"{f1_media:.4f} ± {f1_std:.4f}"
    )


# ============================================================
# 7. RESULTADOS FINALES
# ============================================================

resultados_macro_df = pd.DataFrame(
    resultados_macro
).sort_values(
    by="F1_macro_media",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("RESULTADOS - 3 MACROGRUPOS")
print("=" * 70)

print(
    resultados_macro_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 8. MEJOR RESULTADO OBSERVADO
# ============================================================

mejor_macro = resultados_macro_df.iloc[0]

print("\n" + "=" * 70)
print("MEJOR RESULTADO - 3 MACROGRUPOS")
print("=" * 70)

print(
    f"Modelo: {mejor_macro['Modelo']}"
)

print(
    f"Accuracy promedio: "
    f"{mejor_macro['Accuracy_media']:.4f}"
)

print(
    f"F1 macro promedio: "
    f"{mejor_macro['F1_macro_media']:.4f}"
)

print(
    f"Desviación F1: "
    f"{mejor_macro['F1_macro_std']:.4f}"
)


# ============================================================
# 9. BASELINES
# ============================================================

clase_mayoritaria = (
    y_macro.value_counts(normalize=True).max()
)

referencia_uniforme = (
    1 / y_macro.nunique()
)


print("\n" + "=" * 70)
print("REFERENCIAS BASE")
print("=" * 70)

print(
    f"Referencia uniforme para 3 clases: "
    f"{referencia_uniforme:.4f} "
    f"({referencia_uniforme * 100:.1f}%)"
)

print(
    f"Accuracy de predecir siempre la clase mayoritaria: "
    f"{clase_mayoritaria:.4f} "
    f"({clase_mayoritaria * 100:.1f}%)"
)


print("\n" + "=" * 70)
print("PASO 10 FINALIZADO")
print("=" * 70)

print(
    "\nNo se ha guardado ningún modelo."
)


# ============================================================
# PASO 11: EVALUACIÓN DETALLADA DEL RANDOM FOREST
# CON 3 MACROGRUPOS
# ============================================================

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)


print("\n" + "=" * 70)
print("PASO 11 - EVALUACIÓN DETALLADA DE LOS 3 MACROGRUPOS")
print("=" * 70)


# ============================================================
# 1. RANDOM FOREST
# ============================================================

rf_macro = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


pipeline_rf_macro = Pipeline(
    steps=[
        (
            "preprocesamiento",
            preprocesador_v2
        ),
        (
            "clasificador",
            rf_macro
        )
    ]
)


# ============================================================
# 2. PREDICCIONES OUT-OF-FOLD
# ============================================================

print(
    "\nGenerando predicciones mediante "
    "validación cruzada estratificada..."
)

predicciones_macro_cv = cross_val_predict(
    pipeline_rf_macro,
    X_macro,
    y_macro,
    cv=cv_macro,
    n_jobs=-1,
    method="predict"
)

print("[OK] Predicciones generadas.")


# ============================================================
# 3. ORDEN DE LAS CLASES
# ============================================================

clases_macro = [
    "Motor y térmico",
    "Seguridad y rodaje",
    "Eléctrico"
]


# ============================================================
# 4. MATRIZ DE CONFUSIÓN
# ============================================================

matriz_macro = confusion_matrix(
    y_macro,
    predicciones_macro_cv,
    labels=clases_macro
)


print("\n" + "=" * 70)
print("MATRIZ DE CONFUSIÓN")
print("=" * 70)

matriz_macro_df = pd.DataFrame(
    matriz_macro,
    index=[
        f"Real: {clase}"
        for clase in clases_macro
    ],
    columns=[
        f"Predicho: {clase}"
        for clase in clases_macro
    ]
)

print(
    "\n" + matriz_macro_df.to_string()
)


# ============================================================
# 5. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("REPORTE POR MACROGRUPO")
print("=" * 70)

reporte_macro = classification_report(
    y_macro,
    predicciones_macro_cv,
    labels=clases_macro,
    digits=4,
    zero_division=0
)

print("\n" + reporte_macro)


# ============================================================
# 6. BALANCED ACCURACY
# ============================================================

balanced_acc = balanced_accuracy_score(
    y_macro,
    predicciones_macro_cv
)

print("\n" + "=" * 70)
print("BALANCED ACCURACY")
print("=" * 70)

print(
    f"\nBalanced Accuracy: {balanced_acc:.4f}"
)


# ============================================================
# 7. CANTIDAD DE PREDICCIONES POR CLASE
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUCIÓN DE PREDICCIONES")
print("=" * 70)

distribucion_predicciones = (
    pd.Series(predicciones_macro_cv)
    .value_counts()
    .reindex(
        clases_macro,
        fill_value=0
    )
)

print(
    "\nPredicciones realizadas por el modelo:"
)

print(
    distribucion_predicciones.to_string()
)


# ============================================================
# 8. COMPARAR REAL VS PREDICHO
# ============================================================

print("\nDistribución real:")

distribucion_real = (
    y_macro
    .value_counts()
    .reindex(
        clases_macro,
        fill_value=0
    )
)

print(
    distribucion_real.to_string()
)


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 70)
print("PASO 11 FINALIZADO")
print("=" * 70)

print(
    "\nEvaluación detallada completada."
)

print(
    "NO se ha guardado ningún modelo."
)



# ============================================================
# PASO 12: EXPLORACIÓN BINARIA DE LOS MACROGRUPOS
# ============================================================

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score
)


print("\n" + "=" * 70)
print("PASO 12 - EXPLORACIÓN BINARIA DE LOS MACROGRUPOS")
print("=" * 70)


# ============================================================
# 1. DEFINIR LOS TRES PROBLEMAS BINARIOS
# ============================================================

macrogrupos_binarios = [
    "Motor y térmico",
    "Seguridad y rodaje",
    "Eléctrico"
]


# ============================================================
# 2. VALIDACIÓN CRUZADA
# ============================================================

cv_binario = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 3. RESULTADOS
# ============================================================

resultados_binarios = []


# ============================================================
# 4. EVALUAR CADA MACROGRUPO
# ============================================================

for macrogrupo in macrogrupos_binarios:

    print("\n" + "=" * 70)
    print(f"OBJETIVO BINARIO: {macrogrupo}")
    print("=" * 70)

    # --------------------------------------------------------
    # Crear objetivo:
    # 1 = pertenece al macrogrupo
    # 0 = pertenece a otro macrogrupo
    # --------------------------------------------------------

    y_binario = (
        y_macro == macrogrupo
    ).astype(int)


    positivos = int(y_binario.sum())
    negativos = int(
        len(y_binario) - positivos
    )

    porcentaje_positivos = (
        positivos / len(y_binario)
    )

    print(
        f"\nPositivos ({macrogrupo}): "
        f"{positivos}"
    )

    print(
        f"Negativos (otros): "
        f"{negativos}"
    )

    print(
        f"Porcentaje positivo: "
        f"{porcentaje_positivos * 100:.1f}%"
    )


    # ========================================================
    # 5. RANDOM FOREST BINARIO
    # ========================================================

    modelo_binario = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )


    pipeline_binario = Pipeline(
        steps=[
            (
                "preprocesamiento",
                preprocesador_v2
            ),
            (
                "clasificador",
                modelo_binario
            )
        ]
    )


    # ========================================================
    # 6. PREDICCIONES OUT-OF-FOLD
    # ========================================================

    pred_binarias = cross_val_predict(
        pipeline_binario,
        X_macro,
        y_binario,
        cv=cv_binario,
        n_jobs=-1,
        method="predict"
    )


    # ========================================================
    # 7. PROBABILIDADES OUT-OF-FOLD
    # ========================================================

    prob_binarias = cross_val_predict(
        pipeline_binario,
        X_macro,
        y_binario,
        cv=cv_binario,
        n_jobs=-1,
        method="predict_proba"
    )[:, 1]


    # ========================================================
    # 8. MÉTRICAS
    # ========================================================

    precision_bin = precision_score(
        y_binario,
        pred_binarias,
        zero_division=0
    )

    recall_bin = recall_score(
        y_binario,
        pred_binarias,
        zero_division=0
    )

    f1_bin = f1_score(
        y_binario,
        pred_binarias,
        zero_division=0
    )

    balanced_bin = balanced_accuracy_score(
        y_binario,
        pred_binarias
    )

    roc_auc_bin = roc_auc_score(
        y_binario,
        prob_binarias
    )


    # ========================================================
    # 9. MATRIZ DE CONFUSIÓN
    # ========================================================

    matriz_bin = confusion_matrix(
        y_binario,
        pred_binarias,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matriz_bin.ravel()


    print("\nMatriz de confusión:")
    print(
        f"TN={tn} | FP={fp} | "
        f"FN={fn} | TP={tp}"
    )


    print("\nMétricas:")

    print(
        f"Precision positiva: "
        f"{precision_bin:.4f}"
    )

    print(
        f"Recall positivo:    "
        f"{recall_bin:.4f}"
    )

    print(
        f"F1 positivo:        "
        f"{f1_bin:.4f}"
    )

    print(
        f"Balanced Accuracy:  "
        f"{balanced_bin:.4f}"
    )

    print(
        f"ROC-AUC:            "
        f"{roc_auc_bin:.4f}"
    )


    # ========================================================
    # 10. GUARDAR RESULTADOS
    # ========================================================

    resultados_binarios.append({
        "Macrogrupo": macrogrupo,
        "Positivos": positivos,
        "Negativos": negativos,
        "Precision_positiva": precision_bin,
        "Recall_positivo": recall_bin,
        "F1_positivo": f1_bin,
        "Balanced_Accuracy": balanced_bin,
        "ROC_AUC": roc_auc_bin,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })


# ============================================================
# 11. TABLA COMPARATIVA
# ============================================================

resultados_binarios_df = pd.DataFrame(
    resultados_binarios
).sort_values(
    by="ROC_AUC",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("RESULTADOS DE LA EXPLORACIÓN BINARIA")
print("=" * 70)

print(
    resultados_binarios_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 12. REFERENCIA
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETACIÓN DE REFERENCIA")
print("=" * 70)

print(
    "\nROC-AUC = 0.50 indica una capacidad "
    "de discriminación equivalente aproximadamente al azar."
)

print(
    "Valores superiores a 0.50 indican mayor "
    "capacidad de discriminación."
)

print(
    "La decisión final NO se realizará únicamente "
    "con ROC-AUC."
)

print(
    "También se revisarán Recall, F1 y "
    "Balanced Accuracy."
)


print("\n" + "=" * 70)
print("PASO 12 FINALIZADO")
print("=" * 70)

print(
    "\nExploración binaria completada."
)

print(
    "NO se ha guardado ningún modelo."
)