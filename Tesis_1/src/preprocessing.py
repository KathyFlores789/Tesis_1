import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ==============================================================================
# WHITELIST DE 20 PREDICTORES T0 (Excluye fuga económica y redundancia)
# ==============================================================================
FEATURES_T0_WHITELIST = [
    # Vehículo (7)
    'marca', 'modelo', 'tipo_vehiculo', 'combustible', 
    'transmision', 'cilindraje_l', 'edad_vehiculo_años',
    
    # Uso y operación (10)
    'kilometraje_acumulado_km', 'km_desde_ultimo_mantenimiento',
    'dias_desde_ultimo_mantenimiento', 'frenadas_bruscas_por_100km',
    'tiempo_ralenti_min_dia', 'altitud_operacion_m', 'zona_operacion_quito',
    'km_promedio_estimado_diario', 'ratio_recorrido_reciente', 'indice_estres_termico_altitud',
    
    # Estado / Telemetría (3)
    'temperatura_promedio_motor_c', 'presion_aceite_kpa', 'tiene_codigo_obd_activo'
]

# ==============================================================================
# FUNCIONES DE LIMPIEZA Y FILTRADO
# ==============================================================================
def aplicar_limpieza_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función de compatibilidad utilizada por split_data.py para limpieza inicial.
    """
    df_clean = df.copy()
    # Eliminar duplicados si existen
    df_clean = df_clean.drop_duplicates()
    return df_clean

def filter_t0_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra el DataFrame para retener únicamente las 20 variables T0 válidas.
    """
    cols_presentes = [col for col in FEATURES_T0_WHITELIST if col in df.columns]
    return df[cols_presentes].copy()

# ==============================================================================
# CONSTRUCTOR DEL PREPROCESADOR
# ==============================================================================
def crear_preprocesador_num_cat(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Crea el preprocesador asegurando StandardScaler para variables numéricas 
    y OneHotEncoder para categóricas.
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    return preprocessor