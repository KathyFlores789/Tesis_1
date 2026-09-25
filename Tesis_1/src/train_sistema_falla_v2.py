"""Experimento V2: comparar grupos de variables para el sistema de falla futura.

Ejecutar desde la raíz: python src/train_sistema_falla_v2.py
Usa únicamente registros con falla futura y sistema conocido.
Compara modelos en la VALIDACIÓN temporal ya utilizada en V1: resultados
EXPLORATORIOS, no evaluación independiente. No examina TEST.
No modifica CSV, V7, ni modelos existentes. No guarda un modelo nuevo.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'data' / 'raw' / 'datos_mantenimiento_quito.csv'
TARGET = 'sistema_falla_futura'

# Solo información disponible cuando se realiza el mantenimiento actual.
CAT = ['marca', 'modelo', 'tipo_vehiculo', 'combustible', 'transmision',
       'zona_operacion_quito', 'tiene_codigo_obd_activo',
       'tipo_mantenimiento_actual', 'sistema_intervenido_actual',
       'repuesto_cambiado_actual']
NUM = ['anio', 'cilindraje_l', 'kilometraje_acumulado_km',
       'km_desde_ultimo_mantenimiento', 'dias_desde_ultimo_mantenimiento',
       'temperatura_promedio_motor_c', 'presion_aceite_kpa',
       'frenadas_bruscas_por_100km', 'tiempo_ralenti_min_dia',
       'altitud_operacion_m', 'costo_mantenimiento_actual_usd']

# V1 YA incluía sistema_intervenido_actual y repuesto_cambiado_actual.
# V2 prueba si retirar variables débiles o centrarse en mantenimiento ayuda.
GRUPOS = {
    'Todas (equivalente a V1)': CAT + NUM,
    'Asociacion exploratoria': [
        'modelo', 'transmision', 'repuesto_cambiado_actual',
        'tiene_codigo_obd_activo', 'tipo_vehiculo', 'marca',
        'sistema_intervenido_actual', 'presion_aceite_kpa',
        'costo_mantenimiento_actual_usd', 'cilindraje_l',
        'kilometraje_acumulado_km'],
    'Mantenimiento y sensores': [
        'tiene_codigo_obd_activo', 'tipo_mantenimiento_actual',
        'sistema_intervenido_actual', 'repuesto_cambiado_actual',
        'kilometraje_acumulado_km', 'km_desde_ultimo_mantenimiento',
        'dias_desde_ultimo_mantenimiento', 'temperatura_promedio_motor_c',
        'presion_aceite_kpa', 'frenadas_bruscas_por_100km',
        'tiempo_ralenti_min_dia', 'costo_mantenimiento_actual_usd'],
    'Vehiculo y operacion': [
        'marca', 'modelo', 'tipo_vehiculo', 'combustible', 'transmision',
        'zona_operacion_quito', 'anio', 'cilindraje_l',
        'kilometraje_acumulado_km', 'altitud_operacion_m'],
}


def preprocesador(columnas):
    cats = [c for c in columnas if c in CAT]
    nums = [c for c in columnas if c in NUM]
    transformaciones = []
    if cats:
        transformaciones.append(('categoricas', Pipeline([
            ('imputar', SimpleImputer(strategy='most_frequent')),
            ('codificar', OneHotEncoder(handle_unknown='ignore'))]), cats))
    if nums:
        transformaciones.append(('numericas', Pipeline([
            ('imputar', SimpleImputer(strategy='median')),
            ('escalar', StandardScaler())]), nums))
    return ColumnTransformer(transformaciones)


def main():
    if not CSV.is_file():
        raise FileNotFoundError(f'No se encontró el CSV: {CSV}')
    df = pd.read_csv(CSV)
    requeridas = set(CAT + NUM + ['fecha_mantenimiento', 'falla_futura_30d', TARGET])
    faltantes = sorted(requeridas - set(df.columns))
    if faltantes:
        raise ValueError(f'Faltan columnas: {faltantes}')
    df['fecha_mantenimiento'] = pd.to_datetime(df['fecha_mantenimiento'], errors='coerce')
    df['falla_futura_30d'] = pd.to_numeric(df['falla_futura_30d'], errors='coerce')
    df[TARGET] = df[TARGET].astype('string').str.strip()
    df = df.loc[(df['falla_futura_30d'] == 1) & df[TARGET].notna() &
                (df[TARGET] != '') & df['fecha_mantenimiento'].notna()].copy()
    df = df.sort_values('fecha_mantenimiento', kind='stable').reset_index(drop=True)
    for c in CAT:
        df[c] = df[c].astype('string').str.strip().replace({'': pd.NA}).astype(object)
        df[c] = df[c].where(pd.notna(df[c]), np.nan)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    n = len(df)
    if n < 100:
        raise ValueError(f'Solo hay {n} registros válidos; revisar el CSV.')
    # Mismo corte que V1: TRAIN 70%, VALIDACIÓN 15%, TEST 15%.
    a, b = int(n * .70), int(n * .85)
    train, val = df.iloc[:a].copy(), df.iloc[a:b].copy()
    print(f'Registros válidos: {n} | TRAIN: {len(train)} | VAL: {len(val)} | TEST reservado: {n-b}')
    print('ATENCIÓN: VALIDACIÓN ya se utilizó en V1. Resultados V2 exploratorios.')
    print('No se consulta TEST y no se guarda ningún modelo.\n')
    if train[TARGET].nunique() < 2:
        raise ValueError('TRAIN tiene menos de dos clases.')

    resultados = []
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(np.zeros((len(train), 1)), train[TARGET])
    base = dummy.predict(np.zeros((len(val), 1)))
    resultados.append({'grupo': 'Referencia: clase mayoritaria', 'modelo': 'Dummy',
                       'accuracy': accuracy_score(val[TARGET], base),
                       'f1_macro': f1_score(val[TARGET], base, average='macro', zero_division=0)})

    for nombre_grupo, cols in GRUPOS.items():
        print(f'\n=== GRUPO: {nombre_grupo} ({len(cols)} variables) ===')
        candidatos = {
            'LogisticRegression': LogisticRegression(C=.1, class_weight='balanced',
                                                      max_iter=3000, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=400, max_depth=8,
                             min_samples_leaf=4, class_weight='balanced',
                             random_state=42, n_jobs=-1),
        }
        for nombre_modelo, estimador in candidatos.items():
            modelo = Pipeline([('preprocessor', preprocesador(cols)),
                               ('classifier', estimador)])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=FutureWarning)
                modelo.fit(train[cols], train[TARGET])
            pred = modelo.predict(val[cols])
            acc = accuracy_score(val[TARGET], pred)
            f1 = f1_score(val[TARGET], pred, average='macro', zero_division=0)
            resultados.append({'grupo': nombre_grupo, 'modelo': nombre_modelo,
                               'accuracy': acc, 'f1_macro': f1})
            print(f'{nombre_modelo:20s} | Accuracy: {acc:.4f} | F1 macro: {f1:.4f}')

    tabla = pd.DataFrame(resultados).sort_values('f1_macro', ascending=False)
    print('\n=== RESUMEN ORDENADO POR F1 MACRO (VALIDACIÓN REUTILIZADA) ===')
    print(tabla.to_string(index=False, formatters={
        'accuracy': '{:.4f}'.format, 'f1_macro': '{:.4f}'.format}))
    print('\n=== REFERENCIAS ===')
    print(f'Elección aleatoria uniforme entre {train[TARGET].nunique()} clases: '
          f'accuracy esperada aproximada {1/train[TARGET].nunique():.1%}.')
    print('V1: F1 macro validación 0.1871; F1 macro TEST 0.2282 (ya consultado).')
    print('No comparar V2 en esta misma validación como prueba de mejora generalizable.')
    print('No se ha tocado el CSV, la V7, los modelos ni el conjunto TEST.')


if __name__ == '__main__':
    main()
