"""Clasificador experimental del sistema de falla futura (solo fallas a 30 días).
Ejecutar desde la raíz: python src/train_sistema_falla_v1.py
No modifica el CSV ni los modelos actuales.
"""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'data' / 'raw' / 'datos_mantenimiento_quito.csv'
OUT = ROOT / 'models'
MODEL_FILE = OUT / 'system_classifier_v1.joblib'
META_FILE = OUT / 'system_classifier_v1_metadata.json'
TARGET = 'sistema_falla_futura'
CATEGORICAS = ['marca', 'modelo', 'tipo_vehiculo', 'combustible',
               'transmision', 'zona_operacion_quito', 'tiene_codigo_obd_activo',
               'tipo_mantenimiento_actual', 'sistema_intervenido_actual',
               'repuesto_cambiado_actual']
NUMERICAS = ['anio', 'cilindraje_l', 'kilometraje_acumulado_km',
             'km_desde_ultimo_mantenimiento', 'dias_desde_ultimo_mantenimiento',
             'temperatura_promedio_motor_c', 'presion_aceite_kpa',
             'frenadas_bruscas_por_100km', 'tiempo_ralenti_min_dia',
             'altitud_operacion_m', 'costo_mantenimiento_actual_usd']
FEATURES = CATEGORICAS + NUMERICAS


def preparar(df):
    requeridas = FEATURES + ['fecha_mantenimiento', 'falla_futura_30d', TARGET]
    faltantes = sorted(set(requeridas) - set(df.columns))
    if faltantes:
        raise ValueError(f'Faltan columnas en el CSV: {faltantes}')
    df = df.copy()
    df['fecha_mantenimiento'] = pd.to_datetime(df['fecha_mantenimiento'], errors='coerce')
    df['falla_futura_30d'] = pd.to_numeric(df['falla_futura_30d'], errors='coerce')
    df = df[(df['falla_futura_30d'] == 1) & df[TARGET].notna() &
            (df[TARGET].astype(str).str.strip() != '') & df['fecha_mantenimiento'].notna()].copy()
    df = df.sort_values('fecha_mantenimiento').reset_index(drop=True)
    for c in CATEGORICAS:
        df[c] = df[c].astype('string').replace({'': pd.NA, 'nan': pd.NA}).astype(object)
        df[c] = df[c].where(pd.notna(df[c]), np.nan)
    for c in NUMERICAS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def crear_preprocesador():
    return ColumnTransformer([
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAS),
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), NUMERICAS),
    ])


def evaluar(nombre, modelo, x, y):
    pred = modelo.predict(x)
    print(f'\n=== {nombre} ===')
    print(f'Accuracy: {accuracy_score(y, pred):.4f} | F1 macro: {f1_score(y, pred, average="macro", zero_division=0):.4f}')
    print(classification_report(y, pred, zero_division=0))
    labels = sorted(set(y) | set(pred))
    print('Matriz de confusión (filas: real, columnas: predicho):')
    print(pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels).to_string())
    return {'accuracy': float(accuracy_score(y, pred)),
            'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0))}


def main():
    if not CSV.exists():
        sys.exit(f'No existe el archivo: {CSV}')
    df = preparar(pd.read_csv(CSV))
    n = len(df)
    if n < 60:
        sys.exit(f'Solo hay {n} registros válidos con falla futura; revisar datos antes de entrenar.')
    a, b = int(n * .70), int(n * .85)
    train, val, test = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    print(f'Registros válidos: {n} | Entrenamiento: {len(train)} | Validación: {len(val)} | Prueba: {len(test)}')
    print('\nDistribución por sistema (total):\n', df[TARGET].value_counts().to_string())
    for nombre, parte in [('TRAIN', train), ('VAL', val), ('TEST', test)]:
        print(f'\n{nombre}:\n{parte[TARGET].value_counts().to_string()}')
    if train[TARGET].nunique() < 2:
        sys.exit('No hay suficientes clases distintas en el conjunto de entrenamiento.')
    candidatos = {
        'LogisticRegression': LogisticRegression(C=.1, class_weight='balanced', max_iter=3000),
        'RandomForest': RandomForestClassifier(n_estimators=400, max_depth=8,
                         min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1),
    }
    metricas_val = {}
    modelos = {}
    for nombre, clasificador in candidatos.items():
        modelo = Pipeline([('preprocessor', crear_preprocesador()), ('classifier', clasificador)])
        modelo.fit(train[FEATURES], train[TARGET])
        modelos[nombre] = modelo
        metricas_val[nombre] = evaluar(nombre + ' / VALIDACIÓN', modelo, val[FEATURES], val[TARGET])
    ganador = max(metricas_val, key=lambda x: metricas_val[x]['f1_macro'])
    mejor = modelos[ganador]
    print(f'\nModelo seleccionado por F1 macro de validación: {ganador}')
    metricas_test = evaluar('PRUEBA FINAL / ' + ganador, mejor, test[FEATURES], test[TARGET])
    OUT.mkdir(parents=True, exist_ok=True)
    if MODEL_FILE.exists() or META_FILE.exists():
        sys.exit('Ya existe el modelo o metadata V1. No se sobrescribe; renombra la versión antes de repetir.')
    joblib.dump(mejor, MODEL_FILE)
    metadata = {'target': TARGET, 'conditional_on': 'falla_futura_30d == 1',
                'features': FEATURES, 'categorical_features': CATEGORICAS,
                'numeric_features': NUMERICAS, 'selected_model': ganador,
                'validation_metrics': metricas_val, 'test_metrics': metricas_test,
                'split': {'train': len(train), 'validation': len(val), 'test': len(test)},
                'classes': [str(x) for x in mejor.named_steps['classifier'].classes_],
                'note': 'Experimental. Predicted probabilities are not calibrated.'}
    META_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nGuardado: {MODEL_FILE}\nMetadata: {META_FILE}')
    print('IMPORTANTE: probabilidades experimentales, no calibradas; no equivalen a diagnóstico.')


if __name__ == '__main__':
    main()
