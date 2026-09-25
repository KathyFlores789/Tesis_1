"""Diagnóstico de solo lectura del regresor de costos futuros V2.
Ejecutar desde la raíz del proyecto: python -m src.diagnosticar_costos
No entrena ni sobrescribe modelos, datos o reportes.
"""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw' / 'datos_mantenimiento_quito.csv'
MODEL = ROOT / 'models' / 'regressor_v2.joblib'
TARGET = 'costo_falla_futura_usd'
FEATURES = [
    'marca', 'modelo', 'anio', 'tipo_vehiculo', 'combustible',
    'cilindraje_l', 'transmision', 'kilometraje_acumulado_km',
    'km_desde_ultimo_mantenimiento', 'dias_desde_ultimo_mantenimiento',
    'temperatura_promedio_motor_c', 'presion_aceite_kpa',
    'frenadas_bruscas_por_100km', 'tiempo_ralenti_min_dia',
    'altitud_operacion_m', 'zona_operacion_quito', 'tiene_codigo_obd_activo',
    'tipo_mantenimiento_actual', 'sistema_intervenido_actual',
    'repuesto_cambiado_actual', 'costo_mantenimiento_actual_usd'
]

def mostrar_resumen(nombre, valores):
    serie = pd.Series(valores, dtype='float64')
    print(f'\n{nombre}:')
    print(serie.describe(percentiles=[.10, .25, .50, .75, .90]).round(2).to_string())

def metricas(y, p, etiqueta):
    print(f'\n{etiqueta}')
    print(f'MAE:  ${mean_absolute_error(y, p):.2f}')
    print(f'RMSE: ${np.sqrt(mean_squared_error(y, p)):.2f}')
    print(f'R²:   {r2_score(y, p):.4f}')


def main():
    print('=' * 65)
    print('DIAGNÓSTICO DE COSTOS FUTUROS - SOLO LECTURA')
    print('=' * 65)
    for ruta in (DATA, MODEL):
        if not ruta.is_file():
            sys.exit(f'No se encontró: {ruta}')
    df = pd.read_csv(DATA)
    df['fecha_mantenimiento'] = pd.to_datetime(df['fecha_mantenimiento'], errors='coerce')
    # Replica el orden y los cortes del script original de entrenamiento.
    df = df.sort_values('fecha_mantenimiento').reset_index(drop=True)
    df_reg = df.loc[df['falla_futura_30d'].eq(1)].copy()
    df_reg[TARGET] = pd.to_numeric(df_reg[TARGET], errors='coerce')
    if df_reg[TARGET].isna().any():
        sys.exit('Hay fallas futuras sin costo registrado: revisar antes de evaluar.')
    faltantes = [c for c in FEATURES if c not in df_reg.columns]
    if faltantes:
        sys.exit(f'Faltan columnas: {faltantes}')
    n = len(df_reg)
    train_end, val_end = int(n * .70), int(n * .85)
    trainval = df_reg.iloc[:val_end]
    test = df_reg.iloc[val_end:]
    print(f'Filas totales: {len(df)} | Fallas con costo: {n}')
    print(f'Train + Validation: {len(trainval)} | Test: {len(test)}')
    print('NOTA: esta evaluación presupone que el modelo guardado procede')
    print('del entrenamiento V2 con estos mismos datos y partición temporal.')
    modelo = joblib.load(MODEL)  # Cargar solo modelos locales de confianza.
    pred_log = np.asarray(modelo.predict(test[FEATURES]), dtype=float)
    pred = np.maximum(np.expm1(pred_log), 0)
    reales = test[TARGET].to_numpy(dtype=float)
    baseline = np.full(len(test), trainval[TARGET].median(), dtype=float)
    mostrar_resumen('Costo futuro REAL en test (USD)', reales)
    mostrar_resumen('Costo futuro PREDICHO en test (USD)', pred)
    print('\nCOMPARACIÓN EN EL MISMO TEST')
    metricas(reales, pred, 'Regresor V2 (salida log1p convertida a USD)')
    metricas(reales, baseline, 'Referencia: mediana de Train + Validation')
    print(f'\nMediana usada como referencia: ${baseline[0]:.2f}')
    print(f'Desviación estándar predicciones: ${np.std(pred):.2f}')
    print(f'Desviación estándar costos reales: ${np.std(reales):.2f}')
    rangos = pd.qcut(test[TARGET], q=3, duplicates='drop')
    tabla = pd.DataFrame({'rango_real': rangos, 'real': reales, 'predicho': pred})
    tabla['error_abs'] = abs(tabla['real'] - tabla['predicho'])
    print('\nERRORES SEGÚN EL COSTO REAL (TERCIOS DEL TEST)')
    print(tabla.groupby('rango_real', observed=True).agg(
        casos=('real', 'size'), costo_real_medio=('real', 'mean'),
        costo_estimado_medio=('predicho', 'mean'), mae=('error_abs', 'mean')
    ).round(2).to_string())
    print('\nPrimeros 10 casos del test:')
    print(pd.DataFrame({'real_usd': reales[:10], 'estimado_usd': pred[:10],
                        'error_absoluto': abs(reales[:10] - pred[:10])}).round(2).to_string(index=False))
    print('\nDiagnóstico finalizado. No se guardó ni modificó ningún archivo.')

if __name__ == '__main__':
    main()
