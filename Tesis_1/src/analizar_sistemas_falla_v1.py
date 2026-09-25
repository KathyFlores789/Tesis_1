"""Diagnóstico exploratorio de sistemas de falla futura (sin entrenar modelos).
Ejecutar desde la raíz: python src/analizar_sistemas_falla_v1.py
Solo analiza registros con falla futura conocida y usa TRAIN temporal (70 %)
para explorar asociaciones sin consultar VALIDACIÓN ni TEST.
No modifica el CSV original ni los modelos.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'data' / 'raw' / 'datos_mantenimiento_quito.csv'
TARGET = 'sistema_falla_futura'
CAT = ['marca', 'modelo', 'tipo_vehiculo', 'combustible', 'transmision',
       'zona_operacion_quito', 'tiene_codigo_obd_activo',
       'tipo_mantenimiento_actual', 'sistema_intervenido_actual',
       'repuesto_cambiado_actual']
NUM = ['anio', 'cilindraje_l', 'kilometraje_acumulado_km',
       'km_desde_ultimo_mantenimiento', 'dias_desde_ultimo_mantenimiento',
       'temperatura_promedio_motor_c', 'presion_aceite_kpa',
       'frenadas_bruscas_por_100km', 'tiempo_ralenti_min_dia',
       'altitud_operacion_m', 'costo_mantenimiento_actual_usd']


def cramer_v(tabla):
    """Cramér V descriptivo (no prueba capacidad predictiva)."""
    obs = tabla.to_numpy(dtype=float)
    n = obs.sum()
    if n == 0 or min(obs.shape) < 2:
        return np.nan
    esperado = obs.sum(axis=1, keepdims=True) @ obs.sum(axis=0, keepdims=True) / n
    chi2 = np.divide((obs - esperado)**2, esperado,
                     out=np.zeros_like(obs), where=esperado > 0).sum()
    return float(np.sqrt((chi2 / n) / (min(obs.shape) - 1)))


def eta_cuadrado(x, y):
    tmp = pd.DataFrame({'x': pd.to_numeric(x, errors='coerce'), 'y': y}).dropna()
    if len(tmp) < 2:
        return np.nan
    total = ((tmp.x - tmp.x.mean())**2).sum()
    if total == 0:
        return 0.0
    grupos = tmp.groupby('y')['x']
    entre = sum(len(v) * (v.mean() - tmp.x.mean())**2 for _, v in grupos)
    return float(entre / total)


def main():
    if not CSV.exists():
        raise FileNotFoundError(f'No se encontró: {CSV}')
    df = pd.read_csv(CSV)
    necesarias = ['fecha_mantenimiento', 'falla_futura_30d', TARGET] + CAT + NUM
    faltantes = sorted(set(necesarias) - set(df.columns))
    if faltantes:
        raise ValueError(f'Columnas faltantes: {faltantes}')
    df['fecha_mantenimiento'] = pd.to_datetime(df.fecha_mantenimiento, errors='coerce')
    df['falla_futura_30d'] = pd.to_numeric(df.falla_futura_30d, errors='coerce')
    df[TARGET] = df[TARGET].astype('string').str.strip()
    df = df.loc[(df.falla_futura_30d == 1) & df[TARGET].notna() &
                (df[TARGET] != '') & df.fecha_mantenimiento.notna()].copy()
    df = df.sort_values('fecha_mantenimiento').reset_index(drop=True)
    n_train = int(len(df) * .70)
    train = df.iloc[:n_train].copy()
    print(f'Fallas con sistema conocido: {len(df)}')
    print(f'Análisis EXCLUSIVAMENTE sobre TRAIN temporal: {len(train)} registros')
    print('Validación y prueba reservadas; no se examinan aquí.\n')
    print('=== DISTRIBUCIÓN DE SISTEMAS EN TRAIN ===')
    print(train[TARGET].value_counts().to_string())
    print('\n=== DATOS FALTANTES EN TRAIN ===')
    falt = (train[CAT + NUM].isna().mean() * 100).sort_values(ascending=False)
    print(falt.round(1).to_string(float_format=lambda x: f'{x:.1f}%'))
    print('\n=== VARIABLES NUMÉRICAS: ETA CUADRADO (ASOCIACIÓN DESCRIPTIVA) ===')
    en = pd.Series({c: eta_cuadrado(train[c], train[TARGET]) for c in NUM})
    print(en.sort_values(ascending=False).round(4).to_string())
    print('\n=== MEDIANAS POR SISTEMA DE VARIABLES NUMÉRICAS ===')
    medianas = train.assign(**{c: pd.to_numeric(train[c], errors='coerce') for c in NUM})
    print(medianas.groupby(TARGET)[NUM].median().round(1).to_string())
    print('\n=== VARIABLES CATEGÓRICAS: CRAMÉR V (ASOCIACIÓN DESCRIPTIVA) ===')
    asociaciones = {}
    for c in CAT:
        t = pd.crosstab(train[c].fillna('(sin dato)'), train[TARGET])
        asociaciones[c] = cramer_v(t)
    print(pd.Series(asociaciones).sort_values(ascending=False).round(4).to_string())
    for c in ['tiene_codigo_obd_activo', 'tipo_mantenimiento_actual',
              'sistema_intervenido_actual']:
        print(f'\n=== DISTRIBUCIÓN: {c} frente a sistema futuro ===')
        tabla = pd.crosstab(train[c].fillna('(sin dato)'), train[TARGET])
        print(tabla.to_string())
    print('\nIMPORTANTE: asociaciones exploratorias, no prueban causalidad ni capacidad predictiva.')
    print('No utilices sistema/repuesto/costo de la falla FUTURA como variables de entrada.')
    print('No se han modificado datos ni modelos.')


if __name__ == '__main__':
    main()
