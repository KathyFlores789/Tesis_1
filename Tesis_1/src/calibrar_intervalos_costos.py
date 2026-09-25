"""Experimento de intervalos de prediccion para costo de falla futura.

EJECUCION (desde la raiz del proyecto):
    python src/calibrar_intervalos_costos.py

No modifica los datos, los modelos guardados ni Streamlit.
IMPORTANTE: el test historico ya fue consultado en V2-V4; los resultados
sobre ese test son EXPLORATORIOS, no una validacion independiente nueva.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'data' / 'raw' / 'datos_mantenimiento_quito.csv'
MODEL = ROOT / 'models' / 'regressor_v2.joblib'
REPORTS = ROOT / 'reports'
ALPHA = 0.10  # cobertura nominal marginal del 90 %
TARGET = 'costo_falla_futura_usd'
FEATURES = [
    'marca', 'modelo', 'anio', 'tipo_vehiculo', 'combustible', 'cilindraje_l',
    'transmision', 'kilometraje_acumulado_km', 'km_desde_ultimo_mantenimiento',
    'dias_desde_ultimo_mantenimiento', 'temperatura_promedio_motor_c',
    'presion_aceite_kpa', 'frenadas_bruscas_por_100km', 'tiempo_ralenti_min_dia',
    'altitud_operacion_m', 'zona_operacion_quito', 'tiene_codigo_obd_activo',
    'tipo_mantenimiento_actual', 'sistema_intervenido_actual',
    'repuesto_cambiado_actual', 'costo_mantenimiento_actual_usd'
]


def main():
    for path in (CSV, MODEL):
        if not path.exists():
            raise FileNotFoundError(f'No se encuentra: {path}')
    df = pd.read_csv(CSV)
    df['fecha_mantenimiento'] = pd.to_datetime(df['fecha_mantenimiento'], errors='coerce')
    df = df.loc[df['falla_futura_30d'].eq(1)].copy()
    if df['fecha_mantenimiento'].isna().any():
        raise ValueError('Hay fallas positivas con fecha de mantenimiento invalida.')
    df = df.sort_values('fecha_mantenimiento').reset_index(drop=True)
    if df[TARGET].isna().any() or (df[TARGET] < 0).any():
        raise ValueError('Hay fallas sin costo futuro valido o con costos negativos.')
    if not len(df):
        raise ValueError('No hay registros con falla futura.')
    n = len(df)
    train_end, val_end = int(n * .70), int(n * .85)
    train, cal, test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
    print('=' * 70)
    print('INTERVALOS DE COSTO FUTURO - SPLIT CONFORMAL EXPLORATORIO')
    print('=' * 70)
    print(f'Fallas: {n} | Entrenamiento: {len(train)} | Calibracion: {len(cal)} | Test historico: {len(test)}')
    print('ADVERTENCIA: el test historico ya se utilizo en V2-V4. NO es prueba independiente nueva.')
    print('Se entrena una COPIA del pipeline V2 con solo TRAIN; el modelo guardado NO se modifica.')
    # V2 fue entrenado en escala log1p; clonar conserva el preprocesamiento y algoritmo.
    saved = joblib.load(MODEL)
    pipeline = clone(saved)
    pipeline.fit(train[FEATURES], np.log1p(train[TARGET].to_numpy()))
    pred_cal = np.maximum(np.expm1(pipeline.predict(cal[FEATURES])), 0)
    residuals = np.abs(cal[TARGET].to_numpy() - pred_cal)
    # Cuantil conformal de muestra finita: orden ceil((n_cal+1)*(1-alpha)).
    rank = int(np.ceil((len(cal) + 1) * (1 - ALPHA)))
    q = np.sort(residuals)[min(rank, len(cal)) - 1] if rank <= len(cal) else np.inf
    print('\nCALIBRACION (solo para fijar el ancho; no es evaluacion de cobertura):')
    print(f'MAE calibracion: ${mean_absolute_error(cal[TARGET], pred_cal):.2f}')
    print(f'Cuantil absoluto conformal: ${q:.2f} (rango {rank} de {len(cal)})')
    print(f'Ancho total nominal del intervalo: ${2*q:.2f}')
    print('\nTEST HISTORICO - EVALUACION SOLO EXPLORATORIA:')
    pred = np.maximum(np.expm1(pipeline.predict(test[FEATURES])), 0)
    lo = np.maximum(pred - q, 0)
    hi = pred + q
    y = test[TARGET].to_numpy()
    covered = (y >= lo) & (y <= hi)
    print(f'MAE puntual: ${mean_absolute_error(y, pred):.2f}')
    print(f'Cobertura observada: {covered.mean():.1%} ({covered.sum()}/{len(test)})')
    print(f'Ancho medio: ${(hi - lo).mean():.2f}')
    print(f'Cobertura nominal objetivo: {1-ALPHA:.0%} (marginal; no garantiza cobertura individual)')
    print('\nEJEMPLOS (primeros 5 casos):')
    for actual, p, low, high in zip(y[:5], pred[:5], lo[:5], hi[:5]):
        print(f'Real ${actual:,.2f} | Estimado ${p:,.2f} | Intervalo ${low:,.2f} - ${high:,.2f}')
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / 'intervalos_costos_exploratorio.csv'
    pd.DataFrame({'fecha_mantenimiento': test['fecha_mantenimiento'].to_numpy(),
                  'costo_real_usd': y, 'prediccion_usd': pred,
                  'limite_inferior_usd': lo, 'limite_superior_usd': hi,
                  'cubierto': covered}).to_csv(out, index=False)
    print(f'\nReporte: {out}')
    print('NO se guardo ningun modelo ni se modifico la aplicacion.')
    print('NO aplicar este ancho directamente al modelo V2 desplegado: V2 fue entrenado con TRAIN+VALIDATION.')
    print('Para desplegar intervalos se requiere calibracion independiente para el modelo final.')


if __name__ == '__main__':
    main()
