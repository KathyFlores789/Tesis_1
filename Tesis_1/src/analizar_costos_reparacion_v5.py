"""Análisis exploratorio de costos reales de fallas futuras (NO entrena modelos).

Desde la raíz del proyecto: python .\\src\\analizar_costos_reparacion_v5.py
Solo escribe reportes CSV nuevos; no modifica el dataset, los modelos ni Streamlit.
Los datos de fallas futuras se usan únicamente para describir los costos observados,
NUNCA como variables disponibles al predecir una futura falla.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw' / 'datos_mantenimiento_quito.csv'
OUT = ROOT / 'reports'
TARGET = 'costo_falla_futura_usd'
FUTURE = ['dias_hasta_falla','fecha_falla_futura','sistema_falla_futura',
          'repuesto_falla_futura','tiempo_reparacion_horas']
NUMERIC = ['anio','kilometraje_acumulado_km','km_desde_ultimo_mantenimiento',
           'dias_desde_ultimo_mantenimiento','temperatura_promedio_motor_c',
           'presion_aceite_kpa','frenadas_bruscas_por_100km',
           'tiempo_ralenti_min_dia','altitud_operacion_m',
           'costo_mantenimiento_actual_usd','cilindraje_l']
CATEGORICAL = ['tipo_vehiculo','tipo_mantenimiento_actual','sistema_intervenido_actual',
               'combustible','tiene_codigo_obd_activo','zona_operacion_quito']

def show(title, data, max_rows=20):
    print('\n' + '='*74 + '\n' + title + '\n' + '='*74)
    print(data.head(max_rows).to_string(index=False) if isinstance(data,pd.DataFrame) else data.to_string())

def group_cost(df, column, min_n=10):
    g = df.groupby(column,dropna=False)[TARGET].agg(
        casos='count',media='mean',mediana='median',minimo='min',maximo='max'
    ).reset_index()
    g = g[g.casos>=min_n].sort_values('casos',ascending=False)
    return g.round(2)

def main():
    if not DATA.exists():
        raise FileNotFoundError(f'No encuentro {DATA}. Ejecuta desde la raíz del proyecto.')
    d = pd.read_csv(DATA)
    required = [TARGET,'falla_futura_30d','fecha_mantenimiento']+FUTURE+NUMERIC+CATEGORICAL
    missing = sorted(set(required)-set(d.columns))
    if missing: raise ValueError(f'Faltan columnas: {missing}')
    d['fecha_mantenimiento'] = pd.to_datetime(d.fecha_mantenimiento,errors='coerce')
    d['fecha_falla_futura'] = pd.to_datetime(d.fecha_falla_futura,errors='coerce')
    d = d.sort_values('fecha_mantenimiento',kind='stable').reset_index(drop=True)
    p = d[d.falla_futura_30d.eq(1)].copy()
    p[TARGET] = pd.to_numeric(p[TARGET],errors='coerce')
    if p[TARGET].isna().any():
        raise ValueError('Existen fallas positivas sin costo futuro; revisar el dataset.')
    n=len(p); a=int(n*.70); b=int(n*.85)
    train=p.iloc[:a].copy(); val=p.iloc[a:b].copy(); test=p.iloc[b:].copy()
    print(f'Filas totales: {len(d)} | Fallas con costo: {n}')
    print(f'Partición temporal: Train {len(train)}, Validation {len(val)}, Test {len(test)}')
    print('NOTA: el test ya fue consultado en V2/V3/V4. NO se usa aquí para buscar variables.')
    print('Los patrones exploratorios siguientes se calculan SOLO en TRAIN.')

    # Controles de calidad de etiquetas futuras, informativos, sobre todos los positivos.
    delta=(p.fecha_falla_futura-p.fecha_mantenimiento).dt.days
    reported=pd.to_numeric(p.dias_hasta_falla,errors='coerce')
    checks=pd.DataFrame([{
        'fallas_positivas':n,
        'sin_fecha_mantenimiento':int(p.fecha_mantenimiento.isna().sum()),
        'sin_fecha_falla':int(p.fecha_falla_futura.isna().sum()),
        'falla_fuera_de_1_a_30_dias':int((delta.notna()&~delta.between(1,30)).sum()),
        'dias_hasta_falla_inconsistentes':int((delta.notna()&reported.notna()&(delta!=reported)).sum()),
        'costo_futuro_no_positivo':int((p[TARGET]<=0).sum()),
        'ids_duplicados_en_todo_dataset':int(d.vehiculo_id.duplicated().sum()) if 'vehiculo_id' in d else np.nan
    }])
    show('1. CONTROL DE CALIDAD DE ETIQUETAS',checks)
    show('2. COSTOS OBSERVADOS SOLO EN TRAIN (USD)',train[TARGET].describe(percentiles=[.1,.25,.5,.75,.9,.95]).round(2))

    # Correlaciones descriptivas sin usar etiquetas futuras como predictores.
    x=train[NUMERIC].apply(pd.to_numeric,errors='coerce').copy()
    x['edad_en_mantenimiento']=(train.fecha_mantenimiento.dt.year-pd.to_numeric(train.anio,errors='coerce')).clip(lower=0)
    x['km_por_anio_aprox']=pd.to_numeric(train.kilometraje_acumulado_km,errors='coerce')/(x.edad_en_mantenimiento+1)
    correlations=x.apply(lambda c:c.corr(train[TARGET],method='spearman')).rename('spearman_train').reset_index().rename(columns={'index':'variable'})
    correlations['magnitud']=correlations.spearman_train.abs()
    correlations=correlations.sort_values('magnitud',ascending=False).drop(columns='magnitud').round(3)
    show('3. CORRELACIONES CON COSTO FUTURO (SOLO TRAIN)',correlations)

    # Agrupaciones de características conocidas antes de la falla.
    tables={}
    for col in CATEGORICAL:
        g=group_cost(train,col,min_n=10)
        tables[f'actual_{col}']=g
        show(f'4. COSTO SEGÚN {col} (CONOCIDO AL PREDECIR, TRAIN)',g,12)
    for col in ['kilometraje_acumulado_km','anio','costo_mantenimiento_actual_usd']:
        series=pd.to_numeric(train[col],errors='coerce')
        try:
            labels=pd.qcut(series,q=4,duplicates='drop')
            t=train.assign(rango=labels).groupby('rango',observed=True)[TARGET].agg(
                casos='count',media='mean',mediana='median').reset_index().round(2)
            tables[f'rangos_{col}']=t
            show(f'5. COSTO POR CUARTILES DE {col} (TRAIN)',t)
        except ValueError:
            print(f'No hay suficiente variación para cuartiles de {col}')

    # Solo auditoría retrospectiva: no usar estas columnas como entradas del modelo.
    for col in ['sistema_falla_futura','repuesto_falla_futura']:
        g=group_cost(train,col,min_n=5)
        tables[f'auditoria_{col}']=g
        show(f'6. SOLO AUDITORÍA RETROSPECTIVA: {col} (NO ES PREDICTOR)',g,20)

    OUT.mkdir(parents=True,exist_ok=True)
    checks.to_csv(OUT/'v5_calidad_etiquetas.csv',index=False,encoding='utf-8-sig')
    correlations.to_csv(OUT/'v5_correlaciones_train.csv',index=False,encoding='utf-8-sig')
    for name,tab in tables.items():
        tab.to_csv(OUT/f'v5_{name}.csv',index=False,encoding='utf-8-sig')
    print('\nAnálisis finalizado. Reportes v5_*.csv en reports/.')
    print('Ningún modelo fue entrenado ni guardado; el dataset y Streamlit siguen intactos.')

if __name__=='__main__':
    main()
