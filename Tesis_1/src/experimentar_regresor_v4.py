"""Experimento exploratorio V4: ingeniería de variables para costo de falla futura.
Ejecutar desde raíz: python .\\src\\experimentar_regresor_v4.py
Solo escribe CSV nuevos en reports; NO sobrescribe modelos ni datos.
ATENCIÓN: el conjunto test ya se ha consultado en V2/V3; cualquier comparación V4
sobre él es exploratoria, no una validación independiente definitiva.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw' / 'datos_mantenimiento_quito.csv'
REPORTS = ROOT / 'reports'
TARGET = 'costo_falla_futura_usd'
CAT = ['marca','modelo','tipo_vehiculo','combustible','transmision',
       'zona_operacion_quito','tipo_mantenimiento_actual',
       'sistema_intervenido_actual','repuesto_cambiado_actual']
BASE_NUM = ['anio','cilindraje_l','kilometraje_acumulado_km',
            'km_desde_ultimo_mantenimiento','dias_desde_ultimo_mantenimiento',
            'temperatura_promedio_motor_c','presion_aceite_kpa',
            'frenadas_bruscas_por_100km','tiempo_ralenti_min_dia',
            'altitud_operacion_m','tiene_codigo_obd_activo',
            'costo_mantenimiento_actual_usd']
ENGINEERED = ['edad_vehiculo_en_fecha_registro','km_por_anio_vehiculo',
              'km_diarios_desde_mantenimiento','proporcion_km_desde_mantenimiento',
              'log_kilometraje_acumulado','km_por_anio_al_cuadrado']

def add_features(df):
    x = df.copy()
    fecha = pd.to_datetime(x['fecha_mantenimiento'], errors='coerce')
    anio = pd.to_numeric(x['anio'], errors='coerce')
    km = pd.to_numeric(x['kilometraje_acumulado_km'], errors='coerce').clip(lower=0)
    km_desde = pd.to_numeric(x['km_desde_ultimo_mantenimiento'], errors='coerce').clip(lower=0)
    dias = pd.to_numeric(x['dias_desde_ultimo_mantenimiento'], errors='coerce').clip(lower=0)
    edad = (fecha.dt.year - anio).clip(lower=0)
    x['edad_vehiculo_en_fecha_registro'] = edad
    x['km_por_anio_vehiculo'] = km / (edad + 1)
    x['km_diarios_desde_mantenimiento'] = km_desde / dias.clip(lower=1)
    x['proporcion_km_desde_mantenimiento'] = km_desde / km.clip(lower=1)
    x['log_kilometraje_acumulado'] = np.log1p(km)
    x['km_por_anio_al_cuadrado'] = (km / (edad + 1)) ** 2
    x[ENGINEERED] = x[ENGINEERED].replace([np.inf,-np.inf],np.nan)
    return x

def score(y, p):
    return {'mae':mean_absolute_error(y,p),
            'rmse':np.sqrt(mean_squared_error(y,p)), 'r2':r2_score(y,p)}

def make_pipeline(cat, num, estimator):
    prep = ColumnTransformer([
        ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),
                         ('scale',StandardScaler())]),num),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                         ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)
    ])
    return Pipeline([('prep',prep),('model',clone(estimator))])

def predict_usd(model, x, log):
    p = model.predict(x)
    if log: p = np.expm1(np.clip(p,-20,20))
    return np.maximum(p,0)

def main():
    if not DATA.is_file(): raise FileNotFoundError(f'No encuentro {DATA}')
    df = pd.read_csv(DATA)
    required = CAT + BASE_NUM + [TARGET,'falla_futura_30d','fecha_mantenimiento']
    missing = sorted(set(required)-set(df.columns))
    if missing: raise ValueError(f'Faltan columnas: {missing}')
    df['fecha_mantenimiento'] = pd.to_datetime(df['fecha_mantenimiento'],errors='coerce')
    df = df.sort_values('fecha_mantenimiento',kind='stable').reset_index(drop=True)
    d = df[df['falla_futura_30d'].eq(1)].copy()
    d[TARGET] = pd.to_numeric(d[TARGET],errors='coerce')
    if d[TARGET].isna().any() or (d[TARGET]<0).any():
        raise ValueError('Hay costos de falla nulos o negativos en registros positivos.')
    if d['fecha_mantenimiento'].isna().any():
        raise ValueError('Hay fechas de mantenimiento inválidas.')
    d = add_features(d)
    n=len(d); a=int(n*.70); b=int(n*.85)
    train,val,test = d.iloc[:a],d.iloc[a:b],d.iloc[b:]
    print(f'Fallas con costo: {n} | Train: {len(train)} | Validación: {len(val)} | Test: {len(test)}')
    print('AVISO: V2/V3 ya utilizaron este test; V4 es una comparación exploratoria.')
    groups = {
        'Originales': BASE_NUM,
        'Originales + derivadas': BASE_NUM + ENGINEERED,
        'Sin costo actual + derivadas': [c for c in BASE_NUM if c!='costo_mantenimiento_actual_usd']+ENGINEERED,
        'Kilometraje y edad': ['anio','kilometraje_acumulado_km','km_desde_ultimo_mantenimiento',
                               'dias_desde_ultimo_mantenimiento']+ENGINEERED
    }
    algorithms = {
        'ElasticNet':ElasticNet(alpha=.1,l1_ratio=.5,max_iter=10000,random_state=42),
        'Ridge':Ridge(alpha=10),
        'RandomForest':RandomForestRegressor(n_estimators=150,min_samples_leaf=4,
                                             random_state=42,n_jobs=-1),
        'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=7,
                                               min_samples_leaf=10,l2_regularization=10,
                                               learning_rate=.05,random_state=42)
    }
    candidates=[]
    base=DummyRegressor(strategy='median').fit(train[BASE_NUM],train[TARGET])
    candidates.append(('Mediana','Originales','normal',base,BASE_NUM,score(val[TARGET],base.predict(val[BASE_NUM]))))
    for group,nums in groups.items():
        cat = CAT if group!='Kilometraje y edad' else []
        for name,est in algorithms.items():
            for scale in ('normal','log1p'):
                pipe=make_pipeline(cat,nums,est)
                yy=np.log1p(train[TARGET]) if scale=='log1p' else train[TARGET]
                pipe.fit(train[cat+nums],yy)
                p=predict_usd(pipe,val[cat+nums],scale=='log1p')
                candidates.append((name,group,scale,pipe,cat+nums,score(val[TARGET],p)))
    candidates.sort(key=lambda x:(x[5]['mae'],x[5]['rmse']))
    report=pd.DataFrame([dict(modelo=c[0],variables=c[1],escala=c[2],**c[5]) for c in candidates])
    REPORTS.mkdir(exist_ok=True)
    path=REPORTS/'comparacion_regresor_v4_validacion.csv'
    report.to_csv(path,index=False,encoding='utf-8-sig')
    print('\nMEJORES 15 EN VALIDACIÓN (menor MAE):')
    print(report.head(15).round(3).to_string(index=False))
    name,group,scale,chosen,cols,_=candidates[0]
    print(f'\nSELECCIONADO SOLO CON VALIDACIÓN: {name} | {group} | {scale}')
    trainval=d.iloc[:b]
    final=clone(chosen)
    yy=np.log1p(trainval[TARGET]) if scale=='log1p' else trainval[TARGET]
    final.fit(trainval[cols],yy)
    p=predict_usd(final,test[cols],scale=='log1p')
    baseline=np.full(len(test),trainval[TARGET].median())
    print('\nTEST EXPLORATORIO (no es prueba independiente nueva):')
    for label,pred in [('V4 seleccionado',p),('Mediana histórica',baseline)]:
        m=score(test[TARGET],pred)
        print(f"{label}: MAE=${m['mae']:.2f} | RMSE=${m['rmse']:.2f} | R²={m['r2']:.4f}")
    out=test[['fecha_mantenimiento',TARGET]].copy()
    if 'vehiculo_id' in test.columns: out.insert(0,'vehiculo_id',test['vehiculo_id'])
    out['estimado_v4_usd']=p
    out['error_absoluto_usd']=np.abs(out[TARGET]-p)
    pred_path=REPORTS/'predicciones_regresor_v4_test.csv'
    out.to_csv(pred_path,index=False,encoding='utf-8-sig')
    out['rango_real']=pd.qcut(out[TARGET],3,duplicates='drop')
    print('\nERROR POR RANGO REAL (descriptivo):')
    print(out.groupby('rango_real',observed=True).agg(casos=(TARGET,'size'),
        real_promedio=(TARGET,'mean'),estimado_promedio=('estimado_v4_usd','mean'),
        mae=('error_absoluto_usd','mean')).round(2).to_string())
    print(f'\nReportes: {path} y {pred_path}')
    print('No se modificaron los modelos .joblib ni la aplicación.')

if __name__=='__main__': main()
