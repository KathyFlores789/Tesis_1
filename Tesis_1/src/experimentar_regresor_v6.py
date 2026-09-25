"""V6 exploratorio: predicción de costo futuro, sin tocar test ni guardar modelos.
Colocar en src/ y ejecutar: python .\\src\\experimentar_regresor_v6.py
Se reutiliza validación ya consultada: resultados NO son evaluación independiente.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw' / 'datos_mantenimiento_quito.csv'
REPORTS = ROOT / 'reports'
TARGET = 'costo_falla_futura_usd'
BASE = ['anio','kilometraje_acumulado_km','km_desde_ultimo_mantenimiento',
        'dias_desde_ultimo_mantenimiento','cilindraje_l','temperatura_promedio_motor_c',
        'presion_aceite_kpa','frenadas_bruscas_por_100km','tiempo_ralenti_min_dia',
        'altitud_operacion_m','tiene_codigo_obd_activo']
DERIVED = ['edad_vehiculo','km_por_anio','log_km','edad_por_km','km_desde_por_edad']

def metrics(y,p):
    return {'mae':mean_absolute_error(y,p),'rmse':np.sqrt(mean_squared_error(y,p)),
            'r2':r2_score(y,p),'sesgo_medio':float(np.mean(p-y)),
            'dispersion_pred':float(np.std(p))}

def add_features(d):
    x=d.copy()
    anio=pd.to_numeric(x['anio'],errors='coerce')
    km=pd.to_numeric(x['kilometraje_acumulado_km'],errors='coerce').clip(lower=0)
    desde=pd.to_numeric(x['km_desde_ultimo_mantenimiento'],errors='coerce').clip(lower=0)
    edad=(x['fecha_mantenimiento'].dt.year-anio).clip(lower=0)
    x['edad_vehiculo']=edad
    x['km_por_anio']=km/(edad+1)
    x['log_km']=np.log1p(km)
    x['edad_por_km']=edad*km/100000
    x['km_desde_por_edad']=desde/(edad+1)
    return x.replace([np.inf,-np.inf],np.nan)

def pipeline(est, cols):
    prep=ColumnTransformer([('num',Pipeline([('imputar',SimpleImputer(strategy='median')),
        ('escalar',StandardScaler())]),cols)])
    return Pipeline([('preparar',prep),('modelo',clone(est))])

def main():
    if not DATA.exists(): raise FileNotFoundError(f'No se encuentra: {DATA}')
    df=pd.read_csv(DATA)
    needed=BASE+[TARGET,'falla_futura_30d','fecha_mantenimiento']
    missing=sorted(set(needed)-set(df.columns))
    if missing: raise ValueError(f'Faltan columnas: {missing}')
    df['fecha_mantenimiento']=pd.to_datetime(df['fecha_mantenimiento'],errors='coerce')
    df=df.sort_values('fecha_mantenimiento',kind='stable').reset_index(drop=True)
    d=df.loc[df['falla_futura_30d'].eq(1)].copy()
    d[TARGET]=pd.to_numeric(d[TARGET],errors='coerce')
    if d[TARGET].isna().any() or (d[TARGET]<0).any() or d['fecha_mantenimiento'].isna().any():
        raise ValueError('Revisar fechas y costos de los registros con falla.')
    d=add_features(d)
    n=len(d); a=int(n*.70); b=int(n*.85)
    train,val=d.iloc[:a],d.iloc[a:b]
    print(f'Fallas: {n} | Train: {len(train)} | Validación: {len(val)} | Test reservado: {n-b} (NO se usa)')
    print('ADVERTENCIA: validación ya utilizada en V2–V4; este experimento es exploratorio.')
    groups={'Kilometraje y edad':['anio','kilometraje_acumulado_km','edad_vehiculo','km_por_anio'],
            'Indicadores mecánicos':BASE,
            'Mecánicos + derivadas':BASE+DERIVED,
            'Solo km + derivadas':['kilometraje_acumulado_km','km_desde_ultimo_mantenimiento']+DERIVED}
    models={'Ridge_1':Ridge(alpha=1),'Ridge_30':Ridge(alpha=30),
            'Huber':HuberRegressor(epsilon=1.5,alpha=1,max_iter=2000),
            'RandomForest':RandomForestRegressor(n_estimators=180,min_samples_leaf=8,max_features=.8,random_state=42,n_jobs=-1),
            'ExtraTrees':ExtraTreesRegressor(n_estimators=180,min_samples_leaf=8,random_state=42,n_jobs=-1),
            'HistGB':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=15,
                learning_rate=.05,l2_regularization=20,random_state=42)}
    records=[]; candidates=[]
    for group,cols in groups.items():
        for name,est in models.items():
            for scale in ('normal','log1p'):
                m=pipeline(est,cols)
                y=np.log1p(train[TARGET]) if scale=='log1p' else train[TARGET]
                m.fit(train[cols],y)
                pred=m.predict(val[cols]); pred=np.expm1(np.clip(pred,-20,20)) if scale=='log1p' else pred
                pred=np.maximum(pred,0)
                result=dict(modelo=name,variables=group,escala=scale,**metrics(val[TARGET],pred))
                records.append(result); candidates.append((result,m,cols,pred))
    for strategy in ('median','mean'):
        m=DummyRegressor(strategy=strategy).fit(train[BASE],train[TARGET])
        pred=m.predict(val[BASE]); records.append(dict(modelo=f'Referencia_{strategy}',variables='—',escala='normal',**metrics(val[TARGET],pred)))
    table=pd.DataFrame(records).sort_values(['mae','rmse'])
    REPORTS.mkdir(parents=True,exist_ok=True)
    path=REPORTS/'comparacion_regresor_v6_validacion.csv'
    table.to_csv(path,index=False,encoding='utf-8-sig')
    print('\nMEJORES 15 EN VALIDACIÓN (exploratorio):')
    print(table.head(15).round(3).to_string(index=False))
    chosen=min(candidates,key=lambda c:(c[0]['mae'],c[0]['rmse']))
    result,_,_,pred=chosen
    print('\nCANDIDATO V6:',result['modelo'],'|',result['variables'],'|',result['escala'])
    print(f"MAE=${result['mae']:.2f} | RMSE=${result['rmse']:.2f} | R²={result['r2']:.4f}")
    out=val[['fecha_mantenimiento',TARGET]].copy()
    out['prediccion_v6_usd']=pred; out['error_absoluto_usd']=abs(out[TARGET]-pred)
    out['rango_real']=pd.qcut(out[TARGET],3,duplicates='drop')
    print('\nERROR POR RANGO REAL EN VALIDACIÓN (descriptivo):')
    print(out.groupby('rango_real',observed=True).agg(casos=(TARGET,'size'),
        costo_real_medio=(TARGET,'mean'),predicho_medio=('prediccion_v6_usd','mean'),
        mae=('error_absoluto_usd','mean')).round(2).to_string())
    out.to_csv(REPORTS/'predicciones_regresor_v6_validacion.csv',index=False,encoding='utf-8-sig')
    print(f'\nReportes guardados en: {REPORTS}')
    print('TEST NO UTILIZADO. No se guardaron modelos ni se modificó Streamlit.')

if __name__=='__main__': main()
