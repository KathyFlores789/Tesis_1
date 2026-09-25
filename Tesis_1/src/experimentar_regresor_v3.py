"""Experimento V3 de costos futuros, sin modificar el modelo V2 ni la aplicación.
Desde la raíz del proyecto: python src/experimentar_regresor_v3.py
Selecciona exclusivamente con validación y evalúa una vez en test.
"""
from pathlib import Path
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw' / 'datos_mantenimiento_quito.csv'
REPORTS = ROOT / 'reports'
TARGET = 'costo_falla_futura_usd'
FEATURES = [
    'marca','modelo','anio','tipo_vehiculo','combustible','cilindraje_l',
    'transmision','kilometraje_acumulado_km','km_desde_ultimo_mantenimiento',
    'dias_desde_ultimo_mantenimiento','temperatura_promedio_motor_c',
    'presion_aceite_kpa','frenadas_bruscas_por_100km','tiempo_ralenti_min_dia',
    'altitud_operacion_m','zona_operacion_quito','tiene_codigo_obd_activo',
    'tipo_mantenimiento_actual','sistema_intervenido_actual',
    'repuesto_cambiado_actual','costo_mantenimiento_actual_usd'
]
CAT = ['marca','modelo','tipo_vehiculo','combustible','transmision',
       'zona_operacion_quito','tipo_mantenimiento_actual',
       'sistema_intervenido_actual','repuesto_cambiado_actual']
NUM = [c for c in FEATURES if c not in CAT]

def metrics(y, p):
    return dict(mae=mean_absolute_error(y,p),
                rmse=np.sqrt(mean_squared_error(y,p)),r2=r2_score(y,p))

def pipeline(estimator):
    prep = ColumnTransformer([
        ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),
                         ('scaler',StandardScaler())]),NUM),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                         ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CAT)
    ])
    return Pipeline([('preprocessor',prep),('model',clone(estimator))])

def main():
    if not DATA.is_file():
        raise FileNotFoundError(f'No encuentro {DATA}')
    df = pd.read_csv(DATA)
    missing = [c for c in FEATURES+[TARGET,'fecha_mantenimiento','falla_futura_30d'] if c not in df]
    if missing: raise ValueError(f'Faltan columnas: {missing}')
    df['fecha_mantenimiento'] = pd.to_datetime(df['fecha_mantenimiento'],errors='coerce')
    df = df.sort_values('fecha_mantenimiento').reset_index(drop=True)
    d = df.loc[df['falla_futura_30d'].eq(1)].copy()
    d[TARGET] = pd.to_numeric(d[TARGET],errors='coerce')
    if d[TARGET].isna().any() or (d[TARGET]<0).any():
        raise ValueError('Hay costos futuros nulos o negativos entre las fallas.')
    n=len(d); i=int(n*.70); j=int(n*.85)
    tr, va, te = d.iloc[:i],d.iloc[i:j],d.iloc[j:]
    print(f'Fallas con costo: {n}; Train: {len(tr)}; Validación: {len(va)}; Test: {len(te)}')
    if n<50: raise ValueError('Muy pocos registros para este experimento.')
    algorithms = {
        'Ridge': Ridge(alpha=10),
        'ElasticNet': ElasticNet(alpha=.1,l1_ratio=.5,max_iter=10000,random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=250,min_samples_leaf=3,random_state=42,n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=120,learning_rate=.04,max_depth=2,min_samples_leaf=5,random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=150,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=10,l2_regularization=10,random_state=42)
    }
    # Baseline solo con datos de train: compite en validación, nunca utiliza test.
    candidates=[]
    base=DummyRegressor(strategy='median').fit(tr[FEATURES],tr[TARGET])
    candidates.append(('Mediana (referencia)','normal',base,
                       metrics(va[TARGET],base.predict(va[FEATURES]))))
    for name, est in algorithms.items():
        for scale in ('normal','log1p'):
            model=pipeline(est)
            y=np.log1p(tr[TARGET]) if scale=='log1p' else tr[TARGET]
            model.fit(tr[FEATURES],y)
            pred=model.predict(va[FEATURES])
            if scale=='log1p': pred=np.expm1(np.clip(pred,-20,20))
            pred=np.maximum(pred,0)
            candidates.append((name,scale,model,metrics(va[TARGET],pred)))
    ranked=sorted(candidates,key=lambda x:(x[3]['mae'],x[3]['rmse']))
    REPORTS.mkdir(exist_ok=True)
    val_table=pd.DataFrame([dict(modelo=n,escala=s,**m) for n,s,_,m in ranked])
    val_path=REPORTS/'comparacion_regresor_v3_validacion.csv'
    val_table.to_csv(val_path,index=False,encoding='utf-8-sig')
    print('\nRESULTADOS EN VALIDACIÓN (ordenados por MAE)')
    print(val_table.round(3).to_string(index=False))
    name,scale,chosen,_=ranked[0]
    print(f'\nSELECCIONADO CON VALIDACIÓN: {name}, escala {scale}')
    # Test intacto hasta después de seleccionar. Reentrenar en train+validación.
    trainval=d.iloc[:j]
    if name=='Mediana (referencia)':
        final=DummyRegressor(strategy='median')
    else:
        final=clone(chosen)
    yfit=np.log1p(trainval[TARGET]) if scale=='log1p' else trainval[TARGET]
    final.fit(trainval[FEATURES],yfit)
    pred=final.predict(te[FEATURES])
    if scale=='log1p': pred=np.expm1(np.clip(pred,-20,20))
    pred=np.maximum(pred,0)
    baseline=np.full(len(te),trainval[TARGET].median())
    print('\nTEST (una sola evaluación del modelo seleccionado)')
    for label, pp in [('V3 seleccionado',pred),('Mediana histórica',baseline)]:
        m=metrics(te[TARGET],pp)
        print(f"{label}: MAE=${m['mae']:.2f} | RMSE=${m['rmse']:.2f} | R²={m['r2']:.4f}")
    out=te[['vehiculo_id','fecha_mantenimiento',TARGET]].copy()
    out['estimado_v3_usd']=pred
    out['error_absoluto_usd']=abs(out[TARGET]-pred)
    pred_path=REPORTS/'predicciones_regresor_v3_test.csv'
    out.to_csv(pred_path,index=False,encoding='utf-8-sig')
    # Los rangos son descriptivos, definidos después de la evaluación, no sirven para seleccionar.
    out['rango_real']=pd.qcut(out[TARGET],3,duplicates='drop')
    print('\nERROR POR RANGO REAL (solo análisis descriptivo de test)')
    print(out.groupby('rango_real',observed=True).agg(
        casos=(TARGET,'size'),real_promedio=(TARGET,'mean'),
        estimado_promedio=('estimado_v3_usd','mean'),mae=('error_absoluto_usd','mean')
    ).round(2).to_string())
    print(f'\nReportes: {val_path} y {pred_path}')
    print('No se ha guardado ni sobrescrito ningún modelo .joblib.')
    print('IMPORTANTE: si gana la referencia, no sustituir el regresor V2 automáticamente.')

if __name__=='__main__': main()
