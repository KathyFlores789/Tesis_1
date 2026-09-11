"""
scripts/auditoria_temporal.py
-----------------------------
Script de auditoría para analizar el rango cronológico, la densidad mensual 
y la evolución de la tasa de fallas a lo largo del tiempo.
"""

import sys
from pathlib import Path
import pandas as pd

# Configuración de rutas para ejecución modular
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import load_dataset
from src.preprocessing import aplicar_limpieza_base

def auditar_linea_tiempo():
    df_raw = load_dataset()
    df = aplicar_limpieza_base(df_raw)
    
    # Ordenar por fecha de mantenimiento
    df_sorted = df.sort_values(by='fecha_mantenimiento').reset_index(drop=True)
    
    fecha_min = df_sorted['fecha_mantenimiento'].min()
    fecha_max = df_sorted['fecha_mantenimiento'].max()
    dias_totales = (fecha_max - fecha_min).days
    
    print("\n" + "="*80)
    print("AUDITORÍA CRONOLÓGICA DE DATOS (T0)")
    print("="*80)
    print(f"Total Registros           : {len(df_sorted)}")
    print(f"Fecha Inicial (Primer T0) : {fecha_min.strftime('%Y-%m-%d')}")
    print(f"Fecha Final (Último T0)   : {fecha_max.strftime('%Y-%m-%d')}")
    print(f"Ventana Temporal Total    : {dias_totales} días")
    
    # Agrupación por Año-Mes
    df_sorted['año_mes'] = df_sorted['fecha_mantenimiento'].dt.to_period('M')
    resumen_mensual = df_sorted.groupby('año_mes').agg(
        total_registros=('falla_futura_30d', 'count'),
        total_fallas=('falla_futura_30d', 'sum'),
        tasa_falla=('falla_futura_30d', 'mean')
    )
    resumen_mensual['tasa_falla_%'] = (resumen_mensual['tasa_falla'] * 100).round(2)
    
    print("\nDISTRIBUCIÓN MENSUAL Y TASA DE FALLA EN EL TIEMPO:")
    print("-" * 80)
    print(resumen_mensual[['total_registros', 'total_fallas', 'tasa_falla_%']].to_string())
    
    # Proyección de cortes 70% / 15% / 15%
    idx_train = int(len(df_sorted) * 0.70)
    idx_val = int(len(df_sorted) * 0.85)
    
    train_set = df_sorted.iloc[:idx_train]
    val_set = df_sorted.iloc[idx_train:idx_val]
    test_set = df_sorted.iloc[idx_val:]
    
    print("\n" + "="*80)
    print("PROYECCIÓN DE PUNTOS DE CORTE TEMPORAL (TIME-BASED SPLIT 70 / 15 / 15)")
    print("="*80)
    print(f"1. TRAIN SET (Pasado):")
    print(f"   Rango  : {train_set['fecha_mantenimiento'].min().strftime('%Y-%m-%d')} a {train_set['fecha_mantenimiento'].max().strftime('%Y-%m-%d')}")
    print(f"   Muestras: {len(train_set)} | Tasa Falla: {train_set['falla_futura_30d'].mean():.2%}")
    
    print(f"\n2. VALIDATION SET (Intermedio):")
    print(f"   Rango  : {val_set['fecha_mantenimiento'].min().strftime('%Y-%m-%d')} a {val_set['fecha_mantenimiento'].max().strftime('%Y-%m-%d')}")
    print(f"   Muestras: {len(val_set)} | Tasa Falla: {val_set['falla_futura_30d'].mean():.2%}")
    
    print(f"\n3. TEST SET (Futuro):")
    print(f"   Rango  : {test_set['fecha_mantenimiento'].min().strftime('%Y-%m-%d')} a {test_set['fecha_mantenimiento'].max().strftime('%Y-%m-%d')}")
    print(f"   Muestras: {len(test_set)} | Tasa Falla: {test_set['falla_futura_30d'].mean():.2%}")
    print("="*80)

if __name__ == "__main__":
    auditar_linea_tiempo()