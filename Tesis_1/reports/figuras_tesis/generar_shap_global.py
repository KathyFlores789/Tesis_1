"""Genera la figura SHAP global del clasificador V2 sin reentrenar modelos.
Ejecutar desde la raíz: python reports/figuras_tesis/generar_shap_global.py
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

CARPETA_FIGURAS = Path(__file__).resolve().parent
CARPETA_REPORTS = CARPETA_FIGURAS.parent
CSV = CARPETA_REPORTS / 'shap_importancia_global_v2.csv'

ETIQUETAS = {
    'km_desde_ultimo_mantenimiento': 'Km desde último mantenimiento',
    'tiene_codigo_obd_activo': 'Código OBD activo',
    'frenadas_bruscas_por_100km': 'Frenadas bruscas por 100 km',
    'presion_aceite_kpa': 'Presión de aceite',
    'anio': 'Año del vehículo',
    'temperatura_promedio_motor_c': 'Temperatura promedio del motor',
    'zona_operacion_quito_Sur (Quitumbe/Recreo)': 'Zona de operación: sur de Quito',
    'altitud_operacion_m': 'Altitud de operación',
    'modelo_Santa Fe': 'Modelo: Santa Fe',
    'sistema_intervenido_actual_Sistema de Enfriamiento': 'Sistema intervenido: enfriamiento',
}

with CSV.open(encoding='utf-8-sig', newline='') as archivo:
    datos = list(csv.DictReader(archivo))

mayores = sorted(datos, key=lambda x: float(x['mean_abs_shap']), reverse=True)[:10]
mayores.reverse()
valores = [float(f['mean_abs_shap']) for f in mayores]
labels = [ETIQUETAS.get(f['feature'], f['feature'].replace('_', ' ')) for f in mayores]

fig, ax = plt.subplots(figsize=(11.2, 6.2))
fig.patch.set_facecolor('white')
barras = ax.barh(labels, valores, color='#3978a8', height=0.67)
ax.set_title('Importancia global de variables mediante SHAP', fontsize=16, weight='bold', pad=16)
ax.set_xlabel('Impacto SHAP medio absoluto', fontsize=12)
ax.set_xlim(0, max(valores) * 1.19)
ax.grid(axis='x', alpha=0.18, linestyle='--')
ax.set_axisbelow(True)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.tick_params(axis='y', length=0, labelsize=11)
ax.tick_params(axis='x', labelsize=10)
for barra, valor in zip(barras, valores):
    ax.text(valor + max(valores) * .015, barra.get_y() + barra.get_height()/2,
            f'{valor:.4f}'.replace('.', ','), va='center', fontsize=10)
fig.subplots_adjust(left=.34, right=.95, bottom=.13, top=.88)
for ext in ('png', 'svg'):
    destino = CARPETA_FIGURAS / f'figura_7_2_4_shap_global_top10.{ext}'
    fig.savefig(destino, dpi=300, facecolor='white')
    print(f'Generado: {destino}')
plt.close(fig)
