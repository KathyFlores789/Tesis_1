"""Figura 7.2.1: preparación y partición temporal de datos (clasificador V2)."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SALIDA = Path(__file__).resolve().parent
fig, ax = plt.subplots(figsize=(12, 7.2))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

def caja(x, y, w, h, titulo, detalle='', color='#EAF2FB', borde='#517FA8'):
    p = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=0.16',linewidth=1.6,edgecolor=borde,facecolor=color)
    ax.add_patch(p)
    ax.text(x+w/2,y+h*(0.70 if '\n' in detalle else (0.64 if detalle else 0.5)),titulo,ha='center',va='center',fontsize=13,fontweight='bold',color='#1F3448')
    if detalle:
        ax.text(x+w/2,y+h*(0.28 if '\n' in detalle else 0.27),detalle,ha='center',va='center',fontsize=(9.3 if '\n' in detalle else 10.5),linespacing=1.2,color='#38526B')

def flecha(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=17,linewidth=1.7,color='#66788A'))

caja(3.7,6.6,4.6,0.85,'1.000 registros históricos', 'Datos de mantenimiento vehicular')
flecha(6,6.53,6,6.13)
caja(3.7,5.18,4.6,0.85,'Preparación y preprocesamiento','Revisión · Limpieza · Imputación\nEscalado · Codificación',color='#E9F5F0',borde='#4C9078')
flecha(6,5.1,6,4.69)
caja(3.7,3.73,4.6,0.85,'Partición temporal','Ordenamiento por fecha de mantenimiento',color='#FFF4DB',borde='#C39B46')
# branching
ax.plot([6,6],[3.7,3.30],color='#66788A',linewidth=1.7)
ax.plot([2.2,9.8],[3.30,3.30],color='#66788A',linewidth=1.7)
for x in (2.2,6,9.8): flecha(x,3.3,x,2.84)
caja(0.5,1.55,3.4,1.15,'699 registros','Entrenamiento · 69,9 %',color='#E6F0FA',borde='#4B83B5')
caja(4.3,1.55,3.4,1.15,'151 registros','Validación · 15,1 %',color='#E6F4EB',borde='#4B9470')
caja(8.1,1.55,3.4,1.15,'150 registros','Prueba · 15,0 %',color='#FFF0E4',borde='#C68A5B')
ax.text(6,0.73,'El conjunto de prueba se reserva para evaluar el modelo con datos no usados en el entrenamiento.',ha='center',fontsize=10,color='#4A5C6B')
fig.savefig(SALIDA/'figura_7_2_1_preparacion_particion_temporal.png',dpi=300,bbox_inches='tight',facecolor='white')
fig.savefig(SALIDA/'figura_7_2_1_preparacion_particion_temporal.svg',bbox_inches='tight',facecolor='white')
plt.close(fig)
print('Archivos generados en:',SALIDA)
