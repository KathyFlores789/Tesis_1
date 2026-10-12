from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Resultados del clasificador V2 sobre 150 registros de prueba.
# Filas: clase real (sin falla, con falla).
# Columnas: clase predicha (sin falla, con falla).
matriz = np.array([[57, 19], [29, 45]])

carpeta = Path(__file__).resolve().parent
fig, ax = plt.subplots(figsize=(7.5, 6.1))
im = ax.imshow(matriz, cmap='Blues', vmin=0, vmax=65)
ax.set_xticks([0, 1], labels=['Sin falla', 'Con falla'])
ax.set_yticks([0, 1], labels=['Sin falla', 'Con falla'])
ax.set_xlabel('Predicción del modelo', fontsize=12, labelpad=12)
ax.set_ylabel('Resultado real', fontsize=12, labelpad=12)
ax.set_title('Matriz de confusión del clasificador V2\nFalla futura a 30 días · 150 registros de prueba', fontsize=14, weight='bold', pad=18)
for i in range(2):
    for j in range(2):
        valor = matriz[i, j]
        ax.text(j, i, str(valor), ha='center', va='center', fontsize=23,
                weight='bold', color='white' if valor >= 38 else '#20364a')
ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=3)
ax.tick_params(which='minor', bottom=False, left=False)
fig.colorbar(im, ax=ax, shrink=.75, label='Número de registros')
fig.tight_layout()
for extension in ('png', 'svg'):
    destino = carpeta / f'figura_7_2_2_matriz_confusion.{extension}'
    fig.savefig(destino, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Generado: {destino}')
plt.close(fig)
