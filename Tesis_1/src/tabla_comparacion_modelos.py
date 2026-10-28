
import matplotlib.pyplot as plt

modelos = [
    ["Regresión Logística", "0,6815", "0,7676"],
    ["Random Forest", "0,6515", "0,7722"],
    ["Gradient Boosting", "0,6232", "0,7518"]
]

columnas = ["Modelo", "F1-score", "ROC-AUC"]

fig, ax = plt.subplots(figsize=(9, 2.6))
ax.axis("off")

tabla = ax.table(
    cellText=modelos,
    colLabels=columnas,
    cellLoc="center",
    colLoc="center",
    colWidths=[0.48, 0.25, 0.27],
    loc="center"
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(12)
tabla.scale(1, 2.1)

for (fila, columna), celda in tabla.get_celld().items():
    celda.set_edgecolor("#D9DFE7")
    celda.set_linewidth(0.7)

    if fila == 0:
        celda.set_facecolor("#183153")
        celda.set_text_props(color="white", weight="bold")
    elif fila == 1:
        celda.set_facecolor("#E8F2FA")
        celda.set_text_props(weight="bold")
    else:
        celda.set_facecolor("white")

    if columna == 0 and fila > 0:
        celda.get_text().set_ha("left")

plt.savefig(
    "tabla_comparacion_modelos.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
