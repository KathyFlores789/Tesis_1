from pathlib import Path
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

ARCHIVO_ACTUAL = (
    ROOT_DIR
    / "data"
    / "raw"
    / "datos_mantenimiento_quito.csv"
)

CARPETA_SEGUIMIENTO = (
    ROOT_DIR
    / "data"
    / "seguimiento"
)

ARCHIVO_SEGUIMIENTO = (
    CARPETA_SEGUIMIENTO
    / "vehiculos_seguimiento.csv"
)


# ============================================================
# CARGAR DATOS
# ============================================================

df = pd.read_csv(ARCHIVO_ACTUAL)

print("\n========================================")
print("SEPARACIÓN DE DATOS")
print("========================================")
print(f"Registros encontrados: {len(df)}")


# ============================================================
# IDENTIFICAR REGISTROS PENDIENTES
# ============================================================

# Los vehículos nuevos todavía no tienen resultado real
# de falla futura a 30 días.
pendientes = df["falla_futura_30d"].isna()

df_seguimiento = df.loc[pendientes].copy()
df_historico = df.loc[~pendientes].copy()


print(f"Registros históricos validados: {len(df_historico)}")
print(f"Registros pendientes de seguimiento: {len(df_seguimiento)}")


# ============================================================
# VALIDACIONES DE SEGURIDAD
# ============================================================

if len(df_historico) + len(df_seguimiento) != len(df):
    raise ValueError(
        "ERROR: la suma de históricos y pendientes "
        "no coincide con el dataset original."
    )

if df_seguimiento.empty:
    raise ValueError(
        "ERROR: no se encontraron vehículos pendientes."
    )

print("\nVehículos que pasarán a seguimiento:")

for vehiculo_id in df_seguimiento["vehiculo_id"]:
    print(f" - {vehiculo_id}")


# ============================================================
# CREAR CARPETA DE SEGUIMIENTO
# ============================================================

CARPETA_SEGUIMIENTO.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GUARDAR VEHÍCULOS PENDIENTES
# ============================================================

df_seguimiento.to_csv(
    ARCHIVO_SEGUIMIENTO,
    index=False
)


# ============================================================
# ACTUALIZAR DATASET HISTÓRICO
# ============================================================

df_historico.to_csv(
    ARCHIVO_ACTUAL,
    index=False
)


# ============================================================
# COMPROBACIÓN FINAL
# ============================================================

historico_verificado = pd.read_csv(ARCHIVO_ACTUAL)
seguimiento_verificado = pd.read_csv(ARCHIVO_SEGUIMIENTO)

print("\n========================================")
print("RESULTADO FINAL")
print("========================================")
print(
    f"Dataset histórico: "
    f"{len(historico_verificado)} registros"
)
print(
    f"Dataset seguimiento: "
    f"{len(seguimiento_verificado)} registros"
)
print(
    f"Total conservado: "
    f"{len(historico_verificado) + len(seguimiento_verificado)} registros"
)

print("\nSeparación realizada correctamente.")