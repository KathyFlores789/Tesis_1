from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
HISTORICO_PATH = ROOT_DIR / "data" / "raw" / "datos_mantenimiento_quito.csv"
SEGUIMIENTO_PATH = ROOT_DIR / "data" / "seguimiento" / "vehiculos_seguimiento.csv"
SALIDA_DIR = ROOT_DIR / "data" / "reentrenamiento"
CANDIDATO_PATH = SALIDA_DIR / "dataset_candidato_reentrenamiento.csv"

CAMPOS_FALLA_REQUERIDOS = [
    "fecha_falla_futura", "sistema_falla_futura", "repuesto_falla_futura",
    "costo_falla_futura_usd", "tiempo_reparacion_horas",
]

def vacio(valor):
    return pd.isna(valor) or str(valor).strip() == ""

def main():
    print("=" * 68)
    print("QUALITY GATE - DATASET CANDIDATO PARA REENTRENAMIENTO")
    print("=" * 68)

    if not HISTORICO_PATH.is_file():
        sys.exit(f"ERROR: No existe el histórico: {HISTORICO_PATH}")
    if not SEGUIMIENTO_PATH.is_file():
        sys.exit(f"ERROR: No existe el seguimiento: {SEGUIMIENTO_PATH}")

    historico = pd.read_csv(HISTORICO_PATH)
    seguimiento = pd.read_csv(SEGUIMIENTO_PATH)
    errores = []

    for nombre, df in [("histórico", historico), ("seguimiento", seguimiento)]:
        if "vehiculo_id" not in df.columns:
            errores.append(f"El archivo {nombre} no contiene 'vehiculo_id'.")
    if "estado_seguimiento" not in seguimiento.columns:
        errores.append("El seguimiento no contiene 'estado_seguimiento'.")
    if "falla_futura_30d" not in seguimiento.columns:
        errores.append("El seguimiento no contiene 'falla_futura_30d'.")

    if errores:
        for e in errores:
            print("ERROR:", e)
        sys.exit("QUALITY GATE: NO APTO")

    estado = seguimiento["estado_seguimiento"].fillna("").astype(str).str.strip().str.casefold()
    validados = seguimiento.loc[estado.eq("validado")].copy()

    if historico["vehiculo_id"].astype(str).duplicated().any():
        errores.append("Existen vehiculo_id duplicados dentro del histórico.")
    if validados["vehiculo_id"].astype(str).duplicated().any():
        errores.append("Existen vehiculo_id duplicados dentro de los validados.")

    ids_hist = set(historico["vehiculo_id"].astype(str))
    ids_val = set(validados["vehiculo_id"].astype(str))
    solapados = sorted(ids_hist.intersection(ids_val))
    if solapados:
        errores.append("IDs validados ya presentes en histórico: " + ", ".join(solapados[:10]))

    etiquetas = pd.to_numeric(validados["falla_futura_30d"], errors="coerce")
    invalidas = validados.loc[~etiquetas.isin([0, 1]), "vehiculo_id"].astype(str).tolist()
    if invalidas:
        errores.append("Etiquetas falla_futura_30d inválidas: " + ", ".join(invalidas))

    for _, fila in validados.iterrows():
        etiqueta = pd.to_numeric(pd.Series([fila["falla_futura_30d"]]), errors="coerce").iloc[0]
        if etiqueta == 1:
            faltantes = [
                campo for campo in CAMPOS_FALLA_REQUERIDOS
                if campo not in validados.columns or vacio(fila.get(campo))
            ]
            if faltantes:
                errores.append(
                    f"{fila['vehiculo_id']}: presentó falla pero faltan " + ", ".join(faltantes)
                )
            dias = pd.to_numeric(pd.Series([fila.get("dias_hasta_falla")]), errors="coerce").iloc[0]
            if pd.isna(dias) or dias < 0 or dias > 30:
                errores.append(
                    f"{fila['vehiculo_id']}: dias_hasta_falla debe estar entre 0 y 30."
                )

    faltan_columnas = [c for c in historico.columns if c not in validados.columns]
    if faltan_columnas:
        errores.append(
            "A los validados les faltan columnas históricas: " + ", ".join(faltan_columnas)
        )

    print(f"Históricos:                {len(historico)}")
    print(f"Seguimiento total:         {len(seguimiento)}")
    print(f"Seguimiento validado:      {len(validados)}")
    print(f"Seguimiento pendiente:     {len(seguimiento) - len(validados)}")

    if errores:
        print("\nERRORES DEL QUALITY GATE")
        for e in errores:
            print("ERROR:", e)
        sys.exit("\nQUALITY GATE: NO APTO. No se generó ningún dataset candidato.")

    validados_entrenamiento = validados[historico.columns].copy()
    candidato = pd.concat([historico, validados_entrenamiento], ignore_index=True)

    if candidato["vehiculo_id"].astype(str).duplicated().any():
        sys.exit("QUALITY GATE: NO APTO. IDs duplicados después de combinar.")

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    candidato.to_csv(CANDIDATO_PATH, index=False)

    print("\nQUALITY GATE: APTO")
    print(f"Dataset candidato:         {len(candidato)}")
    print(f"Archivo generado:          {CANDIDATO_PATH}")
    print("\nLos archivos histórico y seguimiento NO fueron modificados.")
    print("Este script prepara datos; NO reentrena ni reemplaza modelos.")

if __name__ == "__main__":
    main()
