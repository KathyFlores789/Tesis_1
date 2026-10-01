from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil
import sys

import pandas as pd


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODELO_PRODUCCION_PATH = (
    ROOT_DIR
    / "models"
    / "classifier_v2.joblib"
)

VERSIONS_DIR = (
    ROOT_DIR
    / "models"
    / "versions"
)

BACKUP_PATH = (
    VERSIONS_DIR
    / "classifier_v2_backup.joblib"
)

METADATA_PATH = (
    VERSIONS_DIR
    / "classifier_v2_backup_metadata.json"
)

METRICAS_PATH = (
    ROOT_DIR
    / "reports"
    / "reentrenamiento"
    / "comparacion_classifier_actual_vs_candidato.csv"
)


# ============================================================
# FUNCIÓN PARA CALCULAR SHA-256
# ============================================================

def calcular_sha256(ruta_archivo):

    sha256 = hashlib.sha256()

    with open(ruta_archivo, "rb") as archivo:

        for bloque in iter(
            lambda: archivo.read(1024 * 1024),
            b""
        ):
            sha256.update(bloque)

    return sha256.hexdigest()


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main():

    print("=" * 76)
    print("VERSIONAMIENTO SEGURO - CLASSIFIER V2")
    print("=" * 76)

    print(
        "\nEste proceso NO reemplaza ni modifica "
        "el modelo actualmente desplegado."
    )

    # --------------------------------------------------------
    # 1. Verificar modelo de producción
    # --------------------------------------------------------

    if not MODELO_PRODUCCION_PATH.is_file():

        sys.exit(
            f"\nERROR: No existe el modelo de producción:\n"
            f"{MODELO_PRODUCCION_PATH}"
        )

    # --------------------------------------------------------
    # 2. Crear carpeta versions si fuera necesario
    # --------------------------------------------------------

    VERSIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 3. Evitar sobrescribir un backup existente
    # --------------------------------------------------------

    if BACKUP_PATH.exists():

        sys.exit(
            "\nSEGURIDAD: ya existe un backup con este nombre.\n"
            "No se sobrescribió ningún archivo.\n\n"
            f"Archivo existente:\n{BACKUP_PATH}"
        )

    if METADATA_PATH.exists():

        sys.exit(
            "\nSEGURIDAD: ya existe metadata para este backup.\n"
            "No se sobrescribió ningún archivo.\n\n"
            f"Archivo existente:\n{METADATA_PATH}"
        )

    # --------------------------------------------------------
    # 4. Hash del modelo original
    # --------------------------------------------------------

    hash_original = calcular_sha256(
        MODELO_PRODUCCION_PATH
    )

    print(
        "\nSHA-256 del modelo original:"
    )

    print(
        hash_original
    )

    # --------------------------------------------------------
    # 5. Crear copia de seguridad
    # --------------------------------------------------------

    shutil.copy2(
        MODELO_PRODUCCION_PATH,
        BACKUP_PATH
    )

    print(
        "\nBackup creado:"
    )

    print(
        BACKUP_PATH
    )

    # --------------------------------------------------------
    # 6. Comprobar integridad de la copia
    # --------------------------------------------------------

    hash_backup = calcular_sha256(
        BACKUP_PATH
    )

    integridad_correcta = (
        hash_original == hash_backup
    )

    if not integridad_correcta:

        # Si algo raro ocurrió, eliminamos la copia defectuosa
        if BACKUP_PATH.exists():
            BACKUP_PATH.unlink()

        sys.exit(
            "\nERROR: la copia no coincide con el modelo original.\n"
            "El backup defectuoso fue eliminado."
        )

    print(
        "\nVerificación de integridad: CORRECTA"
    )

    print(
        "Original y backup tienen el mismo SHA-256."
    )

    # --------------------------------------------------------
    # 7. Recuperar métricas de referencia, si existen
    # --------------------------------------------------------

    metricas_referencia = {}

    if METRICAS_PATH.is_file():

        try:

            metricas = pd.read_csv(
                METRICAS_PATH
            )

            fila_actual = metricas.loc[
                metricas["modelo"]
                == "classifier_v2_actual"
            ]

            if len(fila_actual) == 1:

                fila = fila_actual.iloc[0]

                metricas_referencia = {
                    "accuracy": float(
                        fila["accuracy"]
                    ),
                    "precision": float(
                        fila["precision"]
                    ),
                    "recall": float(
                        fila["recall"]
                    ),
                    "f1": float(
                        fila["f1"]
                    ),
                    "roc_auc": float(
                        fila["roc_auc"]
                    ),
                    "tn": int(
                        fila["tn"]
                    ),
                    "fp": int(
                        fila["fp"]
                    ),
                    "fn": int(
                        fila["fn"]
                    ),
                    "tp": int(
                        fila["tp"]
                    ),
                }

        except Exception as error:

            print(
                "\nADVERTENCIA:"
                " no fue posible leer las métricas."
            )

            print(
                error
            )

    # --------------------------------------------------------
    # 8. Crear metadata del modelo versionado
    # --------------------------------------------------------

    metadata = {

        "modelo": "classifier_v2",

        "archivo_produccion":
            str(MODELO_PRODUCCION_PATH),

        "archivo_backup":
            str(BACKUP_PATH),

        "version": "v2",

        "tipo_registro":
            "backup_modelo_produccion",

        "estado":
            "produccion_actual_respaldada",

        "fecha_versionamiento":
            datetime.now().astimezone().isoformat(),

        "sha256_original":
            hash_original,

        "sha256_backup":
            hash_backup,

        "integridad_verificada":
            integridad_correcta,

        "metricas_referencia":
            metricas_referencia,

        "umbral_clasificacion":
            0.50,

        "promocion_automatica":
            False,

        "rollback_disponible":
            True,

        "nota": (
            "Copia de seguridad del classifier_v2 "
            "actualmente desplegado. Este proceso "
            "no promovió ni reemplazó ningún modelo."
        ),
    }

    # --------------------------------------------------------
    # 9. Guardar metadata
    # --------------------------------------------------------

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 10. Resultado
    # --------------------------------------------------------

    print(
        "\n" + "=" * 76
    )

    print(
        "VERSIONAMIENTO COMPLETADO"
    )

    print(
        "=" * 76
    )

    print(
        "\nModelo de producción:"
    )

    print(
        MODELO_PRODUCCION_PATH
    )

    print(
        "\nBackup:"
    )

    print(
        BACKUP_PATH
    )

    print(
        "\nMetadata:"
    )

    print(
        METADATA_PATH
    )

    print(
        "\nIntegridad verificada: SI"
    )

    print(
        "Modelo de producción modificado: NO"
    )

    print(
        "App V10 modificada: NO"
    )

    print(
        "Candidato promovido: NO"
    )

    print(
        "\nEl classifier_v2 continúa siendo "
        "el modelo activo."
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()