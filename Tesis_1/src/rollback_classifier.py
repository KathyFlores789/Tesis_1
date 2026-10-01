from pathlib import Path
import hashlib
import json
import sys


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODELO_PRODUCCION = (
    ROOT_DIR
    / "models"
    / "classifier_v2.joblib"
)

BACKUP_MODELO = (
    ROOT_DIR
    / "models"
    / "versions"
    / "classifier_v2_backup.joblib"
)

BACKUP_METADATA = (
    ROOT_DIR
    / "models"
    / "versions"
    / "classifier_v2_backup_metadata.json"
)

# IMPORTANTE:
# True = solo simula. No modifica archivos.
DRY_RUN = True


# ============================================================
# SHA-256
# ============================================================

def calcular_sha256(ruta):

    sha256 = hashlib.sha256()

    with open(ruta, "rb") as archivo:

        for bloque in iter(
            lambda: archivo.read(1024 * 1024),
            b""
        ):
            sha256.update(bloque)

    return sha256.hexdigest()


# ============================================================
# PROCESO
# ============================================================

def main():

    print("=" * 76)
    print("ROLLBACK CONTROLADO - CLASSIFIER")
    print("=" * 76)

    if DRY_RUN:
        print("\nMODO: SIMULACION (DRY-RUN)")
        print("No se modificara ningun archivo.")

    # --------------------------------------------------------
    # 1. Verificar archivos
    # --------------------------------------------------------

    if not BACKUP_MODELO.is_file():
        sys.exit(
            f"\nERROR: No existe el backup:\n"
            f"{BACKUP_MODELO}"
        )

    if not BACKUP_METADATA.is_file():
        sys.exit(
            f"\nERROR: No existe la metadata:\n"
            f"{BACKUP_METADATA}"
        )

    if not MODELO_PRODUCCION.is_file():
        sys.exit(
            f"\nERROR: No existe el modelo de produccion:\n"
            f"{MODELO_PRODUCCION}"
        )

    print("\nBackup localizado: SI")
    print("Metadata localizada: SI")
    print("Modelo de produccion localizado: SI")

    # --------------------------------------------------------
    # 2. Leer metadata
    # --------------------------------------------------------

    with open(
        BACKUP_METADATA,
        "r",
        encoding="utf-8"
    ) as archivo:

        metadata = json.load(archivo)

    hash_metadata = metadata.get(
        "sha256_backup"
    )

    if not hash_metadata:
        sys.exit(
            "\nERROR: La metadata no contiene "
            "sha256_backup."
        )

    # --------------------------------------------------------
    # 3. Comprobar integridad del backup
    # --------------------------------------------------------

    hash_backup_actual = calcular_sha256(
        BACKUP_MODELO
    )

    integridad = (
        hash_backup_actual == hash_metadata
    )

    if not integridad:

        sys.exit(
            "\nROLLBACK BLOQUEADO.\n"
            "El SHA-256 del backup no coincide "
            "con la metadata."
        )

    print("\nIntegridad del backup: CORRECTA")

    print(
        "SHA-256:"
    )

    print(
        hash_backup_actual
    )

    # --------------------------------------------------------
    # 4. Información del rollback
    # --------------------------------------------------------

    print("\n" + "-" * 76)

    print("PLAN DE ROLLBACK")

    print("-" * 76)

    print(
        "\nModelo que se restauraria:"
    )

    print(
        BACKUP_MODELO
    )

    print(
        "\nDestino:"
    )

    print(
        MODELO_PRODUCCION
    )

    # --------------------------------------------------------
    # 5. DRY-RUN
    # --------------------------------------------------------

    if DRY_RUN:

        print("\n" + "=" * 76)

        print("RESULTADO DE LA SIMULACION")

        print("=" * 76)

        print(
            "\nBackup localizado        : SI"
        )

        print(
            "Integridad verificada    : SI"
        )

        print(
            "Archivos modificados     : NO"
        )

        print(
            "Rollback ejecutado       : NO"
        )

        print(
            "Modelo produccion tocado : NO"
        )

        print(
            "\nRESULTADO: LISTO PARA ROLLBACK"
        )

        print(
            "\nLa simulacion finalizo correctamente."
        )

        return

    # --------------------------------------------------------
    # SEGURIDAD
    # --------------------------------------------------------
    #
    # En esta version NO implementamos aun la sustitucion
    # fisica del modelo.
    #
    # Incluso si DRY_RUN se cambia accidentalmente a False,
    # el script se detendra aqui.
    # --------------------------------------------------------

    sys.exit(
        "\nSEGURIDAD: la restauracion fisica "
        "todavia no esta habilitada."
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()