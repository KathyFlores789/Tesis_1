from pathlib import Path
import json
import sys
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

METRICAS_PATH = (
    ROOT_DIR
    / "reports"
    / "reentrenamiento"
    / "comparacion_classifier_actual_vs_candidato.csv"
)

METADATA_PATH = (
    ROOT_DIR
    / "reports"
    / "reentrenamiento"
    / "metadata_classifier_candidato_prueba.json"
)

REPORTE_GATE_PATH = (
    ROOT_DIR
    / "reports"
    / "reentrenamiento"
    / "quality_gate_modelo_classifier.csv"
)

DECISION_PATH = (
    ROOT_DIR
    / "reports"
    / "reentrenamiento"
    / "decision_quality_gate_modelo.json"
)


# ============================================================
# CRITERIOS DEL QUALITY GATE
# ============================================================

# Tolerancias máximas de degradación permitidas respecto
# al classifier_v2 actualmente desplegado.

MAX_CAIDA_F1 = 0.01
MAX_CAIDA_AUC = 0.01
MAX_CAIDA_RECALL = 0.02


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main():

    print("=" * 76)
    print("QUALITY GATE DEL MODELO - CLASSIFIER")
    print("=" * 76)

    # --------------------------------------------------------
    # Verificar archivos necesarios
    # --------------------------------------------------------

    if not METRICAS_PATH.is_file():
        sys.exit(
            f"ERROR: No existe el archivo de comparación:\n"
            f"{METRICAS_PATH}"
        )

    if not METADATA_PATH.is_file():
        sys.exit(
            f"ERROR: No existe la metadata del candidato:\n"
            f"{METADATA_PATH}"
        )

    # --------------------------------------------------------
    # Cargar métricas
    # --------------------------------------------------------

    metricas = pd.read_csv(METRICAS_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    columnas_requeridas = {
        "modelo",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    faltantes = columnas_requeridas.difference(
        metricas.columns
    )

    if faltantes:
        sys.exit(
            "ERROR: Faltan métricas requeridas: "
            + ", ".join(sorted(faltantes))
        )

    # --------------------------------------------------------
    # Localizar modelo actual y candidato
    # --------------------------------------------------------

    actual_df = metricas.loc[
        metricas["modelo"] == "classifier_v2_actual"
    ]

    candidato_df = metricas.loc[
        metricas["modelo"]
        == "classifier_candidato_prueba"
    ]

    if len(actual_df) != 1 or len(candidato_df) != 1:
        sys.exit(
            "ERROR: No se encontraron exactamente "
            "los dos modelos esperados."
        )

    actual = actual_df.iloc[0]
    candidato = candidato_df.iloc[0]

    # --------------------------------------------------------
    # Calcular diferencias
    # --------------------------------------------------------

    delta_f1 = float(
        candidato["f1"] - actual["f1"]
    )

    delta_auc = float(
        candidato["roc_auc"] - actual["roc_auc"]
    )

    delta_recall = float(
        candidato["recall"] - actual["recall"]
    )

    # --------------------------------------------------------
    # Controles técnicos
    # --------------------------------------------------------

    controles = [
        {
            "control": "F1",
            "actual": float(actual["f1"]),
            "candidato": float(candidato["f1"]),
            "delta": delta_f1,
            "limite_caida": MAX_CAIDA_F1,
            "cumple": delta_f1 >= -MAX_CAIDA_F1,
        },
        {
            "control": "ROC-AUC",
            "actual": float(actual["roc_auc"]),
            "candidato": float(candidato["roc_auc"]),
            "delta": delta_auc,
            "limite_caida": MAX_CAIDA_AUC,
            "cumple": delta_auc >= -MAX_CAIDA_AUC,
        },
        {
            "control": "Recall",
            "actual": float(actual["recall"]),
            "candidato": float(candidato["recall"]),
            "delta": delta_recall,
            "limite_caida": MAX_CAIDA_RECALL,
            "cumple": delta_recall >= -MAX_CAIDA_RECALL,
        },
    ]

    reporte = pd.DataFrame(controles)

    gate_tecnico = bool(
        reporte["cumple"].all()
    )

    # --------------------------------------------------------
    # Detectar si la ejecución usa datos ficticios/de prueba
    # --------------------------------------------------------

    uso = str(
        metadata.get("uso", "")
    ).strip().casefold()

    nota = str(
        metadata.get("nota", "")
    ).strip().casefold()

    datos_prueba = (
        "prueba" in uso
        or "fictici" in nota
        or "no_promover" in uso
    )

    # --------------------------------------------------------
    # Decisión
    # --------------------------------------------------------

    if not gate_tecnico:

        decision = "RECHAZADO"

        motivo = (
            "El candidato no cumple uno o más límites "
            "técnicos del Quality Gate."
        )

    elif datos_prueba:

        decision = (
            "APTO_TECNICAMENTE_PROMOCION_BLOQUEADA"
        )

        motivo = (
            "El candidato cumple los límites técnicos, "
            "pero la ejecución contiene registros "
            "ficticios/de prueba. No se autoriza promoción."
        )

    else:

        decision = "APTO_PARA_REVISION_HUMANA"

        motivo = (
            "El candidato cumple los límites técnicos. "
            "La promoción no es automática y requiere "
            "revisión y aprobación humana."
        )

    # --------------------------------------------------------
    # IMPORTANTE:
    # Nunca existe promoción automática
    # --------------------------------------------------------

    promocion_automatica = False

    promocion_autorizada = (
        gate_tecnico
        and not datos_prueba
    )

    # --------------------------------------------------------
    # Guardar resultados
    # --------------------------------------------------------

    REPORTE_GATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    reporte.to_csv(
        REPORTE_GATE_PATH,
        index=False
    )

    salida = {
        "decision": decision,
        "gate_tecnico_superado": gate_tecnico,
        "datos_prueba_detectados": datos_prueba,
        "promocion_automatica": promocion_automatica,
        "promocion_autorizada": promocion_autorizada,
        "requiere_revision_humana": True,
        "modelo_actual": "classifier_v2.joblib",
        "modelo_candidato":
            "classifier_candidato_prueba.joblib",
        "criterios": {
            "max_caida_f1": MAX_CAIDA_F1,
            "max_caida_roc_auc": MAX_CAIDA_AUC,
            "max_caida_recall": MAX_CAIDA_RECALL,
        },
        "motivo": motivo,
    }

    DECISION_PATH.write_text(
        json.dumps(
            salida,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print("\nCRITERIOS")

    print(
        f"F1      : caída máxima permitida = "
        f"{MAX_CAIDA_F1:.4f}"
    )

    print(
        f"ROC-AUC : caída máxima permitida = "
        f"{MAX_CAIDA_AUC:.4f}"
    )

    print(
        f"Recall  : caída máxima permitida = "
        f"{MAX_CAIDA_RECALL:.4f}"
    )

    print("\nRESULTADOS")

    for control in controles:

        estado = (
            "CUMPLE"
            if control["cumple"]
            else "NO CUMPLE"
        )

        print(
            f"{control['control']:<8} "
            f"actual={control['actual']:.4f} | "
            f"candidato={control['candidato']:.4f} | "
            f"delta={control['delta']:+.4f} | "
            f"{estado}"
        )

    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("DECISION DEL QUALITY GATE")
    print("=" * 76)

    print(
        "Gate técnico superado :",
        "SI" if gate_tecnico else "NO"
    )

    print(
        "Datos de prueba       :",
        "SI" if datos_prueba else "NO"
    )

    print(
        "Promoción automática  : NO"
    )

    print(
        "Decisión              :",
        decision
    )

    print(
        "Motivo                :",
        motivo
    )

    print("\nModelo desplegado NO modificado.")
    print("App V10 NO modificada.")

    print(
        f"\nReporte generado:\n"
        f"{REPORTE_GATE_PATH}"
    )

    print(
        f"\nDecisión guardada:\n"
        f"{DECISION_PATH}"
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()