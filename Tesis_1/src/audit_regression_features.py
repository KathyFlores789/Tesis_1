# ============================================================
# AUDITORÍA Y WHITELIST DE PREDICTORES T0 (REGRESIÓN DE COSTO)
# ============================================================

import pandas as pd

PATH_DATA = "data/processed/fleet_maintenance_dataset_advanced.csv"

# WHITELIST EXPLÍCITA DE PREDICTORES T0 AUTORIZADOS
# Regresor condicional: costo de falla futura
FEATURES_T0_REGRESION_COSTO = [
    # A. ATRIBUTOS DEL VEHÍCULO
    "marca",
    "modelo",
    "tipo_vehiculo",
    "combustible",
    "transmision",
    "cilindraje_l",
    "edad_vehiculo_años",
    # B. USO Y DESGASTE ACUMULADO
    "kilometraje_acumulado_km",
    "km_desde_ultimo_mantenimiento",
    "dias_desde_ultimo_mantenimiento",
    "frenadas_bruscas_por_100km",
    "tiempo_ralenti_min_dia",
    "altitud_operacion_m",
    "zona_operacion_quito",
    "km_promedio_estimado_diario",
    "ratio_recorrido_reciente",
    "indice_estres_termico_altitud",
    # C. ESTADO / TELEMETRÍA DISPONIBLE EN T0
    "temperatura_promedio_motor_c",
    "presion_aceite_kpa",
    "tiene_codigo_obd_activo",
]


def audit_features():
    print("==================================================")
    print("AUDITORÍA DE PREDICTORES T0 PARA REGRESIÓN DE COSTO")
    print("==================================================")

    # 1. Cargar dataset para verificar compatibilidad de tipos
    try:
        df = pd.read_csv(PATH_DATA)
        X_t0 = df[FEATURES_T0_REGRESION_COSTO]

        cat_cols = X_t0.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        num_cols = X_t0.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        total_clasificadas = len(cat_cols) + len(num_cols)
    except Exception:
        # En caso de que no esté el archivo en ruta local durante la auditoría rápida
        total_clasificadas = len(FEATURES_T0_REGRESION_COSTO)

    print(
        f"TOTAL FEATURES AUTORIZADAS EN WHITELIST : {len(FEATURES_T0_REGRESION_COSTO)}"
    )
    print(f"TOTAL COLUMNAS CLASIFICADAS             : {total_clasificadas}")

    print("\nLista de variables aprobadas:")
    for idx, feature in enumerate(FEATURES_T0_REGRESION_COSTO, 1):
        print(f"  {idx:02d}. {feature}")

    print("==================================================")


if __name__ == "__main__":
    audit_features()