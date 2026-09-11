"""
Auditoría de coherencia de las variables futuras (T1).

Este módulo analiza la relación entre la variable de clasificación
falla_futura_30d y las variables asociadas al evento futuro.

No modifica el dataset original.
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/datos_mantenimiento_quito.csv")


def load_dataset(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Carga el dataset original.

    Parameters
    ----------
    file_path : Path
        Ruta del dataset.

    Returns
    -------
    pd.DataFrame
        Dataset cargado.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {file_path}"
        )

    return pd.read_csv(file_path)


def audit_failure_relationships(df: pd.DataFrame) -> None:
    """
    Analiza la relación entre falla_futura_30d y las variables futuras.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original.
    """

    print("=" * 80)
    print("AUDITORÍA DE COHERENCIA DE VARIABLES FUTURAS")
    print("=" * 80)

    print("\nDistribución de falla_futura_30d:")
    print(df["falla_futura_30d"].value_counts().sort_index())

    print("\n" + "-" * 80)
    print("NULOS SEGÚN FALLA FUTURA")
    print("-" * 80)

    variables = [
        "dias_hasta_falla",
        "fecha_falla_futura",
        "sistema_falla_futura",
        "repuesto_falla_futura",
        "costo_falla_futura_usd",
        "tiempo_reparacion_horas",
    ]

    for variable in variables:
        print(f"\n{variable}")
        print(
            df.groupby("falla_futura_30d")[variable]
            .apply(lambda x: x.isna().sum())
            .to_string()
        )

    print("\n" + "-" * 80)
    print("VALORES DE VARIABLES FUTURAS SEGÚN FALLA")
    print("-" * 80)

    for variable in variables:
        print(f"\n### {variable}")

        if pd.api.types.is_numeric_dtype(df[variable]):
            print(
                df.groupby("falla_futura_30d")[variable]
                .agg(
                    [
                        "count",
                        "min",
                        "mean",
                        "median",
                        "max",
                    ]
                )
                .round(3)
                .to_string()
            )
        else:
            print(
                pd.crosstab(
                    df["falla_futura_30d"],
                    df[variable],
                    dropna=False,
                ).to_string()
            )

    print("\n" + "-" * 80)
    print("POSIBLES CONTRADICCIONES")
    print("-" * 80)

    no_failure_with_cost = df[
        (df["falla_futura_30d"] == 0)
        & (df["costo_falla_futura_usd"] > 0)
    ]

    no_failure_with_repair_time = df[
        (df["falla_futura_30d"] == 0)
        & (df["tiempo_reparacion_horas"] > 0)
    ]

    failure_without_cost = df[
        (df["falla_futura_30d"] == 1)
        & (df["costo_falla_futura_usd"] <= 0)
    ]

    failure_without_repair_time = df[
        (df["falla_futura_30d"] == 1)
        & (df["tiempo_reparacion_horas"] <= 0)
    ]

    print(
        "\nFalla = 0 pero costo futuro > 0:",
        len(no_failure_with_cost),
    )

    print(
        "Falla = 0 pero tiempo reparación > 0:",
        len(no_failure_with_repair_time),
    )

    print(
        "Falla = 1 pero costo futuro <= 0:",
        len(failure_without_cost),
    )

    print(
        "Falla = 1 pero tiempo reparación <= 0:",
        len(failure_without_repair_time),
    )


def main() -> None:
    """Ejecuta la auditoría de coherencia de T1."""

    df = load_dataset()
    audit_failure_relationships(df)


if __name__ == "__main__":
    main()