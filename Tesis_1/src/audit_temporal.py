"""
Auditoría de consistencia temporal del dataset de mantenimiento.

Verifica la relación entre:
    - fecha_mantenimiento
    - fecha_falla_futura
    - dias_hasta_falla
    - falla_futura_30d

Este módulo no modifica el dataset original.
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
        Ruta del archivo CSV.

    Returns
    -------
    pd.DataFrame
        Dataset cargado.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("El dataset está vacío.")

    return df


def prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las variables de fecha a formato datetime.

    Se trabaja sobre una copia para garantizar que el dataset
    original no sea modificado.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original.

    Returns
    -------
    pd.DataFrame
        Copia con las fechas convertidas.
    """

    data = df.copy()

    data["fecha_mantenimiento_dt"] = pd.to_datetime(
        data["fecha_mantenimiento"],
        errors="coerce",
    )

    data["fecha_falla_futura_dt"] = pd.to_datetime(
        data["fecha_falla_futura"],
        errors="coerce",
    )

    return data


def audit_date_conversion(df: pd.DataFrame) -> None:
    """
    Verifica si existen fechas que no pudieron convertirse.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con las fechas convertidas.
    """

    print("\n" + "=" * 80)
    print("1. AUDITORÍA DE CONVERSIÓN DE FECHAS")
    print("=" * 80)

    for column in [
        "fecha_mantenimiento_dt",
        "fecha_falla_futura_dt",
    ]:
        invalid = int(df[column].isna().sum())

        print(f"{column}: {invalid} fechas no convertibles")


def audit_failure_dates(df: pd.DataFrame) -> None:
    """
    Analiza las fechas de falla en los registros donde ocurrió una falla.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con fechas convertidas.
    """

    failures = df[df["falla_futura_30d"] == 1].copy()

    print("\n" + "=" * 80)
    print("2. AUDITORÍA DE FECHAS DE FALLA")
    print("=" * 80)

    print(f"\nRegistros con falla: {len(failures)}")

    before_maintenance = (
        failures["fecha_falla_futura_dt"]
        < failures["fecha_mantenimiento_dt"]
    )

    same_day = (
        failures["fecha_falla_futura_dt"]
        == failures["fecha_mantenimiento_dt"]
    )

    missing_failure_date = (
        failures["fecha_falla_futura_dt"].isna()
    )

    print(
        "\nFalla antes del mantenimiento:",
        int(before_maintenance.sum()),
    )

    print(
        "Falla el mismo día del mantenimiento:",
        int(same_day.sum()),
    )

    print(
        "Falla con fecha faltante:",
        int(missing_failure_date.sum()),
    )


def audit_failure_interval(df: pd.DataFrame) -> None:
    """
    Comprueba el intervalo entre mantenimiento y falla.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con fechas convertidas.
    """

    failures = df[df["falla_futura_30d"] == 1].copy()

    failures["dias_calculados"] = (
        failures["fecha_falla_futura_dt"]
        - failures["fecha_mantenimiento_dt"]
    ).dt.days

    print("\n" + "=" * 80)
    print("3. INTERVALO ENTRE MANTENIMIENTO Y FALLA")
    print("=" * 80)

    print("\nEstadísticas de días calculados:")

    print(
        failures["dias_calculados"]
        .describe()
        .round(3)
        .to_string()
    )

    outside_range = failures[
        (failures["dias_calculados"] < 1)
        | (failures["dias_calculados"] > 30)
    ]

    print(
        "\nRegistros fuera del intervalo 1-30 días:",
        len(outside_range),
    )


def audit_days_consistency(df: pd.DataFrame) -> None:
    """
    Compara dias_hasta_falla con la diferencia real entre fechas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con fechas convertidas.
    """

    failures = df[df["falla_futura_30d"] == 1].copy()

    failures["dias_calculados"] = (
        failures["fecha_falla_futura_dt"]
        - failures["fecha_mantenimiento_dt"]
    ).dt.days

    failures["diferencia"] = (
        failures["dias_hasta_falla"]
        - failures["dias_calculados"]
    )

    inconsistent = failures[
        failures["diferencia"] != 0
    ]

    print("\n" + "=" * 80)
    print("4. CONSISTENCIA DE dias_hasta_falla")
    print("=" * 80)

    print(
        "\nRegistros con diferencia entre "
        "dias_hasta_falla y días calculados:",
        len(inconsistent),
    )

    if not inconsistent.empty:
        print("\nPrimeras inconsistencias:")
        print(
            inconsistent[
                [
                    "fecha_mantenimiento",
                    "fecha_falla_futura",
                    "dias_hasta_falla",
                    "dias_calculados",
                    "diferencia",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )


def audit_non_failure_dates(df: pd.DataFrame) -> None:
    """
    Comprueba que los registros sin falla no tengan fecha de falla.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con fechas convertidas.
    """

    no_failure = df[df["falla_futura_30d"] == 0]

    dates_present = no_failure[
        no_failure["fecha_falla_futura_dt"].notna()
    ]

    days_present = no_failure[
        no_failure["dias_hasta_falla"].notna()
    ]

    print("\n" + "=" * 80)
    print("5. REGISTROS SIN FALLA")
    print("=" * 80)

    print(
        "\nCasos sin falla con fecha_falla_futura:",
        len(dates_present),
    )

    print(
        "Casos sin falla con dias_hasta_falla:",
        len(days_present),
    )


def main() -> None:
    """Ejecuta la auditoría temporal completa."""

    df = load_dataset()

    df = prepare_dates(df)

    audit_date_conversion(df)
    audit_failure_dates(df)
    audit_failure_interval(df)
    audit_days_consistency(df)
    audit_non_failure_dates(df)


if __name__ == "__main__":
    main()