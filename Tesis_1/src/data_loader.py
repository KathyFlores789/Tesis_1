"""
Módulo para carga y auditoría inicial del dataset de mantenimiento.

Este módulo no modifica el dataset original.
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/datos_mantenimiento_quito.csv")


def load_dataset(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Carga el dataset original desde un archivo CSV.

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
            f"No se encontró el archivo: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            "El dataset fue encontrado, pero no contiene registros."
        )

    return df


def audit_structure(df: pd.DataFrame) -> None:
    """
    Muestra una auditoría estructural del dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a auditar.
    """

    print("\n" + "=" * 80)
    print("AUDITORÍA ESTRUCTURAL")
    print("=" * 80)

    print(f"\nRegistros: {df.shape[0]}")
    print(f"Columnas:  {df.shape[1]}")

    print("\nTipos de datos:")
    print("-" * 80)

    for column in df.columns:
        print(f"{column:<40} {str(df[column].dtype)}")


def audit_missing_values(df: pd.DataFrame) -> None:
    """
    Analiza los valores nulos de cada variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a auditar.
    """

    print("\n" + "=" * 80)
    print("AUDITORÍA DE VALORES NULOS")
    print("=" * 80)

    missing = pd.DataFrame(
        {
            "nulos": df.isna().sum(),
            "porcentaje": (df.isna().mean() * 100).round(2),
        }
    )

    print(missing.to_string())


def audit_unique_values(df: pd.DataFrame) -> None:
    """
    Analiza la cantidad de valores únicos por variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a auditar.
    """

    print("\n" + "=" * 80)
    print("VALORES ÚNICOS")
    print("=" * 80)

    unique = pd.DataFrame(
        {
            "valores_unicos": df.nunique(dropna=False),
            "porcentaje_unicidad": (
                df.nunique(dropna=False) / len(df) * 100
            ).round(2),
        }
    )

    print(unique.to_string())


def audit_duplicates(df: pd.DataFrame) -> None:
    """
    Identifica registros completamente duplicados.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a auditar.
    """

    duplicated = int(df.duplicated().sum())

    print("\n" + "=" * 80)
    print("AUDITORÍA DE DUPLICADOS")
    print("=" * 80)

    print(f"\nRegistros completamente duplicados: {duplicated}")


def audit_numeric_ranges(df: pd.DataFrame) -> None:
    """
    Muestra estadísticas básicas de las variables numéricas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a auditar.
    """

    numeric_df = df.select_dtypes(include="number")

    print("\n" + "=" * 80)
    print("RANGOS Y ESTADÍSTICAS NUMÉRICAS")
    print("=" * 80)

    if numeric_df.empty:
        print("\nNo existen variables numéricas.")
        return

    statistics = numeric_df.describe().transpose()

    columns = [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]

    print(
        statistics[columns]
        .round(3)
        .to_string()
    )


def main() -> None:
    """Ejecuta la auditoría estructural inicial."""

    df = load_dataset()

    print("=" * 80)
    print("AUDITORÍA INICIAL DEL DATASET")
    print("=" * 80)

    print(f"\nArchivo: {DATA_PATH.resolve()}")

    audit_structure(df)
    audit_missing_values(df)
    audit_unique_values(df)
    audit_duplicates(df)
    audit_numeric_ranges(df)

# Alias para compatibilidad modular
cargar_datos_raw = load_dataset
if __name__ == "__main__":
    main()