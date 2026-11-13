import io
import zipfile
from pathlib import Path
import pandas as pd

# Directorio base del proyecto (donde reside este archivo y el zip)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP_PATH = BASE_DIR / "bank+marketing.zip"

def cargar_csv_desde_zip(
    nombre_archivo: str,
    zip_path: Path | str = DEFAULT_ZIP_PATH,
    sep: str = ";",
    **kwargs
) -> pd.DataFrame:
    """
    Carga un archivo CSV contenido dentro de bank+marketing.zip
    (buscando tanto en la raíz como dentro de los sub-zips bank.zip o bank-additional.zip)
    o de forma local si ya se encuentra descomprimido.

    Parámetros:
    -----------
    nombre_archivo : str
        Nombre del archivo CSV (ej. 'bank-full.csv', 'bank.csv', 'bank-additional-full.csv').
    zip_path : Path | str
        Ruta al archivo zip principal (por defecto bank+marketing.zip en el proyecto).
    sep : str
        Separador de columnas (por defecto ';').
    **kwargs :
        Argumentos adicionales pasados directamente a pd.read_csv (ej. na_values).

    Retorna:
    --------
    pd.DataFrame
    """
    # 1. Si existe como archivo suelto localmente o en la carpeta del proyecto
    for candidato in [Path(nombre_archivo), BASE_DIR / nombre_archivo, BASE_DIR / "data" / nombre_archivo]:
        if candidato.is_file():
            return pd.read_csv(candidato, sep=sep, **kwargs)

    # 2. Localizar el archivo zip
    zip_file = Path(zip_path)
    if not zip_file.is_file():
        if (BASE_DIR / zip_file.name).is_file():
            zip_file = BASE_DIR / zip_file.name
        else:
            raise FileNotFoundError(f"No se encontró el archivo zip en: {zip_file.resolve()}")

    # 3. Leer directamente desde el zip en memoria
    with zipfile.ZipFile(zip_file, "r") as z_outer:
        # Búsqueda en la raíz del zip
        for member in z_outer.namelist():
            if Path(member).name == nombre_archivo:
                with z_outer.open(member) as f:
                    return pd.read_csv(f, sep=sep, **kwargs)

        # Búsqueda en sub-zips contenidos (bank.zip, bank-additional.zip)
        for member in z_outer.namelist():
            if member.endswith(".zip"):
                with z_outer.open(member) as sub_file:
                    with zipfile.ZipFile(io.BytesIO(sub_file.read())) as z_inner:
                        for inner_member in z_inner.namelist():
                            if Path(inner_member).name == nombre_archivo:
                                with z_inner.open(inner_member) as f:
                                    return pd.read_csv(f, sep=sep, **kwargs)

    raise FileNotFoundError(f"No se encontró '{nombre_archivo}' dentro de {zip_file}")
