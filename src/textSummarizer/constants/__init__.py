from pathlib import Path

# Obtener la ruta absoluta del directorio raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

CONFIG_FILE_PATH = ROOT_DIR / "config" / "config.yaml"
PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"