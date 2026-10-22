"""Módulo de configuración centralizada para el Asistente Jarvis AR.

Propósito:
    Gestionar de forma segura las rutas del proyecto y variables de entorno,
    garantizando que ninguna credencial quede expuesta en el código fuente.

Ejemplo de uso:
    >>> from src.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.server_port)
    8501
"""

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Definición de rutas base usando exclusivamente pathlib.Path
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
ENV_FILE_PATH: Path = ROOT_DIR / ".env"

# Carga segura de variables de entorno desde el archivo .env si existe
if ENV_FILE_PATH.exists():
    load_dotenv(dotenv_path=ENV_FILE_PATH)
else:
    load_dotenv()


class AppSettings(BaseModel):
    """Modelo de validación de parámetros de configuración de la aplicación."""

    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Clave de API para el modelo multimodal Gemini (opcional)."
    )
    server_port: int = Field(
        default=8501,
        ge=1024,
        le=65535,
        description="Puerto para el servidor Streamlit."
    )
    server_address: str = Field(
        default="0.0.0.0",
        description="Dirección de escucha del servidor web."
    )
    camera_index: int = Field(
        default=0,
        ge=0,
        description="Índice de la cámara física para OpenCV."
    )
    ar_hud_theme: str = Field(
        default="CYAN",
        description="Tema visual para la superposición de Realidad Aumentada."
    )
    ar_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Umbral de confianza mínimo para renderizar elementos AR."
    )
    public_tunnel_url: Optional[str] = Field(
        default="https://projects-viruses-killing-issues.trycloudflare.com",
        description="URL pública HTTPS del túnel de acceso remoto."
    )
    root_dir: Path = Field(
        default=ROOT_DIR,
        description="Ruta raíz absoluta del proyecto."
    )


def get_settings() -> AppSettings:
    """Carga y valida la configuración global de la aplicación.

    Returns:
        AppSettings: Objeto inmutable con la configuración validada.

    Raises:
        ValueError: Si los valores de entorno violan las reglas de validación.
    """
    import os

    try:
        # Se extraen los valores del entorno respetando los valores por defecto
        raw_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        cleaned_key: Optional[str] = raw_key.strip() if raw_key and raw_key.strip() else None

        port_str: str = os.getenv("STREAMLIT_SERVER_PORT", "8501")
        camera_idx_str: str = os.getenv("CAMERA_INDEX", "0")
        conf_str: str = os.getenv("AR_CONFIDENCE_THRESHOLD", "0.5")
        tunnel_url: Optional[str] = os.getenv(
            "CLOUDFLARE_TUNNEL_URL",
            "https://projects-viruses-killing-issues.trycloudflare.com"
        )

        return AppSettings(
            gemini_api_key=cleaned_key,
            server_port=int(port_str),
            server_address=os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0"),
            camera_index=int(camera_idx_str),
            ar_hud_theme=os.getenv("AR_HUD_THEME", "CYAN"),
            ar_confidence_threshold=float(conf_str),
            public_tunnel_url=tunnel_url,
            root_dir=ROOT_DIR
        )
    except (ValueError, ValidationError) as err:
        # Se captura el error específico de formato numérico o de Pydantic
        raise ValueError(f"Error al validar las variables de entorno: {err}") from err
