"""Pruebas unitarias para el módulo de configuración src.config.

Patrón de pruebas: Arrange, Act, Assert (AAA).
Validaciones críticas:
- Carga por defecto de parámetros cuando no hay variables personalizadas.
- Conversión adecuada de tipos de datos (puertos numéricos, float de umbral).
- Validación defensiva de límites en puertos con Pydantic.
"""

from pathlib import Path
import pytest
from pydantic import ValidationError
from src.config import AppSettings, get_settings


def test_default_settings_loading() -> None:
    """Verifica que la configuración por defecto se cargue con valores válidos."""
    # Arrange: No se alteran variables de entorno, se usa el estado base.

    # Act: Se invoca la función de carga de configuración.
    settings = get_settings()

    # Assert: Se valida la integridad de los tipos y rangos por defecto.
    assert isinstance(settings, AppSettings)
    assert 1024 <= settings.server_port <= 65535
    assert 0.0 <= settings.ar_confidence_threshold <= 1.0
    assert isinstance(settings.root_dir, Path)
    assert settings.root_dir.exists()


def test_app_settings_validation_invalid_port() -> None:
    """Valida que un puerto fuera de rango genere un error de validación (Defensivo)."""
    # Arrange: Definir un puerto inválido (menor a 1024 o mayor a 65535).
    invalid_port: int = 99999

    # Act & Assert: Comprobar que Pydantic rechaza la instanciación con ValidationError.
    with pytest.raises(ValidationError):
        AppSettings(server_port=invalid_port)


def test_app_settings_confidence_bounds() -> None:
    """Verifica que el umbral de confianza AR no admita valores fuera de [0.0, 1.0]."""
    # Arrange: Umbral fuera de escala.
    invalid_confidence: float = 1.5

    # Act & Assert: Se espera rechazo estricto por validación de rango.
    with pytest.raises(ValidationError):
        AppSettings(ar_confidence_threshold=invalid_confidence)
