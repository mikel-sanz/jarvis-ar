"""Pruebas unitarias para el motor de manuales técnicos src.manual_engine.

Patrón de pruebas: Arrange, Act, Assert (AAA).
Validaciones críticas:
- Obtención estructurada de manuales desde el catálogo técnico integrado.
- Persistencia y carga correcta desde la caché local en JSON.
- Respuesta coherente a consultas técnicas en lenguaje natural sobre especificaciones y seguridad.
"""

from pathlib import Path
from src.manual_engine import CACHE_MANUALS_DIR, DeviceManual, ManualEngine, sanitize_filename


def test_get_manual_from_catalog() -> None:
    """Valida la recuperación de una ficha técnica completa desde el catálogo offline."""
    # Arrange: Inicializar el motor sin clave externa
    engine = ManualEngine(api_key=None)

    # Act: Solicitar el manual técnico de Arduino UNO R3
    manual = engine.get_manual("Arduino", "UNO R3")

    # Assert: Comprobar la integridad de los datos técnicos
    assert isinstance(manual, DeviceManual)
    assert manual.brand == "Arduino"
    assert "ATmega328P" in manual.specifications.get("Microcontrolador", "")
    assert len(manual.safety_warnings) > 0
    assert len(manual.quick_start_steps) > 0


def test_manual_caching_creates_json_file() -> None:
    """Verifica que el manual técnico se guarde en disco en formato JSON en data/cache_manuals/."""
    # Arrange: Definir equipo y ruta esperada de caché
    brand = "Cisco"
    model = "Catalyst 2960"
    expected_cache_file: Path = CACHE_MANUALS_DIR / f"{sanitize_filename(f'{brand}_{model}')}.json"

    # Si existía previamente, eliminar para verificar creación limpia
    if expected_cache_file.exists():
        expected_cache_file.unlink()

    engine = ManualEngine(api_key=None)

    # Act: Obtener el manual para forzar el guardado en caché
    manual = engine.get_manual(brand, model)

    # Assert: El archivo JSON debe haberse creado y ser legible
    assert expected_cache_file.exists(), "El manual debe persistirse en la carpeta de caché."
    assert manual.brand == "Cisco Systems"


def test_query_manual_answers_voltage_question() -> None:
    """Valida que el motor responda consultas técnicas sobre voltaje de forma precisa."""
    # Arrange: Obtener el manual de un multímetro Fluke
    engine = ManualEngine(api_key=None)
    manual = engine.get_manual("Fluke", "87V Industrial")

    # Act: Realizar una pregunta sobre voltaje
    response = engine.query_manual(manual, "¿Cuál es el voltaje máximo que soporta?")

    # Assert: La respuesta debe contener información técnica sobre voltaje
    assert isinstance(response, str)
    assert len(response) > 20
    assert "voltaje" in response.lower() or "1000v" in response.lower()


def test_get_manual_food_supplement_has_nutritional_specs_and_no_voltage() -> None:
    """Verifica que suplementos o alimentos generen fichas nutricionales sin voltajes ni cables."""
    # Arrange
    engine = ManualEngine(api_key=None)

    # Act
    manual = engine.get_manual(
        brand="BIOPROX",
        model_number="3 Protein Bar - Chocolate Dubai",
        category="Nutrición / Suplementación",
        details="Barrita proteica sabor chocolate"
    )

    # Assert
    assert isinstance(manual, DeviceManual)
    assert "Nutrición" in manual.category or "Suplementación" in manual.category
    assert "Aporte Proteico" in manual.specifications or "Tipo de Producto" in manual.specifications
    # Validación crítica: no debe contener especificaciones eléctricas
    for k in manual.specifications.keys():
        assert "voltaje" not in k.lower()
        assert "alimentación estándar" not in k.lower()
        assert "interfaz" not in k.lower()
    # Advertencias de alérgenos presentes
    warnings_text = " ".join(manual.safety_warnings).lower()
    assert "alérgenos" in warnings_text or "leche" in warnings_text or "soja" in warnings_text
