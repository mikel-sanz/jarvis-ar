"""Pruebas unitarias para el motor de Realidad Aumentada src.ar_engine.

Patrón de pruebas: Arrange, Act, Assert (AAA).
Validaciones críticas:
- Preservación de la forma dimensional y tipo de datos de la matriz de imagen.
- Inyección efectiva de píxeles gráficos (la imagen resultante debe modificarse con el overlay).
- Manejo correcto de listas vacías de objetivos (solo barra y esquinas HUD).
- Compatibilidad de múltiples paletas cromáticas sin excepciones.
"""

import numpy as np
from src.ar_engine import AREngine, COLOR_PALETTES
from src.vision_ai import ARTarget


def test_ar_engine_dimensions_and_channels() -> None:
    """Valida que el motor AR mantenga exactamente las dimensiones y canales del fotograma."""
    # Arrange: Crear un frame sintético negro de 480x640x3
    height, width = 480, 640
    frame_base = np.zeros((height, width, 3), dtype=np.uint8)
    engine = AREngine(theme="CYAN")
    target = ARTarget(
        label="Test Object",
        confidence=0.88,
        bbox=(50, 50, 100, 100),
        category="Hardware",
        details="Microcontrolador STM32",
        action_hint="Verificar alimentación"
    )

    # Act: Renderizar la capa de Realidad Aumentada
    rendered = engine.render_overlay(frame_base, [target], telemetry_text="TEST MODE")

    # Assert: Validar dimensiones idénticas y que se hayan modificado píxeles
    assert rendered.shape == (height, width, 3)
    assert rendered.dtype == np.uint8
    assert not np.array_equal(frame_base, rendered), "El overlay debe pintar gráficos en el frame."


def test_ar_engine_empty_targets() -> None:
    """Valida que renderizar sin objetivos mantenga la estabilidad del visor HUD."""
    # Arrange: Frame sintético y lista vacía de objetivos
    frame_base = np.zeros((300, 400, 3), dtype=np.uint8)
    engine = AREngine(theme="GREEN")

    # Act: Renderizar sin objetivos detectados
    rendered = engine.render_overlay(frame_base, targets=[], telemetry_text="IDLE")

    # Assert: Debe procesarse sin errores y contener la barra superior
    assert rendered.shape == (300, 400, 3)
    # Comprobar que la barra superior se dibujó (píxeles no negros en la fila 10)
    assert np.any(rendered[10, :, :] > 0)


def test_ar_engine_supported_palettes() -> None:
    """Verifica que todas las paletas definidas en COLOR_PALETTES se inicialicen correctamente."""
    # Arrange: Lista de temas soportados
    themes = ["CYAN", "GREEN", "ORANGE", "TEMA_INEXISTENTE_FALLBACK"]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Act & Assert: Cada tema debe inicializarse y renderizar sin lanzar excepciones
    for theme in themes:
        engine = AREngine(theme=theme)
        output = engine.render_overlay(frame, targets=[])
        assert output.shape == frame.shape
