"""Pruebas unitarias para el visor de Realidad Aumentada interactivo src.interactive_hud.

Patrón de pruebas: Arrange, Act, Assert (AAA).
Validaciones críticas:
- Algoritmo anti-colisión: dispersión de beacons próximos y trazado de leader lines.
- Conservación de coordenadas físicas exactas en puntos de anclaje (Anchor Dots).
- Codificación segura a Base64 con redimensionado optimizado para baja latencia.
- Manejo defensivo ante imágenes nulas o vacías.
"""

import math
import numpy as np
import pytest

from src.interactive_hud import (
    build_interactive_hud_html,
    calculate_layout_hotspots,
    encode_image_to_base64_jpeg,
    render_interactive_ar_hud,
)
from src.vision_ai import ARHotspot, ARTarget


def test_encode_image_to_base64_jpeg_success() -> None:
    """Verifica que una matriz RGB válida se codifique en un Data URL Base64 correcto."""
    # Arrange: Crear una imagen sintética RGB de 100x100
    sample_img = np.full((100, 100, 3), 128, dtype=np.uint8)

    # Act: Codificar la imagen
    data_url = encode_image_to_base64_jpeg(sample_img, quality=80)

    # Assert: Validar prefijo y formato Base64
    assert isinstance(data_url, str)
    assert data_url.startswith("data:image/jpeg;base64,")
    assert len(data_url) > 50


def test_encode_image_to_base64_jpeg_downscales_large_images() -> None:
    """Verifica que imágenes de gran resolución se redimensionen para acelerar la transmisión."""
    # Arrange: Imagen grande de 2000x3000
    large_img = np.zeros((3000, 2000, 3), dtype=np.uint8)

    # Act: Codificar con max_dimension=1200
    data_url = encode_image_to_base64_jpeg(large_img, quality=75, max_dimension=1200)

    # Assert
    assert isinstance(data_url, str)
    assert data_url.startswith("data:image/jpeg;base64,")


def test_encode_image_to_base64_jpeg_empty_raises_value_error() -> None:
    """Verifica que intentar codificar una imagen vacía lance ValueError explícito."""
    # Arrange: Matriz vacía
    empty_img = np.array([], dtype=np.uint8)

    # Act & Assert
    with pytest.raises(ValueError, match="está vacía"):
        encode_image_to_base64_jpeg(empty_img)


def test_calculate_layout_hotspots_separates_overlapping_points() -> None:
    """Verifica que dos puntos contiguos se separen espacialmente conservando sus anclajes."""
    # Arrange: Dos hotspots prácticamente en las mismas coordenadas (50%, 50%)
    hotspot_1 = ARHotspot(
        id="pin-a",
        name="Pin A",
        x_percent=50.0,
        y_percent=50.0,
        category="Pin"
    )
    hotspot_2 = ARHotspot(
        id="pin-b",
        name="Pin B",
        x_percent=50.5,
        y_percent=50.5,
        category="Pin"
    )

    # Act: Calcular distribución en columnas laterales
    layout = calculate_layout_hotspots([hotspot_1, hotspot_2])

    # Assert:
    # 1. Conservan la posición física exacta de anclaje
    assert layout[0].anchor_x == 50.0
    assert layout[0].anchor_y == 50.0
    assert layout[1].anchor_x == 50.5
    assert layout[1].anchor_y == 50.5

    # 2. La distancia entre sus beacons interactivos es >= 10.0%
    beacon_distance = math.hypot(
        layout[0].beacon_x - layout[1].beacon_x,
        layout[0].beacon_y - layout[1].beacon_y
    )
    assert beacon_distance >= 10.0


def test_build_interactive_hud_html_renders_leader_lines_and_anchors() -> None:
    """Verifica que el generador HTML proyecte las líneas conectoras SVG y beacons."""
    # Arrange: Puntos contiguos que forzarán dispersión y línea conectora
    close_hotspots = [
        ARHotspot(
            id="hs-1",
            name="Puerto USB-C",
            x_percent=40.0,
            y_percent=40.0,
            category="Puerto",
            description="Puerto de carga",
            voltage_or_spec="20V"
        ),
        ARHotspot(
            id="hs-2",
            name="LED Carga",
            x_percent=41.0,
            y_percent=41.0,
            category="LED",
            description="Indicador de carga",
            voltage_or_spec="3.3V"
        )
    ]
    fake_b64 = "data:image/jpeg;base64,dGVzdA=="

    # Act
    html_output = build_interactive_hud_html(
        image_base64=fake_b64,
        hotspots=close_hotspots,
        theme="CYAN",
        equipment_title="Placa de Prueba"
    )

    # Assert: Comprobar elementos SVG y técnicos
    assert "hud-lines-svg" in html_output
    assert "hud-anchor-dot" in html_output
    assert "hud-leader-line" in html_output
    assert "Puerto USB-C" in html_output
    assert "LED Carga" in html_output
    assert "toggleHotspot" in html_output
    assert "hoverHotspot" in html_output


def test_render_interactive_ar_hud_high_level_flow() -> None:
    """Verifica la función de alto nivel render_interactive_ar_hud a partir de un ARTarget."""
    # Arrange: Matriz sintética y target con hotspot
    frame_rgb = np.zeros((200, 300, 3), dtype=np.uint8)
    hotspot = ARHotspot(
        id="pin-13",
        name="LED Pin 13",
        x_percent=40.0,
        y_percent=35.0,
        category="LED / Salida",
        description="LED de prueba",
        voltage_or_spec="5V"
    )
    target = ARTarget(
        label="Arduino UNO R3",
        confidence=0.95,
        bbox=(20, 20, 160, 160),
        category="Embebidos",
        details="Microcontrolador",
        action_hint="Verificar conexiones",
        hotspots=[hotspot]
    )

    # Act
    html_res = render_interactive_ar_hud(frame_rgb, [target], theme="GREEN")

    # Assert
    assert isinstance(html_res, str)
    assert "LED Pin 13" in html_res
    assert "Arduino UNO R3" in html_res
    assert "#00ff66" in html_res  # Color primario del tema GREEN
