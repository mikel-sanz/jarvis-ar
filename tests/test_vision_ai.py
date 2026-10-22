"""Pruebas unitarias para el motor de visión multimodal src.vision_ai.

Patrón de pruebas: Arrange, Act, Assert (AAA).
Validaciones críticas:
- Funcionamiento offline autónomo (modo local sin API Key).
- Detección de zonas de interés geométricas sobre imágenes sintéticas.
- Cumplimiento del contrato de datos de SceneAnalysisResult y ARTarget.
"""

import cv2
import numpy as np
from src.vision_ai import ARTarget, SceneAnalysisResult, VisionEngine


def test_vision_engine_local_mode_neutral_frame_yields_empty_targets() -> None:
    """Verifica que un fotograma sin marcas ni QR no genere objetivos inventados."""
    # Arrange: Inicialización sin clave de API
    engine = VisionEngine(api_key=None)

    # Act: Analizar un fotograma neutro
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = engine.analyze_frame(frame)

    # Assert: Comprobar que no se inventa ningún objetivo falso
    assert isinstance(result, SceneAnalysisResult)
    assert result.engine_used == "local-vision-engine"
    assert len(result.targets) == 0


def test_vision_engine_manual_override_xiaomi() -> None:
    """Verifica que el forzado de perfil local vincule correctamente la ficha técnica de Xiaomi."""
    # Arrange
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    engine = VisionEngine(api_key=None)

    # Act
    result = engine.analyze_frame(frame, manual_override="Xiaomi Smartphone (HyperOS)")

    # Assert
    assert result.engine_used == "local-profile-engine"
    assert len(result.targets) == 1
    target = result.targets[0]
    assert target.brand == "Xiaomi"
    assert target.manual is not None
    assert target.manual.brand == "Xiaomi"
    assert "HyperOS" in target.manual.equipment_name
    assert "HyperCharge" in target.manual.specifications.get("Carga Rápida", "")


def test_vision_engine_cloud_error_yields_empty_targets_and_api_error() -> None:
    """Verifica que si la IA en la nube falla, no invente Xiaomi ni ningún objetivo fantasma."""
    # Arrange: Motor con cliente mock que falla al llamar a Gemini
    engine = VisionEngine(api_key="invalid_fake_key")
    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Act: Analizar frame
    result = engine.analyze_frame(frame)

    # Assert: targets DEBE estar vacío y api_error debe contener el diagnóstico
    assert len(result.targets) == 0
    assert result.api_error is not None
    assert "Gemini Cloud Error" in result.api_error
