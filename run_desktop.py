"""Visor de escritorio nativo de Realidad Aumentada (Jarvis AR Desktop).

Propósito:
    Ejecutar un visor en ventana nativa de alta velocidad (OpenCV) utilizando
    directamente el hardware de la webcam del PC a 30 FPS, eliminando cualquier
    restricción de permisos o contextos seguros de navegadores web.

Controles:
    - [Espacio]: Ejecutar re-escaneo de inteligencia artificial.
    - [C]: Alternar paleta de colores del HUD AR (CYAN, GREEN, ORANGE).
    - [Q] o [ESC]: Cerrar visor.

Uso:
    $ python run_desktop.py
"""

import sys
import time
from typing import List
import cv2
import numpy as np

from src.ar_engine import AREngine, COLOR_PALETTES
from src.camera import CameraManager
from src.config import AppSettings, get_settings
from src.vision_ai import ARTarget, SceneAnalysisResult, VisionEngine


def main() -> None:
    """Bucle principal de renderizado continuo en tiempo real."""
    settings: AppSettings = get_settings()
    print("=" * 60)
    print("🔮 INICIANDO JARVIS AR (VISOR DE ESCRITORIO NATIVO)")
    print("=" * 60)
    print("Controles:")
    print("  [Espacio] -> Analizar escena con Inteligencia Artificial")
    print("  [C]       -> Cambiar paleta de colores del HUD")
    print("  [Q / ESC] -> Salir")
    print("=" * 60)

    themes: List[str] = list(COLOR_PALETTES.keys())
    theme_idx: int = 0

    vision_engine = VisionEngine(api_key=settings.gemini_api_key)
    ar_engine = AREngine(theme=themes[theme_idx])

    current_targets: List[ARTarget] = []
    telemetry_status: str = "JARVIS DESKTOP // ONLINE"

    with CameraManager(camera_index=settings.camera_index) as cam:
        if not cam.is_opened():
            print(f"❌ Error: No se pudo abrir la cámara física con índice {settings.camera_index}.")
            sys.exit(1)

        window_name = "JARVIS AR - Spatial Intelligence HUD"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 600)

        # Análisis inicial de arranque
        success, initial_frame = cam.read_frame()
        if success and initial_frame is not None:
            analysis = vision_engine.analyze_frame(initial_frame)
            current_targets = analysis.targets
            telemetry_status = f"JARVIS // {analysis.engine_used.upper()}"

        prev_time: float = time.time()

        while True:
            success, frame = cam.read_frame()
            if not success or frame is None:
                print("Advertencia: No se pudo leer el fotograma de la cámara.")
                time.sleep(0.03)
                continue

            # Cálculo de frames por segundo (FPS) en tiempo real
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
            prev_time = curr_time

            hud_text = f"{telemetry_status} | FPS: {int(fps)} | TEMA: {themes[theme_idx]}"

            # Superposición gráfica de Realidad Aumentada
            rendered_frame = ar_engine.render_overlay(
                frame_bgr=frame,
                targets=current_targets,
                telemetry_text=hud_text,
                show_hud_grid=True
            )

            cv2.imshow(window_name, rendered_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                break
            elif key == ord(" "):  # Barra espaciadora para nuevo escaneo IA
                print("Escaneando escena con IA...")
                analysis = vision_engine.analyze_frame(frame)
                current_targets = analysis.targets
                telemetry_status = f"JARVIS // {analysis.engine_used.upper()}"
                print(f"Resumen: {analysis.summary} ({len(current_targets)} objetos detectados)")
            elif key in (ord("c"), ord("C")):  # Alternar tema
                theme_idx = (theme_idx + 1) % len(themes)
                ar_engine = AREngine(theme=themes[theme_idx])
                print(f"Tema AR cambiado a: {themes[theme_idx]}")

    cv2.destroyAllWindows()
    print("Visor de escritorio finalizado con éxito.")


if __name__ == "__main__":
    main()
