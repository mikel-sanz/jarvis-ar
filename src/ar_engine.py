"""Motor de renderizado de Realidad Aumentada (AR Engine).

Propósito:
    Superponer elementos gráficos futuristas de Realidad Aumentada (HUD)
    sobre fotogramas en tiempo real: retículas, corchetes de esquinas,
    paneles translúcidos con telemetría y tarjetas holográficas flotantes.

Ejemplo de uso:
    >>> import numpy as np
    >>> from src.ar_engine import AREngine
    >>> from src.vision_ai import ARTarget
    >>> engine = AREngine()
    >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
    >>> target = ARTarget(label="Test", confidence=0.9, bbox=(100, 100, 50, 50), category="Demo", details="Dato", action_hint="Paso 1")
    >>> rendered = engine.render_overlay(frame, [target], telemetry_text="JARVIS ONLINE")
    >>> print(rendered.shape)
    (480, 640, 3)
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np
from src.vision_ai import ARTarget


# Paletas cromáticas estilo HUD holográfico en formato BGR (OpenCV)
COLOR_PALETTES: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "CYAN": {
        "primary": (255, 220, 0),      # Cian eléctrico
        "secondary": (255, 150, 0),    # Azul cielo
        "accent": (0, 255, 255),       # Amarillo neón
        "bg_overlay": (20, 25, 30),    # Fondo oscuro
        "text": (255, 255, 255),       # Blanco
    },
    "GREEN": {
        "primary": (0, 255, 120),      # Verde matrix
        "secondary": (0, 200, 80),     # Verde esmeralda
        "accent": (0, 255, 255),       # Amarillo
        "bg_overlay": (10, 30, 15),
        "text": (240, 255, 240),
    },
    "ORANGE": {
        "primary": (0, 140, 255),      # Naranja táctico
        "secondary": (0, 90, 220),     # Ámbar
        "accent": (255, 255, 0),       # Cian de contraste
        "bg_overlay": (30, 20, 10),
        "text": (255, 250, 245),
    }
}


class AREngine:
    """Motor de superposición gráfica para Realidad Aumentada."""

    def __init__(self, theme: str = "CYAN") -> None:
        """Inicializa el motor AR con un tema de color específico.

        Args:
            theme: Nombre del tema ("CYAN", "GREEN", "ORANGE").
        """
        self._palette = COLOR_PALETTES.get(theme.upper(), COLOR_PALETTES["CYAN"])

    def render_overlay(
        self,
        frame_bgr: np.ndarray,
        targets: List[ARTarget],
        telemetry_text: str = "JARVIS AR // ACTIVO",
        show_hud_grid: bool = True
    ) -> np.ndarray:
        """Dibuja los elementos de Realidad Aumentada sobre el fotograma.

        Args:
            frame_bgr: Fotograma base de la cámara en formato BGR.
            targets: Lista de objetivos identificados por el motor de IA.
            telemetry_text: Texto para la barra superior de telemetría.
            show_hud_grid: Si es True, proyecta retículas de encuadre en las esquinas.

        Returns:
            np.ndarray: Nuevo fotograma con la capa AR superpuesta.
        """
        output_frame = frame_bgr.copy()
        h, w = output_frame.shape[:2]

        # 1. Proyectar retículas HUD de marco periférico
        if show_hud_grid:
            self._draw_screen_corners(output_frame, w, h)

        # 2. Barra de telemetría superior
        self._draw_telemetry_bar(output_frame, w, telemetry_text, len(targets))

        # 3. Renderizar cada tarjeta de objetivo AR detectado
        for target in targets:
            self._draw_ar_target(output_frame, target, w, h)

        return output_frame

    def _draw_screen_corners(self, frame: np.ndarray, w: int, h: int) -> None:
        """Dibuja corchetes en las 4 esquinas de la pantalla para estética de visor AR."""
        c = self._palette["secondary"]
        length = 35
        thickness = 2
        margin = 15

        # Esquina Superior Izquierda
        cv2.line(frame, (margin, margin), (margin + length, margin), c, thickness)
        cv2.line(frame, (margin, margin), (margin, margin + length), c, thickness)

        # Esquina Superior Derecha
        cv2.line(frame, (w - margin, margin), (w - margin - length, margin), c, thickness)
        cv2.line(frame, (w - margin, margin), (w - margin, margin + length), c, thickness)

        # Esquina Inferior Izquierda
        cv2.line(frame, (margin, h - margin), (margin + length, h - margin), c, thickness)
        cv2.line(frame, (margin, h - margin), (margin, h - margin - length), c, thickness)

        # Esquina Inferior Derecha
        cv2.line(frame, (w - margin, h - margin), (w - margin - length, h - margin), c, thickness)
        cv2.line(frame, (w - margin, h - margin), (w - margin, h - margin - length), c, thickness)

    def _draw_telemetry_bar(self, frame: np.ndarray, w: int, text: str, target_count: int) -> None:
        """Renderiza una barra superior translúcida con telemetría de estado."""
        bar_height = 36
        overlay = frame.copy()

        # Fondo translúcido superior
        cv2.rectangle(overlay, (0, 0), (w, bar_height), self._palette["bg_overlay"], -1)
        # Línea de brillo inferior
        cv2.line(overlay, (0, bar_height), (w, bar_height), self._palette["primary"], 1)

        # Fusión alfa para dar efecto de vidrio / holograma
        alpha = 0.65
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Texto de telemetría y conteo de objetivos
        telemetry_info = f"{text} | OBJETIVOS: {target_count}"
        cv2.putText(
            frame,
            telemetry_info,
            (20, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self._palette["primary"],
            1,
            cv2.LINE_AA
        )

    def _draw_ar_target(self, frame: np.ndarray, target: ARTarget, screen_w: int, screen_h: int) -> None:
        """Dibuja la caja de seguimiento y la tarjeta flotante para un objetivo específico."""
        x, y, bw, bh = target.bbox
        primary = self._palette["primary"]
        accent = self._palette["accent"]
        bg = self._palette["bg_overlay"]
        txt_color = self._palette["text"]

        # 1. Dibujar corchetes en las esquinas del Bounding Box
        corner_len = max(8, min(bw, bh) // 4)
        thickness = 2

        # Superior Izquierda
        cv2.line(frame, (x, y), (x + corner_len, y), primary, thickness)
        cv2.line(frame, (x, y), (x, y + corner_len), primary, thickness)
        # Superior Derecha
        cv2.line(frame, (x + bw, y), (x + bw - corner_len, y), primary, thickness)
        cv2.line(frame, (x + bw, y), (x + bw, y + corner_len), primary, thickness)
        # Inferior Izquierda
        cv2.line(frame, (x, y + bh), (x + corner_len, y + bh), primary, thickness)
        cv2.line(frame, (x, y + bh), (x, y + bh - corner_len), primary, thickness)
        # Inferior Derecha
        cv2.line(frame, (x + bw, y + bh), (x + bw - corner_len, y + bh), primary, thickness)
        cv2.line(frame, (x + bw, y + bh), (x + bw, y + bh - corner_len), primary, thickness)

        # 2. Retícula central en el objeto
        cx, cy = x + bw // 2, y + bh // 2
        cv2.drawMarker(frame, (cx, cy), accent, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)

        # 3. Cálculo de posición de la Tarjeta Flotante AR (evitar desbordes de pantalla)
        card_w = 300
        card_h = 105

        card_x = x + bw + 15
        if card_x + card_w > screen_w - 10:
            # Si no cabe a la derecha, colocarla a la izquierda o encima
            card_x = max(10, x - card_w - 15)

        card_y = max(40, min(y, screen_h - card_h - 15))

        # Línea de anclaje (conector holográfico entre objeto y tarjeta)
        cv2.line(frame, (x + bw, y + 10), (card_x, card_y + 15), accent, 1, cv2.LINE_AA)
        cv2.circle(frame, (card_x, card_y + 15), 3, accent, -1)

        # 4. Renderizado del fondo de la tarjeta con transparencia
        card_overlay = frame.copy()
        cv2.rectangle(card_overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), bg, -1)
        cv2.rectangle(card_overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), primary, 1)

        # Fusión alfa
        alpha = 0.75
        cv2.addWeighted(card_overlay, alpha, frame, 1 - alpha, 0, frame)

        # 5. Textos informativos de alta resolución de la tarjeta
        # Línea 1: Marca y Modelo con porcentaje de confianza
        if target.brand and target.model_number:
            header_str = f"[{target.brand.upper()}] {target.model_number[:16]}"
        else:
            header_str = f"{target.label[:20]}"
        header_text = f"{header_str} [{int(target.confidence * 100)}%]"
        cv2.putText(
            frame,
            header_text,
            (card_x + 10, card_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            primary,
            1,
            cv2.LINE_AA
        )

        # Línea 2: Indicador de manual oficial vinculado
        manual_status = "[📖 MANUAL OFICIAL VINCULADO]" if target.manual else f"CAT: {target.category[:22]}"
        cv2.putText(
            frame,
            manual_status,
            (card_x + 10, card_y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            accent,
            1,
            cv2.LINE_AA
        )

        # Línea 3: Especificaciones técnicas clave
        details_truncated = target.details[:38] + ("..." if len(target.details) > 38 else "")
        cv2.putText(
            frame,
            details_truncated,
            (card_x + 10, card_y + 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            txt_color,
            1,
            cv2.LINE_AA
        )

        # Línea 4: Código serial / Identificador visible
        serial_str = f"SERIAL: {target.serial_number}" if target.serial_number else f"TIPO: {target.category[:26]}"
        cv2.putText(
            frame,
            serial_str[:38],
            (card_x + 10, card_y + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        # Línea 5: Acción guiada
        hint_truncated = f"> {target.action_hint[:36]}"
        cv2.putText(
            frame,
            hint_truncated,
            (card_x + 10, card_y + 93),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 240, 255),
            1,
            cv2.LINE_AA
        )
