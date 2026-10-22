"""Módulo de renderizado de visor de Realidad Aumentada interactivo con Hotspots táctiles y arrastrables.

Propósito:
    Generar un componente visual interactivo en HTML5, CSS3 y JavaScript vanilla
    que proyecta puntos de interés interactivos (hotspots) sobre el fotograma capturado.
    Organiza los botones en columnas laterales para no obstruir el equipo y traza
    líneas técnicas (Leader Lines) directas al punto de anclaje físico.
    Permite arrastrar libremente los hotspots con el dedo (pantalla táctil) o ratón,
    actualizando las líneas conectoras en tiempo real para máxima visibilidad.

Ejemplo de uso:
    >>> from src.interactive_hud import render_interactive_ar_hud
    >>> html_content = render_interactive_ar_hud(image_rgb, targets, theme="CYAN")
    >>> streamlit.components.v1.html(html_content, height=650)
"""

import base64
from dataclasses import dataclass
import html
from typing import List, Tuple
import cv2
import numpy as np

from src.vision_ai import ARHotspot, ARTarget


# Definición de paletas cromáticas holográficas HUD
HUD_COLORS = {
    "CYAN": {
        "primary": "#00f0ff",
        "glow": "rgba(0, 240, 255, 0.4)",
        "card_bg": "rgba(6, 18, 36, 0.94)",
        "border": "#00f0ff",
        "accent": "#00ffee"
    },
    "GREEN": {
        "primary": "#00ff66",
        "glow": "rgba(0, 255, 102, 0.4)",
        "card_bg": "rgba(6, 30, 18, 0.94)",
        "border": "#00ff66",
        "accent": "#66ff99"
    },
    "AMBER": {
        "primary": "#ffaa00",
        "glow": "rgba(255, 170, 0, 0.4)",
        "card_bg": "rgba(36, 24, 6, 0.94)",
        "border": "#ffaa00",
        "accent": "#ffcc44"
    },
    "RED": {
        "primary": "#ff3366",
        "glow": "rgba(255, 51, 102, 0.4)",
        "card_bg": "rgba(36, 6, 12, 0.94)",
        "border": "#ff3366",
        "accent": "#ff6688"
    }
}


@dataclass
class DisplacedHotspot:
    """Estructura de coordenadas calculadas para evitar solapamientos visuales."""

    hotspot: ARHotspot
    index: int
    anchor_x: float  # Posición física exacta en el componente (%)
    anchor_y: float  # Posición física exacta en el componente (%)
    beacon_x: float  # Posición del botón táctil arrastrable (%)
    beacon_y: float  # Posición del botón táctil arrastrable (%)


def calculate_layout_hotspots(
    hotspots: List[ARHotspot]
) -> List[DisplacedHotspot]:
    """Distribuye los beacons en columnas laterales ordenadas con líneas de llamada.

    Separa los puntos entre el lateral izquierdo (x=8%) y derecho (x=92%) según su
    posición horizontal, ordenándolos verticalmente con espaciado garantizado para
    que NUNCA se solapen y dejando el área central del equipo completamente despejada.

    Args:
        hotspots: Lista de puntos interactivos originales.

    Returns:
        List[DisplacedHotspot]: Lista con coordenadas de anclaje y de botón separadas.
    """
    if not hotspots:
        return []

    # Clasificar en lateral izquierdo y derecho según anchor_x
    left_group: List[Tuple[int, ARHotspot]] = []
    right_group: List[Tuple[int, ARHotspot]] = []

    for idx, hs in enumerate(hotspots, start=1):
        if hs.x_percent < 50.0:
            left_group.append((idx, hs))
        else:
            right_group.append((idx, hs))

    # Si un grupo tiene todos y el otro ninguno, equilibrar para estética y distribución
    if len(left_group) == 0 and len(right_group) > 3:
        sorted_by_x = sorted(right_group, key=lambda item: item[1].x_percent)
        half = len(sorted_by_x) // 2
        left_group = sorted_by_x[:half]
        right_group = sorted_by_x[half:]
    elif len(right_group) == 0 and len(left_group) > 3:
        sorted_by_x = sorted(left_group, key=lambda item: item[1].x_percent)
        half = len(sorted_by_x) // 2
        right_group = sorted_by_x[half:]
        left_group = sorted_by_x[:half]

    # Ordenar cada grupo por su coordenada vertical anchor_y para evitar cruces innecesarios de líneas
    left_group.sort(key=lambda item: item[1].y_percent)
    right_group.sort(key=lambda item: item[1].y_percent)

    placed: List[DisplacedHotspot] = []

    # Asignar posiciones verticales regulares en la columna izquierda (x = 8.0%)
    if left_group:
        n_left = len(left_group)
        for i, (idx, hs) in enumerate(left_group):
            beacon_y = 15.0 + (i * (70.0 / max(1, n_left - 1))) if n_left > 1 else 50.0
            placed.append(DisplacedHotspot(
                hotspot=hs,
                index=idx,
                anchor_x=float(hs.x_percent),
                anchor_y=float(hs.y_percent),
                beacon_x=8.0,
                beacon_y=beacon_y
            ))

    # Asignar posiciones verticales regulares en la columna derecha (x = 92.0%)
    if right_group:
        n_right = len(right_group)
        for i, (idx, hs) in enumerate(right_group):
            beacon_y = 15.0 + (i * (70.0 / max(1, n_right - 1))) if n_right > 1 else 50.0
            placed.append(DisplacedHotspot(
                hotspot=hs,
                index=idx,
                anchor_x=float(hs.x_percent),
                anchor_y=float(hs.y_percent),
                beacon_x=92.0,
                beacon_y=beacon_y
            ))

    # Reordenar por índice original para consistencia
    placed.sort(key=lambda item: item.index)
    return placed


def encode_image_to_base64_jpeg(
    image_rgb: np.ndarray,
    quality: int = 80,
    max_dimension: int = 1200
) -> str:
    """Codifica una imagen NumPy RGB a Base64 optimizando resolución y calidad para alta velocidad.

    Args:
        image_rgb: Matriz NumPy con formato de color RGB.
        quality: Calidad de compresión JPEG (1 a 100).
        max_dimension: Dimensión máxima de ancho o alto para evitar latencias de red.

    Returns:
        str: Cadena en formato data URL Base64 ('data:image/jpeg;base64,...').

    Raises:
        ValueError: Si la imagen está vacía o es inválida.
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("La imagen proporcionada para codificación Base64 está vacía.")

    h, w = image_rgb.shape[:2]
    # Redimensionado proporcional si excede max_dimension para acelerar renderizado
    if max(h, w) > max_dimension:
        scale = max_dimension / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image_proc = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        image_proc = image_rgb

    image_bgr = cv2.cvtColor(image_proc, cv2.COLOR_RGB2BGR)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(100, quality))]
    success, buffer = cv2.imencode(".jpg", image_bgr, encode_params)

    if not success or buffer is None:
        raise ValueError("Error al codificar el fotograma en formato JPEG.")

    b64_str = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{b64_str}"


def build_interactive_hud_html(
    image_base64: str,
    hotspots: List[ARHotspot],
    theme: str = "CYAN",
    equipment_title: str = "Dispositivo Inspeccionado"
) -> str:
    """Construye el código HTML5/CSS3/JS con columnas laterales ordenadas y drag & drop libre.

    Args:
        image_base64: Imagen de fondo en formato Data URL Base64.
        hotspots: Lista de puntos interactivos a posicionar.
        theme: Clave del tema cromático ('CYAN', 'GREEN', 'AMBER', 'RED').
        equipment_title: Título descriptivo para la barra de estado superior.

    Returns:
        str: Código HTML listo para ser inyectado en Streamlit vía components.html.
    """
    theme_cfg = HUD_COLORS.get(theme.upper(), HUD_COLORS["CYAN"])
    p_color = theme_cfg["primary"]
    p_glow = theme_cfg["glow"]
    p_bg = theme_cfg["card_bg"]
    p_border = theme_cfg["border"]

    safe_title = html.escape(equipment_title)

    # 1. Distribuir en columnas laterales libres de solapamiento
    layout_items = calculate_layout_hotspots(hotspots)

    svg_lines_list: List[str] = []
    hotspots_html_list: List[str] = []
    cards_html_list: List[str] = []

    for item in layout_items:
        hs = item.hotspot
        hs_id = html.escape(hs.id)
        hs_name = html.escape(hs.name)
        hs_cat = html.escape(hs.category)
        hs_desc = html.escape(hs.description)
        hs_spec = html.escape(hs.voltage_or_spec or "N/D")
        hs_action = html.escape(hs.action_guidance or "Verificar conexión según manual técnico.")

        ax, ay = item.anchor_x, item.anchor_y
        bx, by = item.beacon_x, item.beacon_y

        # Elementos SVG: Punto exacto (Anchor) y Línea de llamada (Leader Line)
        svg_lines_list.append(
            f"""
            <g class="hud-leader-group" id="leader-{hs_id}">
                <circle class="hud-anchor-dot" cx="{ax:.2f}" cy="{ay:.2f}" r="0.9" />
                <circle class="hud-anchor-pulse" cx="{ax:.2f}" cy="{ay:.2f}" r="1.8" />
                <line class="hud-leader-line"
                      id="line-{hs_id}"
                      x1="{ax:.2f}" y1="{ay:.2f}"
                      x2="{bx:.2f}" y2="{by:.2f}" />
            </g>
            """
        )

        # Pin circular con radar pulsante situado en columna lateral, con soporte de arrastre
        hotspots_html_list.append(
            f"""
            <div class="hud-hotspot-beacon"
                 id="beacon-{hs_id}"
                 data-hsid="{hs_id}"
                 style="left: {bx:.2f}%; top: {by:.2f}%;"
                 onclick="handleBeaconClick('{hs_id}')"
                 onmouseenter="hoverHotspot('{hs_id}', true)"
                 onmouseleave="hoverHotspot('{hs_id}', false)"
                 title="{hs_name} (Arrastra para mover)">
                <div class="radar-pulse"></div>
                <div class="beacon-core">{item.index}</div>
                <div class="beacon-tag">{hs_name}</div>
            </div>
            """
        )

        # Tarjeta de inspección emergente
        cards_html_list.append(
            f"""
            <div class="hud-inspect-card" id="card-{hs_id}">
                <div class="card-header">
                    <div class="card-badge">HOTSPOT #{item.index} // {hs_cat.upper()}</div>
                    <button class="card-close" onclick="closeAllCards()">✕</button>
                </div>
                <h4 class="card-title">{hs_name}</h4>
                <div class="card-spec-row">
                    <span class="spec-label">⚡ ESPECIFICACIÓN:</span>
                    <span class="spec-value">{hs_spec}</span>
                </div>
                <p class="card-desc">{hs_desc}</p>
                <div class="card-guidance">
                    <strong>⚠️ Guía Técnica:</strong> {hs_action}
                </div>
            </div>
            """
        )

    svg_markup = "\n".join(svg_lines_list)
    hotspots_markup = "\n".join(hotspots_html_list)
    cards_markup = "\n".join(cards_html_list)
    total_count = len(hotspots)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
            -webkit-tap-highlight-color: transparent;
        }}
        body {{
            background-color: #07090e;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 8px;
        }}
        .hud-wrapper {{
            position: relative;
            width: 100%;
            max-width: 900px;
            background: #090d16;
            border: 1px solid {p_border};
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 0 20px {p_glow};
        }}
        .hud-status-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 14px;
            background: rgba(10, 16, 28, 0.95);
            border-bottom: 1px solid rgba(0, 240, 255, 0.2);
            font-size: 11px;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: {p_color};
            font-weight: 600;
        }}
        .hud-status-controls {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        .hud-btn-toggle {{
            background: rgba(0, 240, 255, 0.15);
            border: 1px solid {p_border};
            color: #ffffff;
            font-size: 10px;
            padding: 3px 8px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .hud-btn-toggle:hover {{
            background: {p_color};
            color: #07090e;
        }}
        .hud-viewport {{
            position: relative;
            width: 100%;
            display: block;
            background: #000;
            touch-action: none;
        }}
        .hud-bg-image {{
            width: 100%;
            height: auto;
            max-height: 520px;
            object-fit: contain;
            display: block;
        }}
        /* RETÍCULA HOLOGRÁFICA */
        .hud-grid-overlay {{
            position: absolute;
            inset: 0;
            pointer-events: none;
            background: linear-gradient(rgba(0, 240, 255, 0.03) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(0, 240, 255, 0.03) 1px, transparent 1px);
            background-size: 24px 24px;
        }}
        /* LÍNEAS GUÍA Y PUNTOS EXACTOS EN SVG */
        .hud-lines-svg {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 6;
            transition: opacity 0.25s ease;
        }}
        .hud-anchor-dot {{
            fill: {p_color};
            filter: drop-shadow(0 0 2px {p_color});
        }}
        .hud-anchor-pulse {{
            fill: none;
            stroke: {p_color};
            stroke-width: 0.35;
            opacity: 0.8;
            animation: anchor-breathe 2s infinite ease-in-out;
        }}
        @keyframes anchor-breathe {{
            0%, 100% {{ opacity: 0.4; r: 1.4; }}
            50% {{ opacity: 1; r: 2.2; }}
        }}
        .hud-leader-line {{
            stroke: {p_color};
            stroke-width: 0.45;
            stroke-dasharray: 1.2 0.8;
            opacity: 0.75;
            transition: stroke 0.2s ease, stroke-width 0.2s ease;
        }}
        .hud-leader-group.active .hud-leader-line {{
            stroke: #ffffff;
            stroke-width: 0.85;
            stroke-dasharray: none;
            opacity: 1;
            filter: drop-shadow(0 0 4px {p_color});
        }}
        .hud-leader-group.active .hud-anchor-dot {{
            fill: #ffffff;
            filter: drop-shadow(0 0 4px #ffffff);
        }}
        /* PIN INTERACTIVO / BEACON ARRASTRABLE */
        .hud-hotspot-beacon {{
            position: absolute;
            width: 28px;
            height: 28px;
            transform: translate(-50%, -50%);
            cursor: grab;
            z-index: 10;
            display: flex;
            justify-content: center;
            align-items: center;
            transition: transform 0.15s ease-out, opacity 0.25s ease;
            touch-action: none;
        }}
        .hud-hotspot-beacon:active, .hud-hotspot-beacon.dragging {{
            cursor: grabbing;
            transform: translate(-50%, -50%) scale(1.35);
            z-index: 50;
        }}
        .hud-hotspot-beacon:hover, .hud-hotspot-beacon.active {{
            transform: translate(-50%, -50%) scale(1.28);
            z-index: 25;
        }}
        .beacon-core {{
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: {p_color};
            color: #07090e;
            font-size: 11px;
            font-weight: 800;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 12px {p_color};
            border: 2px solid #ffffff;
            z-index: 2;
        }}
        .radar-pulse {{
            position: absolute;
            width: 38px;
            height: 38px;
            border-radius: 50%;
            border: 2px solid {p_color};
            animation: pulse-ring 1.8s infinite ease-out;
            pointer-events: none;
        }}
        @keyframes pulse-ring {{
            0% {{ transform: scale(0.6); opacity: 1; }}
            100% {{ transform: scale(1.6); opacity: 0; }}
        }}
        .beacon-tag {{
            position: absolute;
            top: 28px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(7, 10, 18, 0.90);
            border: 1px solid {p_border};
            color: #ffffff;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
            white-space: nowrap;
            pointer-events: none;
            opacity: 0.9;
            box-shadow: 0 2px 6px rgba(0,0,0,0.6);
        }}
        /* TARJETA DE INSPECCIÓN FLOTANTE */
        .hud-inspect-card {{
            display: none;
            position: absolute;
            bottom: 12px;
            left: 50%;
            transform: translateX(-50%);
            width: 92%;
            max-width: 480px;
            background: {p_bg};
            border: 1px solid {p_border};
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.8), 0 0 15px {p_glow};
            backdrop-filter: blur(8px);
            z-index: 60;
            animation: slide-up 0.22s ease-out forwards;
        }}
        @keyframes slide-up {{
            from {{ opacity: 0; transform: translate(-50%, 12px); }}
            to {{ opacity: 1; transform: translate(-50%, 0); }}
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}
        .card-badge {{
            font-size: 10px;
            font-weight: 700;
            color: {p_color};
            letter-spacing: 0.5px;
        }}
        .card-close {{
            background: transparent;
            border: none;
            color: #94a3b8;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            padding: 2px 6px;
        }}
        .card-close:hover {{
            color: #ffffff;
        }}
        .card-title {{
            font-size: 14px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 6px;
        }}
        .card-spec-row {{
            display: flex;
            gap: 6px;
            font-size: 11px;
            margin-bottom: 6px;
            background: rgba(255, 255, 255, 0.05);
            padding: 4px 6px;
            border-radius: 4px;
        }}
        .spec-label {{
            color: #94a3b8;
            font-weight: 600;
        }}
        .spec-value {{
            color: {p_color};
            font-weight: 700;
        }}
        .card-desc {{
            font-size: 11px;
            color: #cbd5e1;
            line-height: 1.4;
            margin-bottom: 8px;
        }}
        .card-guidance {{
            font-size: 11px;
            color: #f1f5f9;
            background: rgba(0, 240, 255, 0.08);
            border-left: 3px solid {p_color};
            padding: 6px 8px;
            border-radius: 0 4px 4px 0;
            line-height: 1.35;
        }}
    </style>
</head>
<body>
    <div class="hud-wrapper">
        <div class="hud-status-bar">
            <span>🔮 JARVIS // HUD INTERACTIVO</span>
            <div class="hud-status-controls">
                <span>{safe_title} ({total_count} PUNTOS)</span>
                <button class="hud-btn-toggle" onclick="toggleOverlayVisibility()">👁️ Ocultar / Mostrar</button>
            </div>
        </div>
        <div class="hud-viewport" id="viewport" onclick="handleViewportClick(event)">
            <img class="hud-bg-image" src="{image_base64}" alt="Captura AR">
            <div class="hud-grid-overlay"></div>
            <!-- CAPA SVG DE LÍNEAS CONECTORAS Y PUNTOS EXACTOS -->
            <svg class="hud-lines-svg" id="svg-layer" viewBox="0 0 100 100" preserveAspectRatio="none">
                {svg_markup}
            </svg>
            <!-- BOTONES TÁCTILES EN COLUMNAS LATERALES (ARRASTRABLES) -->
            {hotspots_markup}
            <!-- TARJETAS HOLOGRÁFICAS EMERGENTES -->
            {cards_markup}
        </div>
    </div>

    <script>
        let currentDrag = null;
        let startClientX = 0;
        let startClientY = 0;
        let hasMoved = false;
        let overlayVisible = true;

        // Inicializar listeners de arrastre (Drag & Drop) para cada beacon
        document.querySelectorAll('.hud-hotspot-beacon').forEach(beacon => {{
            const hsId = beacon.getAttribute('data-hsid');
            const line = document.getElementById('line-' + hsId);
            const leader = document.getElementById('leader-' + hsId);

            function startDrag(e) {{
                const clientX = e.touches ? e.touches[0].clientX : e.clientX;
                const clientY = e.touches ? e.touches[0].clientY : e.clientY;
                currentDrag = {{
                    beacon: beacon,
                    line: line,
                    leader: leader,
                    hsId: hsId,
                    viewport: document.getElementById('viewport')
                }};
                startClientX = clientX;
                startClientY = clientY;
                hasMoved = false;
                beacon.classList.add('dragging');
                if (leader) leader.classList.add('active');
            }}

            beacon.addEventListener('mousedown', startDrag);
            beacon.addEventListener('touchstart', startDrag, {{ passive: true }});
        }});

        window.addEventListener('mousemove', handleMove);
        window.addEventListener('touchmove', handleMove, {{ passive: false }});

        function handleMove(e) {{
            if (!currentDrag) return;
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;

            const delta = Math.hypot(clientX - startClientX, clientY - startClientY);
            if (delta > 4) {{
                hasMoved = true;
                if (e.cancelable) e.preventDefault();
            }}

            const rect = currentDrag.viewport.getBoundingClientRect();
            let xPct = ((clientX - rect.left) / rect.width) * 100;
            let yPct = ((clientY - rect.top) / rect.height) * 100;

            // Restringir a márgenes seguros de pantalla
            xPct = Math.max(3.0, Math.min(97.0, xPct));
            yPct = Math.max(4.0, Math.min(96.0, yPct));

            currentDrag.beacon.style.left = xPct.toFixed(2) + '%';
            currentDrag.beacon.style.top = yPct.toFixed(2) + '%';

            // Actualizar la línea de llamada en tiempo real
            if (currentDrag.line) {{
                currentDrag.line.setAttribute('x2', xPct.toFixed(2));
                currentDrag.line.setAttribute('y2', yPct.toFixed(2));
            }}
        }}

        window.addEventListener('mouseup', endDrag);
        window.addEventListener('touchend', endDrag);

        function endDrag(e) {{
            if (!currentDrag) return;
            currentDrag.beacon.classList.remove('dragging');
            currentDrag = null;
        }}

        function handleBeaconClick(id) {{
            // Si el usuario estaba arrastrando para recolocarlo, no abrir la tarjeta
            if (hasMoved) {{
                return;
            }}
            toggleHotspot(id);
        }}

        function closeAllCards() {{
            document.querySelectorAll('.hud-inspect-card').forEach(c => c.style.display = 'none');
            document.querySelectorAll('.hud-hotspot-beacon').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.hud-leader-group').forEach(g => g.classList.remove('active'));
        }}

        function toggleHotspot(id) {{
            const card = document.getElementById('card-' + id);
            const beacon = document.getElementById('beacon-' + id);
            const leader = document.getElementById('leader-' + id);
            if (!card || !beacon) return;

            const isCurrentlyOpen = (card.style.display === 'block');
            closeAllCards();

            if (!isCurrentlyOpen) {{
                card.style.display = 'block';
                beacon.classList.add('active');
                if (leader) leader.classList.add('active');
            }}
        }}

        function hoverHotspot(id, isHovering) {{
            if (currentDrag) return;
            const card = document.getElementById('card-' + id);
            const leader = document.getElementById('leader-' + id);
            if (!leader) return;

            // Si la tarjeta ya está abierta por clic, mantener activo
            if (card && card.style.display === 'block') return;

            if (isHovering) {{
                leader.classList.add('active');
            }} else {{
                leader.classList.remove('active');
            }}
        }}

        function toggleOverlayVisibility() {{
            overlayVisible = !overlayVisible;
            const beacons = document.querySelectorAll('.hud-hotspot-beacon');
            const svgLayer = document.getElementById('svg-layer');
            beacons.forEach(b => {{
                b.style.opacity = overlayVisible ? '1' : '0';
                b.style.pointerEvents = overlayVisible ? 'auto' : 'none';
            }});
            if (svgLayer) svgLayer.style.opacity = overlayVisible ? '1' : '0';
            closeAllCards();
        }}

        function handleViewportClick(event) {{
            if (!event.target.closest('.hud-hotspot-beacon') && !event.target.closest('.hud-inspect-card') && !event.target.closest('.hud-btn-toggle')) {{
                closeAllCards();
            }}
        }}
    </script>
</body>
</html>
"""


def render_interactive_ar_hud(
    image_rgb: np.ndarray,
    targets: List[ARTarget],
    theme: str = "CYAN"
) -> str:
    """Función de alto nivel que consolida todos los hotspots y genera el HTML del visor.

    Args:
        image_rgb: Fotograma original o procesado en formato NumPy RGB.
        targets: Lista de objetivos detectados que contienen hotspots.
        theme: Clave de tema cromático HUD.

    Returns:
        str: Código HTML listo para renderizar.
    """
    all_hotspots: List[ARHotspot] = []
    title_parts: List[str] = []

    for t in targets:
        if t.label:
            title_parts.append(t.label)
        all_hotspots.extend(t.hotspots)

    equipment_title = " // ".join(title_parts) if title_parts else "Inspección AR"
    image_b64 = encode_image_to_base64_jpeg(image_rgb, quality=80, max_dimension=1200)

    return build_interactive_hud_html(
        image_base64=image_b64,
        hotspots=all_hotspots,
        theme=theme,
        equipment_title=equipment_title
    )
