"""Aplicación principal de Realidad Aumentada e Inteligencia Artificial (Jarvis AR).

Propósito:
    Interfaz web interactiva construida sobre Streamlit y optimizada tanto
    para teléfonos inteligentes (iOS / Android) como para PC de escritorio.
    Procesa capturas de cámara, identifica marcas, modelos y números de serie,
    proyecta retículas y tarjetas holográficas AR en tiempo real y ofrece
    un Asistente Interactivo de Manuales Técnicos con consultas en lenguaje natural.

Ejemplo de ejecución:
    $ streamlit run src/app.py
"""

import io
from pathlib import Path
import socket
import time
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# Soporte para formato HEIC de iPhones
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from src.ar_engine import AREngine, COLOR_PALETTES
from src.camera import CameraManager
from src.config import AppSettings, get_settings
from src.interactive_hud import render_interactive_ar_hud
from src.manual_engine import DeviceManual
from src.vision_ai import SceneAnalysisResult, VisionEngine

# Directorio de muestras visuales
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
SAMPLES_DIR: Path = ROOT_DIR / "assets" / "samples"


def get_local_ip() -> str:
    """Obtiene la dirección IP local de la máquina para conexión desde el móvil.

    Returns:
        str: Dirección IP local (ej. '192.168.1.43').
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def apply_custom_styles() -> None:
    """Inyecta estilos CSS para estética HUD sci-fi y responsividad móvil."""
    st.markdown(
        """
        <style>
            .main {
                background-color: #0b0e14;
                color: #e2e8f0;
            }
            .stAppHeader {
                background-color: rgba(11, 14, 20, 0.9);
            }
            .telemetry-card {
                background: linear-gradient(135deg, rgba(14, 23, 42, 0.8), rgba(15, 23, 42, 0.4));
                border: 1px solid #00f0ff;
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 12px;
                box-shadow: 0 0 10px rgba(0, 240, 255, 0.15);
            }
            .target-card {
                background: rgba(15, 23, 42, 0.85);
                border-left: 4px solid #00f0ff;
                border-radius: 4px;
                padding: 10px;
                margin-bottom: 8px;
            }
            .spec-badge {
                display: inline-block;
                background: rgba(0, 240, 255, 0.15);
                border: 1px solid #00f0ff;
                color: #00f0ff;
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 0.8rem;
                margin-right: 6px;
                margin-bottom: 6px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodifica bytes de imagen en cualquier formato (JPEG, PNG, HEIC, WEBP) a matriz BGR.

    Args:
        image_bytes: Contenido binario del archivo de imagen.

    Returns:
        np.ndarray: Imagen en formato BGR lista para OpenCV.

    Raises:
        ValueError: Si los bytes no corresponden a un formato reconocible.
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb_arr = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    except Exception as err:
        file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame_bgr is not None:
            return frame_bgr
        raise ValueError(f"No se pudo decodificar el formato de la imagen capturada: {err}") from err


def process_image_frame(
    image_bytes: bytes,
    vision_engine: VisionEngine,
    ar_engine: AREngine,
    manual_override: Optional[str] = None
) -> tuple[np.ndarray, np.ndarray, SceneAnalysisResult, float]:
    """Decodifica el frame, realiza la inferencia de IA y renderiza la capa AR.

    Args:
        image_bytes: Bytes del archivo de imagen subido o capturado.
        vision_engine: Motor de inferencia multimodal.
        ar_engine: Motor gráfico de Realidad Aumentada.
        manual_override: Perfil forzado para modo desconectado / local.

    Returns:
        tuple: (frame_original_rgb, frame_ar_rgb, resultado_analisis, tiempo_ms).
    """
    start_time: float = time.perf_counter()

    frame_bgr = decode_image_bytes(image_bytes)

    # 1. Inferencia de visión por IA con detección de marca, modelo y manual
    analysis: SceneAnalysisResult = vision_engine.analyze_frame(
        frame_bgr,
        manual_override=manual_override
    )

    # 2. Renderizado de la capa de Realidad Aumentada de alta resolución
    telemetry_header = f"JARVIS // {analysis.engine_used.upper()}"
    frame_ar_bgr = ar_engine.render_overlay(
        frame_bgr=frame_bgr,
        targets=analysis.targets,
        telemetry_text=telemetry_header,
        show_hud_grid=True
    )

    elapsed_ms: float = (time.perf_counter() - start_time) * 1000.0

    frame_orig_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_ar_rgb = cv2.cvtColor(frame_ar_bgr, cv2.COLOR_BGR2RGB)

    return frame_orig_rgb, frame_ar_rgb, analysis, elapsed_ms


def render_ar_dashboard(
    image_bytes: bytes,
    vision_engine: VisionEngine,
    ar_engine: AREngine,
    manual_override: Optional[str] = None
) -> None:
    """Renderiza el visor AR, métricas, tarjetas holográficas y el asistente de manual."""
    try:
        with st.spinner("Identificando marcas, códigos y vinculando manuales técnicos..."):
            orig_rgb, ar_rgb, analysis, latency_ms = process_image_frame(
                image_bytes, vision_engine, ar_engine, manual_override=manual_override
            )

        # Métricas principales
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Equipos Identificados", len(analysis.targets))
        with col_m2:
            st.metric("Latencia de Inferencia", f"{latency_ms:.1f} ms")
        with col_m3:
            st.metric("Motor de Análisis", analysis.engine_used)

        if analysis.api_error:
            st.warning(f"⚠️ **Diagnóstico de Conectividad IA:** {analysis.api_error}")

        st.divider()

        # Visor Principal de Realidad Aumentada
        st.markdown("### 🔮 **Visor de Realidad Aumentada (HUD)**")
        total_hotspots = sum(len(t.hotspots) for t in analysis.targets)

        if total_hotspots > 0:
            tab_interactive, tab_static = st.tabs([
                f"✨ HUD Interactivo Táctil ({total_hotspots} Hotspots)",
                "📸 Superposición AR Clásica (OpenCV)"
            ])

            with tab_interactive:
                st.caption(
                    "👆 **Toca directamente sobre los números, pines o componentes en la pantalla** "
                    "para desplegar sus especificaciones holográficas, voltajes y recomendaciones."
                )
                hud_theme = getattr(ar_engine, "theme", "CYAN")
                interactive_html = render_interactive_ar_hud(
                    orig_rgb,
                    analysis.targets,
                    theme=hud_theme
                )
                st.components.v1.html(interactive_html, height=580, scrolling=False)

                # Desglose rápido por componentes detectados
                all_hotspots = [h for t in analysis.targets for h in t.hotspots]
                with st.expander(f"📋 Desglose Rápido de Componentes ({len(all_hotspots)} detectados)", expanded=False):
                    for h in all_hotspots:
                        c_h1, c_h2, c_h3 = st.columns([2, 2, 3])
                        with c_h1:
                            st.markdown(f"**{h.name}**  \n`{h.category}`")
                        with c_h2:
                            st.markdown(f"⚡ **Valor / Tensión:** `{h.voltage_or_spec or 'N/D'}`")
                        with c_h3:
                            st.caption(f"{h.description} ({h.action_guidance or 'Verificado'})")
                        st.divider()

            with tab_static:
                st.image(ar_rgb, width="stretch")
        else:
            st.image(ar_rgb, width="stretch")

        with st.expander("📸 Ver captura original sin procesar"):
            st.image(orig_rgb, width="stretch")

        # -------------------------------------------------------------
        # SECCIÓN DE MANUAL TÉCNICO Y ASISTENTE INTERACTIVO
        # -------------------------------------------------------------
        targets_with_manual = [t for t in analysis.targets if t.manual is not None]

        if not analysis.targets:
            st.info("ℹ️ **No se detectó ningún equipo en el encuadre.** Apunta la cámara directamente al dispositivo, panel frontal o etiqueta técnica.")

        if targets_with_manual:
            st.divider()
            st.markdown("## 📖 **Manual Técnico y Asistente Interactivo del Equipo**")

            selected_target_idx = 0
            if len(targets_with_manual) > 1:
                options = [f"[{t.brand.upper()}] {t.model_number} - {t.label}" for t in targets_with_manual]
                choice = st.selectbox("Selecciona el equipo a consultar:", options=options)
                selected_target_idx = options.index(choice)

            active_target = targets_with_manual[selected_target_idx]
            manual: DeviceManual = active_target.manual

            # Cabecera de identidad del dispositivo
            st.success(
                f"Dispositivo Identificado: **{manual.equipment_name}** | "
                f"Fabricante: **{manual.brand}** | "
                f"Modelo: **{manual.model_number}**"
            )
            st.caption(f"Fuente del Manual: `{manual.manual_source}` | Estado: Sincronizado en tiempo real")

            # Pestañas de inspección técnica
            tab_specs, tab_guide, tab_trouble, tab_chat = st.tabs([
                "📊 Ficha Técnica",
                "⚡ Puesta en Marcha & Seguridad",
                "🛠️ Solución de Problemas (Troubleshooting)",
                "💬 Preguntar al Manual del Equipo"
            ])

            with tab_specs:
                st.markdown("#### ⚙️ Especificaciones de Ingeniería Oficiales")
                if manual.specifications:
                    for k, v in manual.specifications.items():
                        st.markdown(f"- **{k}:** `{v}`")
                else:
                    st.info("Sin especificaciones adicionales registradas.")

            with tab_guide:
                st.markdown("#### ⚠️ Advertencias Críticas de Seguridad")
                for w in manual.safety_warnings:
                    st.warning(f"⚠️ {w}")

                st.markdown("#### 🚀 Guía de Arranque y Conexión")
                for step in manual.quick_start_steps:
                    st.markdown(f"- {step}")

            with tab_trouble:
                st.markdown("#### 🔧 Códigos de Error y Diagnóstico")
                if manual.troubleshooting_codes:
                    for err_name, sol in manual.troubleshooting_codes.items():
                        with st.expander(f"🔴 {err_name}", expanded=False):
                            st.write(f"**Causa y Solución Técnica:** {sol}")
                else:
                    st.info("No se registran anomalías ni códigos de falla conocidos.")

            with tab_chat:
                st.markdown("#### 💬 Asistente Técnico Especializado")
                st.caption("Formula preguntas sobre procedimientos, esquemas de pines, errores o mantenimiento:")

                # Sugerencias rápidas de preguntas
                q_col1, q_col2, q_col3 = st.columns(3)
                suggested_query = ""
                with q_col1:
                    if st.button("🔌 ¿Qué voltaje de entrada usa?", width="stretch"):
                        suggested_query = "¿Cuál es el voltaje de alimentación recomendado y límites máximos?"
                with q_col2:
                    if st.button("🔄 ¿Cómo se reinicia o arranca?", width="stretch"):
                        suggested_query = "¿Cuál es el procedimiento paso a paso de puesta en marcha o reinicio?"
                with q_col3:
                    if st.button("⚠️ ¿Qué precauciones debo tener?", width="stretch"):
                        suggested_query = "¿Cuáles son las advertencias de seguridad más críticas de este equipo?"

                user_query = st.text_input(
                    "Escribe tu pregunta sobre el equipo:",
                    value=suggested_query,
                    placeholder="Ejemplo: ¿Qué hago si la luz parpadea en ámbar o se calienta?",
                    key="manual_chat_input"
                )

                if st.button("🚀 Consultar al Manual", type="primary") and user_query:
                    with st.spinner(f"Analizando manual técnico de {manual.equipment_name}..."):
                        answer = vision_engine.query_target_manual(active_target, user_query)
                    st.markdown("### 🤖 Respuesta del Manual Técnico:")
                    st.info(answer)

        # Tarjetas holográficas estándar
        st.divider()
        st.markdown("### 📋 **Resumen de Detección AR**")
        st.info(f"**Resumen de la Escena:** {analysis.summary}")

        for idx, target in enumerate(analysis.targets, start=1):
            with st.expander(f"🎯 Objetivo #{idx}: [{target.brand.upper()}] {target.label} (Confianza: {int(target.confidence * 100)}%)", expanded=True):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.write(f"**Fabricante:** `{target.brand}`")
                    st.write(f"**Modelo:** `{target.model_number}`")
                    st.write(f"**Serial / Ref:** `{target.serial_number or 'N/A'}`")
                    st.write(f"**Categoría:** `{target.category}`")
                with c2:
                    st.write(f"**Detalle Técnico:** {target.details}")
                    st.success(f"**Acción Guiada:** {target.action_hint}")

    except ValueError as val_err:
        st.error(f"Error en formato de imagen: {val_err}")
    except Exception as unk_err:
        st.error(f"Error inesperado en pipeline AR: {unk_err}")


def main() -> None:
    """Función principal de inicialización y ciclo de vida de la UI."""
    st.set_page_config(
        page_title="JARVIS AR // Spatial AI",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_custom_styles()
    settings: AppSettings = get_settings()

    # Barra lateral: Telemetría y Configuración
    with st.sidebar:
        st.markdown("## 🛰️ **NEXO AR COMMAND**")
        st.caption("Sistema de Fusión: Realidad Aumentada + IA Multimodal")
        st.divider()

        theme_names = list(COLOR_PALETTES.keys())
        selected_theme = st.selectbox(
            "Paleta Cromática HUD:",
            options=theme_names,
            index=theme_names.index(settings.ar_hud_theme) if settings.ar_hud_theme in theme_names else 0
        )

        st.markdown("### 🔑 **Configuración de IA (Gemini API)**")
        gemini_key_input = st.text_input(
            "Clave Gemini API (Opcional):",
            value=settings.gemini_api_key or "",
            type="password",
            help="Si se deja vacía, se utiliza el motor analítico local y catálogo técnico integrado."
        )

        effective_key = gemini_key_input.strip() if gemini_key_input and gemini_key_input.strip() else None

        if effective_key:
            st.success("🟢 Modo IA: Gemini Multimodal (Cloud)")
        else:
            st.info("🔵 Modo IA: Analítico Local (Catálogo Integrado)")

        st.markdown("### 🎯 **Modo de Reconocimiento Local**")
        profile_options = [
            "Auto (Detección Heurística + QR)",
            "Xiaomi Smartphone (HyperOS)",
            "Arduino UNO R3",
            "Cisco Catalyst Switch",
            "Multímetro Fluke 87V"
        ]
        selected_profile = st.selectbox(
            "Perfil Forzado en Modo Local:",
            options=profile_options,
            index=0,
            help="Permite elegir directamente el equipo a consultar sin depender de la nube o visión multimodal."
        )

        manual_override_code = None
        if selected_profile != profile_options[0]:
            manual_override_code = selected_profile

        if not effective_key:
            st.caption(
                "💡 **¿Por qué salía Arduino?** Sin clave API, el sistema trabaja en modo offline "
                "(geometría + QR). Para reconocimiento multimodal automático de marcas y texto serigrafiado, "
                "ingresa tu clave gratuita de Google AI Studio, o selecciona 'Xiaomi Smartphone' en el selector superior."
            )

        st.divider()

        local_ip: str = get_local_ip()
        mobile_url: str = settings.public_tunnel_url or f"http://{local_ip}:{settings.server_port}"
        qr_image_url: str = f"https://api.qrserver.com/v1/create-qr-code/?size=250x250&data={mobile_url}"

        st.markdown("### 📱 **Conectar Teléfono Móvil**")
        st.image(qr_image_url, caption="Escanea el QR con tu móvil", width="stretch")
        st.markdown(f"**Enlace Web:** [{mobile_url}]({mobile_url})")
        st.caption("Protocolo HTTPS cifrado: permite usar la cámara de tu smartphone sin restricciones.")

    # Inicialización de motores con la clave activa
    vision_engine = VisionEngine(api_key=effective_key)
    ar_engine = AREngine(theme=selected_theme)

    st.title("👁️ JARVIS AR — Asistente Espacial & Manuales Técnicos")
    st.markdown(
        "Apunta a cualquier equipo o dispositivo para identificar **marcas, números de modelo y logotipos**, "
        "desplegar **tarjetas holográficas AR** y consultar en tiempo real el **manual de usuario y ficha técnica**."
    )

    # Bloque de conexión rápida con Smartphone mediante QR
    with st.expander("📲 **Escanear Código QR para Abrir en el Móvil**", expanded=False):
        c_qr_img, c_qr_txt = st.columns([1, 2])
        with c_qr_img:
            st.image(qr_image_url, width="stretch")
        with c_qr_txt:
            st.markdown("#### ⚡ Conexión Instantánea Móvil")
            st.markdown(f"- **Enlace HTTPS Directo:** [{mobile_url}]({mobile_url})")
            st.markdown("- **Cámara Habilitada:** Protocolo SSL/HTTPS que habilita el sensor de cámara en Android y iPhone.")
            st.markdown("- **IA Multimodal:** Clave API cargada en el servidor (gemini-3.8-flash).")

    if not effective_key:
        st.warning(
            "ℹ️ **Modo Offline / Local:** La clave Gemini API no está configurada en esta sesión. "
            "El sistema funcionará con códigos QR y perfiles predefinidos. Para reconocimiento multimodal automático "
            "de cualquier objeto o producto fotografiado con el móvil, despliega el menú lateral izquierdo (`>`) e introduce tu API Key, "
            "o agrégala al archivo `.env` del servidor."
        )

    # Botones de prueba instantánea de un solo clic
    st.markdown("#### 🧪 **Prueba Inmediata con un Clic (Muestras con Manuales Vinculados):**")
    col_s1, col_s2, col_s3 = st.columns(3)
    sample_to_load: Optional[str] = None

    with col_s1:
        if st.button("🔬 Muestra: Placa Arduino UNO", width="stretch"):
            sample_to_load = "gadget.jpg"
    with col_s2:
        if st.button("☕ Muestra: Taza en Escritorio", width="stretch"):
            sample_to_load = "desk.jpg"
    with col_s3:
        if st.button("📖 Muestra: Documento Técnico", width="stretch"):
            sample_to_load = "book.jpg"

    if sample_to_load is not None:
        sample_path = SAMPLES_DIR / sample_to_load
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.session_state["active_image_bytes"] = f.read()

    # Selector de pestañas según dispositivo
    tab_pc, tab_mobile, tab_webrtc, tab_upload = st.tabs([
        "💻 Webcam PC (Directa)",
        "📱 Móvil (Hacer Foto con Cámara)",
        "🌐 Visor WebRTC",
        "📁 Subir Imagen Manual"
    ])

    with tab_pc:
        st.markdown("### 💻 **Webcam del PC**")
        st.caption("Captura un fotograma directamente desde el sensor de video de tu equipo.")
        capture_btn = st.button("📸 Capturar con la Webcam del PC", type="primary", width="stretch")

        if capture_btn:
            try:
                with CameraManager(camera_index=settings.camera_index) as cam:
                    if not cam.is_opened():
                        st.error(
                            "❌ La webcam no respondió. Esto ocurre si otra aplicación (como Zoom, Teams "
                            "o una pestaña del navegador con WebRTC) la está usando. "
                            "Cierra la otra app o prueba una de las muestras superiores."
                        )
                    else:
                        success, frame_cam = cam.read_frame()
                        if success and frame_cam is not None:
                            ok_enc, buf = cv2.imencode(".jpg", frame_cam)
                            if ok_enc:
                                st.session_state["active_image_bytes"] = buf.tobytes()
                                st.success("✅ ¡Captura tomada desde la webcam!")
                        else:
                            st.error("No se pudo leer el fotograma del sensor de cámara.")
            except Exception as cam_err:
                st.error(f"Error al acceder al hardware de cámara: {cam_err}")

    with tab_mobile:
        st.markdown("### 📱 **Cámara de Smartphone**")
        st.info(
            "👇 **Pulsa el cuadro de abajo**. Tu teléfono te dará la opción de abrir la **Cámara** "
            "para hacer una foto en este momento. Al disparar la foto, el visor AR se proyectará inmediatamente."
        )

        with st.expander("📷 ¿Viendo esta web desde el PC? Escanea el QR para pasar al móvil:", expanded=False):
            col_m_qr, col_m_info = st.columns([1, 2])
            with col_m_qr:
                st.image(qr_image_url, width="stretch")
            with col_m_info:
                st.markdown("Apunta con la cámara de tu smartphone para abrir la app en tu teléfono.")
                st.caption(f"URL: {mobile_url}")
        mobile_file = st.file_uploader(
            "Toca aquí para disparar la cámara del móvil:",
            type=["jpg", "jpeg", "png", "heic", "heif", "webp"],
            key="mobile_camera_input"
        )
        if mobile_file is not None:
            st.session_state["active_image_bytes"] = mobile_file.getvalue()

    with tab_webrtc:
        st.markdown("### 🌐 **Visor WebRTC Integrado**")
        st.caption("Funciona en conexiones seguras (ej. http://localhost:8501 en este PC).")
        webrtc_file = st.camera_input("Visor WebRTC", key="webrtc_camera_widget")
        if webrtc_file is not None:
            st.session_state["active_image_bytes"] = webrtc_file.getvalue()

    with tab_upload:
        st.markdown("### 📁 **Subir Archivo de Imagen**")
        uploaded_file = st.file_uploader(
            "Selecciona cualquier foto guardada en tu dispositivo:",
            type=["jpg", "jpeg", "png", "heic", "heif", "webp"],
            key="file_upload_widget"
        )
        if uploaded_file is not None:
            st.session_state["active_image_bytes"] = uploaded_file.getvalue()

    # RENDERIZADO DEL DASHBOARD AR
    if "active_image_bytes" in st.session_state and st.session_state["active_image_bytes"] is not None:
        st.divider()
        render_ar_dashboard(
            st.session_state["active_image_bytes"],
            vision_engine,
            ar_engine,
            manual_override=manual_override_code
        )
    else:
        st.info("👆 Selecciona una muestra superior o dispara la cámara para activar la proyección holográfica.")


if __name__ == "__main__":
    main()
