"""Módulo de Inteligencia Artificial para análisis de escenas y visión multimodal.

Propósito:
    Analizar fotogramas provenientes de la cámara para detectar objetos y equipos,
    identificar marcas, logotipos y números de modelo serigrafiados, vincular
    el manual técnico correspondiente y estructurar los metadatos proyectados por AR.

Ejemplo de uso:
    >>> import numpy as np
    >>> from src.vision_ai import VisionEngine
    >>> engine = VisionEngine(api_key=None)
    >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
    >>> result = engine.analyze_frame(frame)
    >>> print(result.targets[0].brand)
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from src.manual_engine import DeviceManual, ManualEngine

logger = logging.getLogger(__name__)


class ARHotspot(BaseModel):
    """Punto interactivo de interés (pin, conector, botón o componente) en el equipo."""

    id: str = Field(description="Identificador único del hotspot (ej. 'pin-13', 'usb-port').")
    name: str = Field(description="Nombre descriptivo del componente o punto de prueba.")
    x_percent: float = Field(ge=0.0, le=100.0, description="Posición horizontal en porcentaje relativo (0-100%).")
    y_percent: float = Field(ge=0.0, le=100.0, description="Posición vertical en porcentaje relativo (0-100%).")
    category: str = Field(default="Componente", description="Tipo de componente (ej. Puerto, Botón, LED, Sensor, Chip).")
    description: str = Field(default="", description="Detalle técnico de la función del componente.")
    voltage_or_spec: Optional[str] = Field(default=None, description="Tensión o valor nominal (ej. '5V DC', '3.3V', 'Gigabit Ethernet').")
    action_guidance: Optional[str] = Field(default=None, description="Consejo de conexión o advertencia técnica.")
    status: str = Field(default="NORMAL", description="Estado operativo o alerta ('NORMAL', 'WARNING', 'ACTIVE').")


class ARTarget(BaseModel):
    """Representa un objetivo detectado para ser renderizado en Realidad Aumentada."""

    label: str = Field(description="Nombre o clase identificada del objeto.")
    confidence: float = Field(ge=0.0, le=1.0, description="Nivel de certidumbre del modelo (0.0 a 1.0).")
    bbox: Tuple[int, int, int, int] = Field(description="Bounding box en formato (x, y, ancho, alto).")
    category: str = Field(description="Categoría semántica (ej. 'Hardware', 'Documento', 'Entorno').")
    details: str = Field(description="Descripción técnica corta proyectada en la tarjeta AR.")
    action_hint: str = Field(description="Sugerencia de acción o guía paso a paso.")
    brand: str = Field(default="", description="Marca o fabricante del equipo (ej. Cisco, HP, Arduino).")
    model_number: str = Field(default="", description="Modelo exacto o referencia serigrafiada.")
    serial_number: Optional[str] = Field(default=None, description="Número de serie o etiqueta visible.")
    manual: Optional[DeviceManual] = Field(default=None, description="Manual técnico vinculado al objetivo.")
    hotspots: List[ARHotspot] = Field(default_factory=list, description="Puntos de interés o componentes interactivos en el dispositivo.")


class SceneAnalysisResult(BaseModel):
    """Estructura de datos que consolida el análisis completo del fotograma."""

    summary: str = Field(description="Resumen global de la escena capturada.")
    targets: List[ARTarget] = Field(default_factory=list, description="Lista de objetivos AR detectados.")
    timestamp: str = Field(description="Marca temporal ISO del análisis.")
    engine_used: str = Field(description="Identificador del motor utilizado ('gemini-vision' o 'local-vision-engine').")
    api_error: Optional[str] = Field(default=None, description="Mensaje de error si la API de Gemini falló.")


class VisionEngine:
    """Motor de análisis de visión multimodal con soporte para Gemini y motor local."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Inicializa el motor de visión y el motor de manuales técnicos.

        Args:
            api_key: Clave opcional para Google Gemini API.
        """
        self._api_key: Optional[str] = api_key
        self._gemini_client = None
        self._manual_engine = ManualEngine(api_key=self._api_key)

        if self._api_key:
            self._init_gemini_client()

    def update_api_key(self, new_key: Optional[str]) -> None:
        """Actualiza dinámicamente la clave de API y recrea los clientes."""
        self._api_key = new_key.strip() if new_key and new_key.strip() else None
        self._gemini_client = None
        self._manual_engine = ManualEngine(api_key=self._api_key)
        if self._api_key:
            self._init_gemini_client()

    def _init_gemini_client(self) -> None:
        """Intenta inicializar el cliente oficial de Gemini SDK."""
        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=self._api_key)
        except Exception:
            self._gemini_client = None

    def _get_available_models(self) -> List[str]:
        """Obtiene dinámicamente la lista de modelos multimodales disponibles para la clave API."""
        discovered: List[str] = []
        if self._gemini_client is not None:
            try:
                for m in self._gemini_client.models.list():
                    m_name = getattr(m, "name", "")
                    if not m_name:
                        continue
                    clean_name = m_name.replace("models/", "").strip()
                    if "gemini" in clean_name and "embedding" not in clean_name and "aqa" not in clean_name:
                        discovered.append(clean_name)
            except Exception:
                pass

        priority_order = [
            "gemini-3.5-flash",
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
        final_list: List[str] = []
        for p in priority_order:
            if p in discovered:
                final_list.append(p)
        for d in discovered:
            if d not in final_list:
                final_list.append(d)

        return final_list if final_list else priority_order

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
        manual_override: Optional[str] = None
    ) -> SceneAnalysisResult:
        """Analiza un fotograma en formato OpenCV BGR y retorna los metadatos AR.

        Args:
            frame_bgr: Imagen en formato numpy array BGR.
            manual_override: Forzado opcional de modelo para modo desconectado / local.

        Returns:
            SceneAnalysisResult con objetivos detectados y manuales vinculados.
        """
        now_iso: str = datetime.now().isoformat()

        if self._gemini_client is not None and not manual_override:
            try:
                return self._analyze_with_gemini(frame_bgr, now_iso)
            except Exception as gemini_err:
                error_msg: str = f"Gemini Cloud Error ({type(gemini_err).__name__}): {gemini_err}"
                logger.warning("Fallo en inferencia Gemini (%s).", gemini_err)
                # No inventar objetivos: si la IA falló y no hay override manual, retornar lista vacía
                res = SceneAnalysisResult(
                    summary="No se pudo completar el análisis con la IA en la nube.",
                    targets=[],
                    timestamp=now_iso,
                    engine_used="error-cloud-fallback",
                    api_error=error_msg
                )
                return res

        return self._analyze_locally(frame_bgr, now_iso, manual_override)

    def query_target_manual(self, target: ARTarget, question: str) -> str:
        """Responde una consulta en lenguaje natural sobre el manual del equipo enfocado.

        Args:
            target: Objetivo detectado con manual técnico vinculado.
            question: Pregunta del usuario (ej. "¿Cómo se reinicia?", "¿Qué voltaje usa?").

        Returns:
            str: Respuesta fundamentada en el manual oficial.
        """
        if target.manual is None:
            # Si no hay manual precargado, intentar buscarlo por marca/modelo
            target.manual = self._manual_engine.get_manual(target.brand, target.model_number)

        return self._manual_engine.query_manual(target.manual, question)

    def _analyze_locally(
        self,
        frame_bgr: np.ndarray,
        timestamp: str,
        manual_override: Optional[str] = None
    ) -> SceneAnalysisResult:
        """Motor local heurístico que detecta componentes, QR y vincula manuales del catálogo."""
        h, w = frame_bgr.shape[:2]
        targets: List[ARTarget] = []

        # 1. Escaneo de códigos QR y códigos de serie impresos en pegatinas técnicas
        detected_qr_text: str = ""
        qr_bbox_coords: Optional[tuple[int, int, int, int]] = None
        try:
            qr_detector = cv2.QRCodeDetector()
            qr_data, qr_points, _ = qr_detector.detectAndDecode(frame_bgr)
            if qr_data:
                detected_qr_text = qr_data.strip()
            if qr_points is not None and len(qr_points) > 0:
                pts = qr_points[0].astype(int)
                qx = int(np.min(pts[:, 0]))
                qy = int(np.min(pts[:, 1]))
                qw = int(np.max(pts[:, 0]) - qx)
                qh = int(np.max(pts[:, 1]) - qy)
                if qw > 5 and qh > 5:
                    qr_bbox_coords = (qx, qy, qw, qh)
        except Exception:
            pass

        # 2. Si el usuario seleccionó un perfil específico explícito en modo local
        if manual_override and manual_override != "auto":
            override_lower = manual_override.lower()
            profile_hotspots: List[ARHotspot] = []
            if "xiaomi" in override_lower or "redmi" in override_lower or "phone" in override_lower or "movil" in override_lower:
                brand = "Xiaomi"
                model = "Xiaomi / Redmi Series"
                label = "Smartphone Xiaomi (HyperOS / Android)"
                category = "Dispositivos Móviles / Comunicaciones"
                details = "AMOLED 120Hz | 5000 mAh | HyperCharge 67W/120W"
                action_hint = "Presiona [Power + Vol-] para forzar reinicio. Revisa el manual para diagnósticos."
                profile_hotspots = [
                    ARHotspot(
                        id="xmi-cam",
                        name="Módulo Cámara Principal",
                        x_percent=50.0,
                        y_percent=22.0,
                        category="Sensor Óptico",
                        description="Sensor de alta resolución con estabilización óptica OIS",
                        voltage_or_spec="50 MP / f/1.8",
                        action_guidance="Limpiar con paño de microfibra sin líquidos abrasivos"
                    ),
                    ARHotspot(
                        id="xmi-power",
                        name="Botón Encendido / Sensor Huella",
                        x_percent=88.0,
                        y_percent=42.0,
                        category="Botón / Biometría",
                        description="Lector capacitivo y control de encendido",
                        voltage_or_spec="Sensor Biométrico",
                        action_guidance="Pulsar durante 10 segundos para forzar reinicio en caso de bloqueo"
                    ),
                    ARHotspot(
                        id="xmi-usbc",
                        name="Puerto USB-C HyperCharge",
                        x_percent=50.0,
                        y_percent=92.0,
                        category="Puerto / Carga",
                        description="Conector de carga rápida y transferencia OTG",
                        voltage_or_spec="67W / 120W Max",
                        action_guidance="Verificar ausencia de polvo o suciedad antes de conectar"
                    )
                ]
            elif "cisco" in override_lower:
                brand = "Cisco"
                model = "Catalyst 2960"
                label = "Switch de Red Cisco Catalyst"
                category = "Telecomunicaciones / Networking"
                details = "24 Puertos Gigabit | Alimentación 100-240V AC"
                action_hint = "Consulta el manual para códigos de error en LEDs de estado."
                profile_hotspots = [
                    ARHotspot(
                        id="csc-ports",
                        name="Matriz 24 Puertos Gigabit RJ45",
                        x_percent=55.0,
                        y_percent=50.0,
                        category="Puerto de Red",
                        description="Puertos de conmutación 10/100/1000Base-T",
                        voltage_or_spec="Auto-MDIX Gigabit",
                        action_guidance="Verificar LED verde constante para enlace operativo"
                    ),
                    ARHotspot(
                        id="csc-console",
                        name="Puerto de Consola RJ45",
                        x_percent=15.0,
                        y_percent=50.0,
                        category="Gestión / Serial",
                        description="Interfaz serial para administración CLI local",
                        voltage_or_spec="9600 baud, 8N1",
                        action_guidance="Usar cable consola azul RJ45 a DB9/USB"
                    )
                ]
            elif "fluke" in override_lower:
                brand = "Fluke"
                model = "87V Industrial"
                label = "Multímetro Digital Fluke 87V"
                category = "Instrumentación de Medida"
                details = "True-RMS | CAT IV 600V / CAT III 1000V"
                action_hint = "Revisa advertencias de seguridad antes de medir alta tensión."
                profile_hotspots = [
                    ARHotspot(
                        id="flk-com",
                        name="Borne de Entrada COM",
                        x_percent=58.0,
                        y_percent=82.0,
                        category="Terminal de Medida",
                        description="Borne común de referencia negativa",
                        voltage_or_spec="Referencia 0V",
                        action_guidance="Conectar punta de prueba negra estándar"
                    ),
                    ARHotspot(
                        id="flk-volt",
                        name="Borne de Medida V/Ω/Hz",
                        x_percent=78.0,
                        y_percent=82.0,
                        category="Terminal de Medida",
                        description="Entrada para medición de tensión, resistencia y frecuencia",
                        voltage_or_spec="CAT IV 600V / CAT III 1000V",
                        action_guidance="Comprobar categoría de aislamiento de las puntas antes de medir"
                    )
                ]
            else:
                brand = "Arduino"
                model = "UNO R3"
                label = "Microcontrolador Arduino UNO"
                category = "Electrónica / Embebidos"
                details = "Chip ATmega328P | 14 pines I/O | 5V DC"
                action_hint = "Consulta el manual para asignación de pines o solución de errores."
                profile_hotspots = [
                    ARHotspot(
                        id="ard-pin-13",
                        name="LED / Pin Digital 13",
                        x_percent=42.0,
                        y_percent=38.0,
                        category="LED / Salida",
                        description="LED integrado de pruebas en la placa",
                        voltage_or_spec="5V / HIGH",
                        action_guidance="Controlable vía digitalWrite(13, HIGH/LOW)"
                    ),
                    ARHotspot(
                        id="ard-reset",
                        name="Pulsador Reset",
                        x_percent=22.0,
                        y_percent=26.0,
                        category="Botón / Control",
                        description="Reinicia la ejecución del sketch cargado",
                        voltage_or_spec="Activo a 0V",
                        action_guidance="Pulsar brevemente para reinicio limpio"
                    ),
                    ARHotspot(
                        id="ard-usb",
                        name="Conector USB Tipo B",
                        x_percent=18.0,
                        y_percent=45.0,
                        category="Puerto / Datos",
                        description="Puerto de comunicación serial y alimentación",
                        voltage_or_spec="5V DC / 500mA",
                        action_guidance="Conectar a PC con cable USB A-B estándar"
                    ),
                    ARHotspot(
                        id="ard-dc-jack",
                        name="Jack de Alimentación Externa",
                        x_percent=18.0,
                        y_percent=78.0,
                        category="Alimentación",
                        description="Entrada para adaptador de corriente o batería",
                        voltage_or_spec="7 - 12V DC (Centro positivo)",
                        action_guidance="No superar 12V para evitar sobrecalentamiento del regulador"
                    ),
                    ARHotspot(
                        id="ard-atmega",
                        name="Microcontrolador ATmega328P",
                        x_percent=62.0,
                        y_percent=68.0,
                        category="Microcontrolador",
                        description="Núcleo de procesamiento AVR de 8 bits a 16 MHz",
                        voltage_or_spec="32KB Flash / 2KB SRAM",
                        action_guidance="Evitar cortocircuitos entre terminales contiguos y GND"
                    )
                ]

            manual_obj = self._manual_engine.get_manual(brand, model)
            target_bbox = qr_bbox_coords if qr_bbox_coords else (w // 6, h // 8, (w * 2) // 3, (h * 3) // 4)
            serial_display = detected_qr_text if detected_qr_text else "SN-MI-849204-EU"

            targets.append(
                ARTarget(
                    label=label,
                    confidence=0.96,
                    bbox=target_bbox,
                    category=category,
                    details=details,
                    action_hint=action_hint,
                    brand=brand,
                    model_number=model,
                    serial_number=serial_display,
                    manual=manual_obj,
                    hotspots=profile_hotspots
                )
            )

            return SceneAnalysisResult(
                summary=f"Equipo identificado por selección de perfil técnico: [{brand}] {model}.",
                targets=targets,
                timestamp=timestamp,
                engine_used="local-profile-engine"
            )

        # 3. Si se detectó un código QR físico en el encuadre
        if detected_qr_text:
            target_bbox = qr_bbox_coords if qr_bbox_coords else (w // 4, h // 4, w // 2, h // 2)
            targets.append(
                ARTarget(
                    label="Código / Etiqueta Técnica Detectada",
                    confidence=0.92,
                    bbox=target_bbox,
                    category="Etiqueta / Serial",
                    details=f"Contenido decodificado: {detected_qr_text}",
                    action_hint="Código técnico verificado.",
                    brand="QR Code",
                    model_number=detected_qr_text[:20],
                    serial_number=detected_qr_text,
                    manual=None
                )
            )
            return SceneAnalysisResult(
                summary=f"Código QR detectado: {detected_qr_text}",
                targets=targets,
                timestamp=timestamp,
                engine_used="local-qr-engine"
            )

        # 4. Si no hay QR ni selección forzada de perfil, no inventar ningún objetivo
        return SceneAnalysisResult(
            summary="No se identificó ningún equipo en el encuadre (Modo Local: se requiere Gemini API para análisis multimodal de cualquier objeto).",
            targets=[],
            timestamp=timestamp,
            engine_used="local-vision-engine"
        )

    def _analyze_with_gemini(self, frame_bgr: np.ndarray, timestamp: str) -> SceneAnalysisResult:
        """Inferencia multimodal utilizando el SDK de Gemini con extracción profunda de marcas y modelos."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode(".jpg", frame_rgb)
        image_bytes: bytes = buffer.tobytes()

        prompt: str = (
            "Eres un experto en ingeniería y visión espacial para Realidad Aumentada industrial.\n"
            "Analiza minuciosamente la imagen adjunta. Identifica todos los equipos, herramientas, "
            "placas electrónicas, electrodomésticos o dispositivos visibles.\n"
            "Presta especial atención a:\n"
            "1. LOGOTIPOS y MARCAS visibles (ej. Cisco, HP, Dell, Siemens, Fluke, Arduino, Apple, Sony, etc.).\n"
            "2. NÚMEROS DE MODELO, CÓDIGOS de serie o referencias serigrafiadas en el chasis o etiquetas.\n"
            "3. Especificaciones técnicas (voltaje, interfaces, advertencias de seguridad).\n\n"
            "Responde estrictamente en formato JSON con esta estructura:\n"
            "{\n"
            '  "summary": "Resumen técnico de los equipos o productos identificados en la escena en español",\n'
            '  "targets": [\n'
            "    {\n"
            '      "label": "Nombre descriptivo del producto o equipo",\n'
            '      "brand": "Marca o fabricante identificado (ej. BIOPROX, Xiaomi, Arduino, etc.)",\n'
            '      "model_number": "Modelo, sabor o referencia exacta",\n'
            '      "serial_number": "Número de serie, lote o código si es visible",\n'
            '      "confidence": 0.95,\n'
            '      "bbox_normalized": [ymin, xmin, ymax, xmax] (valores de 0 a 1000),\n'
            '      "category": "Categoría (ej. Nutrición / Suplementación, Dispositivos Móviles, Networking, etc.)",\n'
            '      "details": "Descripción precisa y características destacadas",\n'
            '      "action_hint": "Instrucción de uso, consumo recomendado o advertencia clave",\n'
            '      "hotspots": [\n'
            '        {\n'
            '          "id": "hs-1",\n'
            '          "name": "Nombre del pin, puerto, conector o componente visible (ej. Pin 13, Reset, USB-C)",\n'
            '          "x_percent": 45.0,\n'
            '          "y_percent": 30.0,\n'
            '          "category": "Puerto / Botón / LED / Pin / Chip / Sensor",\n'
            '          "description": "Función técnica concreta de este punto",\n'
            '          "voltage_or_spec": "Tensión o especificación (ej. 5V DC, Reset, Gigabit)",\n'
            '          "action_guidance": "Instrucción de conexión o precaución"\n'
            '        }\n'
            '      ],\n'
            '      "specifications": {\n'
            '        "Parámetro 1": "Valor (ej. Proteínas, Kcal, Alérgenos si es alimento; o Voltaje, Puertos si es electrónico)",\n'
            '        "Parámetro 2": "Valor relevante para este tipo de producto"\n'
            '      },\n'
            '      "safety_warnings": ["Advertencia o precaución de seguridad/alérgenos 1", "Advertencia 2"],\n'
            '      "quick_start_steps": ["Paso 1: Uso o consumo adecuado", "Paso 2", "Paso 3"],\n'
            '      "troubleshooting_codes": {"Duda o problema común": "Solución o indicación técnica"}\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "REGLA ESTRICTA DE COMPATIBILIDAD:\n"
            "- Si es un ALIMENTO o SUPLEMENTO (barrita, proteína, bebida): las especificaciones DEBEN ser nutricionales (Proteína, Kcal, Sabor, Conservación, Alérgenos). NUNCA inventes voltajes ni cables.\n"
            "- Si es ELECTRÓNICA: tensión, puertos, procesador.\n"
            "- Si es HERRAMIENTA o MECÁNICA: materiales, peso, dimensiones."
        )

        contents_list = [prompt]
        if types is not None:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            contents_list.append(image_part)
        else:
            from PIL import Image
            import io
            contents_list.append(Image.open(io.BytesIO(image_bytes)))

        models_to_try = self._get_available_models()
        response = None
        last_error = None

        for model_name in models_to_try:
            for attempt in range(3):
                try:
                    response = self._gemini_client.models.generate_content(
                        model=model_name,
                        contents=contents_list
                    )
                    if response and response.text:
                        logger.info("Inferencia multimodal exitosa con modelo: %s", model_name)
                        break
                except Exception as exc:
                    last_error = exc
                    logger.warning("Intento %d con modelo %s: %s", attempt + 1, model_name, exc)
                    # Si Google reporta 503 (alta demanda momentánea), esperar antes de reintentar
                    if "503" in str(exc) and attempt < 2:
                        import time
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    break
            if response and response.text:
                break

        if response is None or not response.text:
            raise RuntimeError(
                f"Servidores de Google temporalmente congestionados o modelo no disponible. "
                f"Detalle: {last_error}"
            )

        text_content: str = response.text.strip()
        if text_content.startswith("```json"):
            text_content = text_content[7:]
        if text_content.endswith("```"):
            text_content = text_content[:-3]
        text_content = text_content.strip()

        data = json.loads(text_content)
        targets: List[ARTarget] = []

        for item in data.get("targets", []):
            norm_box = item.get("bbox_normalized", [200, 200, 800, 800])
            ymin, xmin, ymax, xmax = norm_box
            x = int((xmin / 1000.0) * w)
            y = int((ymin / 1000.0) * h)
            bw = int(((xmax - xmin) / 1000.0) * w)
            bh = int(((ymax - ymin) / 1000.0) * h)

            brand = item.get("brand", "Equipo Técnico")
            model = item.get("model_number", "")
            cat = item.get("category", "General")
            det = item.get("details", "")

            # Si Gemini ya extrajo las especificaciones del producto en la misma llamada visual
            if item.get("specifications") and isinstance(item.get("specifications"), dict):
                manual_obj = DeviceManual(
                    equipment_name=item.get("label", f"{brand} {model}".strip()),
                    brand=brand,
                    model_number=model,
                    category=cat,
                    specifications=item.get("specifications", {}),
                    safety_warnings=item.get("safety_warnings", []),
                    quick_start_steps=item.get("quick_start_steps", []),
                    troubleshooting_codes=item.get("troubleshooting_codes", {}),
                    manual_source="IA Multimodal Gemini (Inspección Visual Directa)",
                    raw_text_summary=det
                )
            else:
                manual_obj = self._manual_engine.get_manual(brand, model, category=cat, details=det)

            # Extracción y validación defensiva de puntos de interés interactivos (Hotspots)
            parsed_hotspots: List[ARHotspot] = []
            for h_idx, h_item in enumerate(item.get("hotspots", [])):
                if not isinstance(h_item, dict):
                    continue
                try:
                    h_x = float(h_item.get("x_percent", 50.0))
                    h_y = float(h_item.get("y_percent", 50.0))
                    h_x = max(0.0, min(100.0, h_x))
                    h_y = max(0.0, min(100.0, h_y))
                    h_id = str(h_item.get("id") or f"hs-{h_idx + 1}")
                    h_name = str(h_item.get("name") or f"Punto de Inspección {h_idx + 1}")
                    parsed_hotspots.append(
                        ARHotspot(
                            id=h_id,
                            name=h_name,
                            x_percent=h_x,
                            y_percent=h_y,
                            category=str(h_item.get("category", "Componente")),
                            description=str(h_item.get("description", "")),
                            voltage_or_spec=str(h_item.get("voltage_or_spec")) if h_item.get("voltage_or_spec") else None,
                            action_guidance=str(h_item.get("action_guidance")) if h_item.get("action_guidance") else None,
                            status=str(h_item.get("status", "NORMAL"))
                        )
                    )
                except (ValueError, TypeError):
                    continue

            targets.append(
                ARTarget(
                    label=item.get("label", f"{brand} {model}".strip()),
                    confidence=float(item.get("confidence", 0.9)),
                    bbox=(max(0, x), max(0, y), max(10, bw), max(10, bh)),
                    category=cat,
                    details=det,
                    action_hint=item.get("action_hint", "Consulta la ficha técnica del producto"),
                    brand=brand,
                    model_number=model,
                    serial_number=item.get("serial_number"),
                    manual=manual_obj,
                    hotspots=parsed_hotspots
                )
            )

        return SceneAnalysisResult(
            summary=data.get("summary", "Análisis multimodal completado exitosamente."),
            targets=targets,
            timestamp=timestamp,
            engine_used="gemini-vision"
        )
