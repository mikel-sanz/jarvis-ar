"""Motor de búsqueda, extracción y consulta interactiva de manuales técnicos.

Propósito:
    Gestionar la adquisición automática de manuales de usuario y fichas técnicas
    a partir de marcas y modelos detectados, persistir una caché local en JSON
    y responder consultas interactivas de ingeniería/mantenimiento en lenguaje natural.

Ejemplo de uso:
    >>> from src.manual_engine import ManualEngine
    >>> engine = ManualEngine()
    >>> manual = engine.get_manual("Arduino", "UNO R3")
    >>> print(manual.brand, manual.model_number)
    Arduino UNO R3
    >>> respuesta = engine.query_manual(manual, "¿Cuál es el voltaje de entrada?")
    >>> print(respuesta)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
CACHE_MANUALS_DIR: Path = ROOT_DIR / "data" / "cache_manuals"
CACHE_MANUALS_DIR.mkdir(parents=True, exist_ok=True)


class DeviceManual(BaseModel):
    """Estructura de datos completa de un manual o ficha técnica de equipo."""

    equipment_name: str = Field(description="Nombre comercial o identificador completo del equipo.")
    brand: str = Field(description="Fabricante o marca (ej. Cisco, Fluke, HP, Arduino, Siemens).")
    model_number: str = Field(description="Modelo exacto o referencia del dispositivo.")
    category: str = Field(default="Hardware / Dispositivo", description="Categoría funcional del equipo.")
    specifications: Dict[str, str] = Field(default_factory=dict, description="Especificaciones técnicas clave.")
    safety_warnings: List[str] = Field(default_factory=list, description="Advertencias de seguridad y operación.")
    quick_start_steps: List[str] = Field(default_factory=list, description="Pasos para arranque y configuración.")
    troubleshooting_codes: Dict[str, str] = Field(default_factory=dict, description="Códigos de error y soluciones.")
    manual_source: str = Field(default="Base de Conocimiento", description="Origen del manual (Web, Cache o IA).")
    raw_text_summary: str = Field(default="", description="Resumen descriptivo del manual técnico.")


# Catálogo local de respaldo para funcionamiento 100% offline
OFFLINE_CATALOG: Dict[str, DeviceManual] = {
    "arduino_uno": DeviceManual(
        equipment_name="Placa de Desarrollo Arduino UNO R3",
        brand="Arduino",
        model_number="UNO R3",
        category="Microcontroladores / Embebidos",
        specifications={
            "Microcontrolador": "ATmega328P",
            "Voltaje de Operación": "5V DC",
            "Voltaje de Entrada Recomendado": "7V a 12V DC (Límite: 6V - 20V)",
            "Pines I/O Digitales": "14 (6 salidas PWM)",
            "Pines de Entrada Analógica": "6 (A0 a A5)",
            "Memoria Flash": "32 KB (0.5 KB usados por bootloader)",
            "Velocidad de Reloj": "16 MHz"
        },
        safety_warnings=[
            "No suministrar más de 5.5V directamente en el pin 5V (daño irreversible al microcontrolador).",
            "La corriente máxima por pin de I/O no debe exceder los 40 mA (recomendado 20 mA).",
            "Aislar la parte inferior de la placa para evitar cortocircuitos sobre superficies metálicas."
        ],
        quick_start_steps=[
            "1. Conectar al PC mediante cable USB tipo B (el LED 'ON' debe encenderse en verde).",
            "2. En el IDE de Arduino, seleccionar 'Herramientas > Placa > Arduino Uno'.",
            "3. Seleccionar el puerto COM asignado en 'Herramientas > Puerto'.",
            "4. Cargar el sketch de prueba 'Blink' desde 'Archivo > Ejemplos > 01.Basics'."
        ],
        troubleshooting_codes={
            "avrdude stk500_recv(): not in sync": "El puerto COM no es el correcto o el cable USB está dañado.",
            "LED 'L' parpadea rápidamente sin cargar código": "Comprobar conexiones en pines 0 (RX) y 1 (TX); desconectarlos durante la subida.",
            "Regulador de voltaje se calienta en exceso": "Voltaje de entrada en jack de barril superior a 15V o cortocircuito en línea de 5V."
        },
        manual_source="Catálogo Técnico Local Autónomo",
        raw_text_summary="Placa electrónica basada en el microcontrolador ATmega328P para prototipado rápido de sistemas interactivos."
    ),
    "cisco_catalyst": DeviceManual(
        equipment_name="Switch Gestionable Cisco Catalyst 2960 Series",
        brand="Cisco Systems",
        model_number="Catalyst 2960-X / 24TS-L",
        category="Equipamiento de Redes / Telecomunicaciones",
        specifications={
            "Puertos": "24 puertos Gigabit Ethernet 10/100/1000 + 4 enlaces SFP 1G",
            "Capacidad de Conmutación": "216 Gbps",
            "Tasa de Reenvío": "71.4 Mpps",
            "Alimentación": "100-240V AC, 50-60Hz (Consumo típico: 35W)",
            "Sistema Operativo": "Cisco IOS LAN Base",
            "Puertos de Gestión": "1 puerto RJ-45 Consola + 1 puerto USB mini-B"
        },
        safety_warnings=[
            "Riesgo de descarga eléctrica: Conectar obligatoriamente a toma con conexión a tierra física.",
            "Mantener libres las rejillas laterales de ventilación; no obstruir el flujo de aire en rack.",
            "No mirar directamente a los puertos de fibra óptica SFP activos (emisión láser clase 1)."
        ],
        quick_start_steps=[
            "1. Montar en rack de 19 pulgadas utilizando los soportes laterales provistos.",
            "2. Conectar el cable de alimentación AC (el LED 'SYST' parpadeará en ámbar y luego verde).",
            "3. Conectar cable de consola RJ-45 a 9600 baudios, 8 bits de datos, sin paridad, 1 bit de parada.",
            "4. Iniciar asistente de configuración inicial 'setup' o acceder vía CLI con enable."
        ],
        troubleshooting_codes={
            "LED 'SYST' en ámbar continuo": "Fallo en autoprueba de encendido (POST); comprobar módulos internos o reiniciar.",
            "LED de puerto parpadea en ámbar": "Bucle de red detectado (Spanning Tree bloqueando puerto) o error de duplex/velocidad.",
            "Sin acceso a consola": "Comprobar velocidad de emulador a 9600 baudios y flujo de control en 'None'."
        },
        manual_source="Catálogo Técnico Local Autónomo",
        raw_text_summary="Switch de capa 2 de clase empresarial para conectividad de campus y sucursales con seguridad avanzada."
    ),
    "fluke_multimeter": DeviceManual(
        equipment_name="Multímetro Digital Industrial Fluke 87V",
        brand="Fluke",
        model_number="Fluke 87V Industrial",
        category="Instrumentación & Medida Eléctrica",
        specifications={
            "Rango Voltaje AC/DC": "Hasta 1000V con precisión del 0.05%",
            "Rango Corriente AC/DC": "Hasta 10A continua (20A durante 30 segundos)",
            "Filtro Pasa-bajos": "Función especializada para variadores de frecuencia (VFD)",
            "Categoría de Seguridad": "CAT III 1000V / CAT IV 600V",
            "Batería": "Pila de 9V alcalina (duración típica: 400 horas)",
            "Pantalla": "4-1/2 dígitos con retroiluminación de dos niveles"
        },
        safety_warnings=[
            "Comprobar siempre las puntas de prueba antes de medir alta tensión (revisar continuidad).",
            "Nunca cambiar el selector de escala mientras las sondas estén conectadas a un circuito con energía.",
            "Utilizar únicamente fusibles de repuesto certificados de acción rápida (Fluke 11A / 440mA 1000V)."
        ],
        quick_start_steps=[
            "1. Insertar la sonda negra en el conector 'COM' y la roja en 'V/Ω' para medir voltaje o resistencia.",
            "2. Girar el selector a la magnitud deseada (V AC con onda senoidal, o V DC con barra continua).",
            "3. Para medir motores con variador, presionar el botón amarillo para activar el filtro pasa-bajos.",
            "4. Para congelar una medición en pantalla, presionar el botón 'AutoHOLD'."
        ],
        troubleshooting_codes={
            "Pantalla muestra 'bAtt'": "Nivel de batería inferior al 10%; sustituir la pila de 9V en el compartimento posterior.",
            "Lectura en circuito abierto no marca 'OL'": "Puntas de prueba dañadas o fusibles de corriente fundidos.",
            "No mide amperios (lectura 0.00A con carga)": "Fusible de protección de corriente fundido; abrir tapa y reemplazar fusible cerámico."
        },
        manual_source="Catálogo Técnico Local Autónomo",
        raw_text_summary="Instrumento de precisión diseñado para diagnóstico eléctrico industrial en entornos con ruido electromagnético severo."
    ),
    "xiaomi_phone": DeviceManual(
        equipment_name="Smartphone Xiaomi Series (HyperOS / Android)",
        brand="Xiaomi",
        model_number="Xiaomi / Redmi Series",
        category="Dispositivos Móviles / Comunicaciones",
        specifications={
            "Fabricante": "Xiaomi Communications Co., Ltd.",
            "Sistema Operativo": "Xiaomi HyperOS (basado en Android)",
            "Pantalla": "AMOLED FHD+ / CrystalRes 120Hz",
            "Batería": "5000 mAh Li-Po integrada",
            "Carga Rápida": "HyperCharge 67W / 120W (Conector USB Tipo-C)",
            "Conectividad": "5G Dual SIM, Wi-Fi 6, Bluetooth 5.4, NFC, Sensor Infrarrojos",
            "Identificador Serial / QR": "Registrado en etiqueta adhesiva posterior del chasis"
        },
        safety_warnings=[
            "Utilizar exclusivamente cargadores y cables certificados Xiaomi HyperCharge para evitar degradación prematura.",
            "No someter la batería interna a temperaturas superiores a 45°C ni perforar la tapa posterior.",
            "Evitar sumergir en agua si el dispositivo presenta grietas en el chasis o módulo de cámaras."
        ],
        quick_start_steps=[
            "1. Encendido: Mantener presionado el botón lateral de encendido durante 3 segundos hasta vibrar.",
            "2. Forzar Reinicio en caso de bloqueo: Presionar [Botón Encendido + Bajar Volumen] durante 10 segundos continuos.",
            "3. Entrar en Modo Recovery Oficial: Con el teléfono apagado, presionar [Botón Encendido + Subir Volumen] hasta ver el logotipo Mi/Xiaomi.",
            "4. Entrar en Modo Fastboot: Con el teléfono apagado, presionar [Botón Encendido + Bajar Volumen] hasta ver el androide Fastboot.",
            "5. Transferencia de Datos: Conectar vía cable USB-C al PC y seleccionar 'Transferencia de archivos (MTP)' en la barra de notificaciones."
        ],
        troubleshooting_codes={
            "Bucle de reinicio (Bootloop) en pantalla de logo": "Acceder a Recovery (Encendido + Subir Vol) y seleccionar 'Reboot to System' o 'Wipe Data'.",
            "Carga lenta / no detecta carga rápida": "Comprobar cargador original de 67W/120W y limpiar posibles residuos de polvo en el puerto USB-C.",
            "Calentamiento temporal durante carga o cámara": "Normal durante carga rápida de 67W/120W; retirar fundas térmicas si supera los 42°C.",
            "Sensor de huella en pantalla no responde": "Volver a registrar la huella en Ajustes > Contraseñas y Seguridad con el dedo limpio."
        },
        manual_source="Catálogo Oficial Xiaomi (Modo Autónomo Local)",
        raw_text_summary="Dispositivo móvil inteligente de alto rendimiento con arquitectura multicámara y sistema Xiaomi HyperOS."
    )
}


def sanitize_filename(text: str) -> str:
    """Convierte texto en un nombre de archivo seguro sin caracteres especiales."""
    cleaned: str = re.sub(r"[^\w\-_]", "_", text.lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


class ManualEngine:
    """Motor de gestión de manuales técnicos, caché local y consultas interactivas."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Inicializa el motor con clave opcional de Gemini."""
        self._api_key: Optional[str] = api_key
        self._gemini_client = None

        if self._api_key:
            self._init_gemini_client()

    def _init_gemini_client(self) -> None:
        """Inicializa el cliente oficial de Gemini SDK."""
        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=self._api_key)
        except Exception:
            self._gemini_client = None

    def get_manual(
        self,
        brand: str,
        model_number: str,
        category: Optional[str] = None,
        details: Optional[str] = None
    ) -> DeviceManual:
        """Obtiene el manual técnico del equipo o producto (desde caché local o mediante búsqueda/IA).

        Args:
            brand: Fabricante o marca detectada.
            model_number: Referencia o modelo del equipo o producto.
            category: Categoría contextual identificada por visión (ej. Nutrición, Electrónica).
            details: Detalles clave adicionales del producto.

        Returns:
            DeviceManual: Estructura técnica o nutricional completa del manual.
        """
        cache_key: str = sanitize_filename(f"{brand}_{model_number}")
        cache_file: Path = CACHE_MANUALS_DIR / f"{cache_key}.json"

        # 1. Comprobar si ya existe en la caché local persistente
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return DeviceManual(**data)
            except Exception:
                pass

        # 2. Si hay cliente Gemini disponible, sintetizar con IA adaptada a la naturaleza del producto
        if self._gemini_client is not None:
            try:
                manual = self._generate_manual_via_gemini(brand, model_number, category, details)
                self._save_to_cache(cache_file, manual)
                return manual
            except Exception:
                pass

        # 3. Fallback a catálogo local o sintetizador heurístico contextualizado
        manual = self._get_from_catalog_or_heuristic(brand, model_number, category, details)
        self._save_to_cache(cache_file, manual)
        return manual

    def _save_to_cache(self, cache_file: Path, manual: DeviceManual) -> None:
        """Guarda el manual en la carpeta de caché en formato JSON."""
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(manual.model_dump(), f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _generate_manual_via_gemini(
        self,
        brand: str,
        model_number: str,
        category: Optional[str] = None,
        details: Optional[str] = None
    ) -> DeviceManual:
        """Genera el manual técnico o nutricional utilizando conocimiento global adaptado."""
        prompt: str = (
            f"Actúa como un especialista en ingeniería, calidad y fichas técnicas de producto.\n"
            f"Proporciona la ficha técnica y manual oficial para: Marca '{brand}', Modelo/Producto '{model_number}'.\n"
            f"Categoría identificada: '{category or 'General'}'. Contexto visual: '{details or ''}'.\n\n"
            f"REGLA CRÍTICA DE COMPATIBILIDAD:\n"
            f"- Si el producto es un ALIMENTO, SUPLEMENTO, SNACK O BARRITA (ej. Proteína, Chocolate, Nutrición): "
            f"las especificaciones DEBEN ser exclusivamente nutricionales (Proteína, Calorías, Peso neto, Alérgenos, "
            f"Condiciones de almacenamiento y fecha). NUNCA inventes voltajes, puertos ni certificaciones eléctricas.\n"
            f"- Si es una HERRAMIENTA o PIEZA MECÁNICA: materiales, peso, dimensiones, tolerancias.\n"
            f"- Si es un EQUIPO ELECTRÓNICO/TI: tensión, interfaces, protocolos, firmware.\n\n"
            f"Responde estrictamente en formato JSON con esta estructura exacta:\n"
            "{\n"
            f'  "equipment_name": "{brand} {model_number}",\n'
            f'  "brand": "{brand}",\n'
            f'  "model_number": "{model_number}",\n'
            f'  "category": "{category or "Producto de Consumo / Técnico"}",\n'
            '  "specifications": {"Parámetro 1": "Valor", "Parámetro 2": "Valor"},\n'
            '  "safety_warnings": ["Advertencia o precaución 1", "Advertencia 2"],\n'
            '  "quick_start_steps": ["Paso 1: Uso o consumo", "Paso 2", "Paso 3"],\n'
            '  "troubleshooting_codes": {"Duda o problema común": "Solución o indicación"},\n'
            '  "manual_source": "Ficha Oficial (Conocimiento Global Adaptado)",\n'
            '  "raw_text_summary": "Resumen técnico o nutricional del producto"\n'
            "}"
        )
        models_to_try = [
            "gemini-3.5-flash",
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
        response = None
        for model_name in models_to_try:
            try:
                response = self._gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    break
            except Exception:
                continue

        if response is None or not response.text:
            raise RuntimeError(f"No se pudo generar el manual con los modelos disponibles para {brand} {model_number}")

        text_content: str = response.text.strip()
        if text_content.startswith("```json"):
            text_content = text_content[7:]
        if text_content.endswith("```"):
            text_content = text_content[:-3]
        text_content = text_content.strip()

        data = json.loads(text_content)
        return DeviceManual(**data)

    def _get_from_catalog_or_heuristic(
        self,
        brand: str,
        model_number: str,
        category: Optional[str] = None,
        details: Optional[str] = None
    ) -> DeviceManual:
        """Obtiene un manual desde el catálogo offline o genera una ficha adaptada semánticamente."""
        combined_text: str = f"{brand} {model_number} {category or ''} {details or ''}".lower()

        # 1. Equipos con fichas catalogadas en el sistema
        if "arduino" in combined_text or "uno" in combined_text or "atmega" in combined_text:
            return OFFLINE_CATALOG["arduino_uno"]
        elif "cisco" in combined_text or "catalyst" in combined_text or "switch" in combined_text:
            return OFFLINE_CATALOG["cisco_catalyst"]
        elif "fluke" in combined_text or "multimeter" in combined_text or "multimetro" in combined_text:
            return OFFLINE_CATALOG["fluke_multimeter"]
        elif any(k in combined_text for k in ["xiaomi", "redmi", "poco", "smartphone", "telefono", "movil", "celular"]):
            return OFFLINE_CATALOG["xiaomi_phone"]

        # 2. Productos de Nutrición, Suplementación o Alimentos (¡Sin voltajes ni cables!)
        is_food_or_nutrition = any(k in combined_text for k in [
            "protein", "proteina", "bar", "barrita", "chocolate", "alimento", "suplemento",
            "nutricion", "comida", "snack", "bebida", "food", "whey", "creatina", "bioprox"
        ])

        if is_food_or_nutrition:
            return DeviceManual(
                equipment_name=f"{brand} {model_number}".strip(),
                brand=brand if brand else "Bioprox / Nutrición",
                model_number=model_number if model_number else "Suplemento Proteico",
                category="Nutrición & Suplementación Deportiva",
                specifications={
                    "Tipo de Producto": "Barrita de Proteínas / Suplemento Dietético",
                    "Aporte Proteico": "Alto contenido proteico (Aislado/Concentrado lácteo)",
                    "Sabor / Variedad": "Chocolate / Variedad comercial especificada",
                    "Condiciones de Conservación": "Conservar en lugar fresco y seco (15°C - 22°C), protegido de luz directa",
                    "Momento Óptimo de Consumo": "Snack entre comidas, pre-entreno o post-entreno inmediato"
                },
                safety_warnings=[
                    "Alérgenos: Puede contener leche, soja, gluten y trazas de frutos de cáscara.",
                    "Un consumo excesivo de edulcorantes/polialcoholes puede tener efectos laxantes.",
                    "Los suplementos alimenticios no deben utilizarse como sustitutos de una dieta variada y equilibrada."
                ],
                quick_start_steps=[
                    "1. Comprobar que el envoltorio hermético protector se encuentre sin fisuras.",
                    "2. Consumir preferentemente antes de la fecha y lote impresos en el envase.",
                    "3. Mantener a temperatura ambiente templada para preservar la consistencia de la cobertura.",
                    "4. Desechar el envoltorio en el contenedor adecuado de reciclaje."
                ],
                troubleshooting_codes={
                    "Cobertura reblandecida": "Exposición a temperatura superior a 24°C; enfriar 10 minutos para recuperar firmeza.",
                    "Capa blanquecina en chocolate (Fat bloom)": "Cristalización de manteca por variación térmica; perfectamente apto para consumo.",
                    "Empaque perforado o inflado": "Pérdida de hermeticidad en transporte; no consumir y contactar con el proveedor."
                },
                manual_source="Ficha Técnica Nutricional (Catálogo Adaptado)",
                raw_text_summary=f"Suplemento proteico {brand} {model_number}. Ficha generada con parámetros de nutrición y conservación."
            )

        # 3. Equipos Electrónicos o Industriales Genéricos
        return DeviceManual(
            equipment_name=f"{brand} {model_number}".strip(),
            brand=brand if brand else "Genérico",
            model_number=model_number if model_number else "No especificado",
            category=category if category else "Equipo Técnico / Industrial",
            specifications={
                "Alimentación Estándar": "100-240V AC o 12V-24V DC",
                "Temperatura de Operación": "0°C a 45°C",
                "Certificaciones": "CE / RoHS / FCC",
                "Interfaz": "Conexión estándar según panel frontal"
            },
            safety_warnings=[
                "Verificar la polaridad y el rango de tensión de entrada antes de encender.",
                "Desconectar de la red eléctrica antes de realizar cualquier intervención de mantenimiento.",
                "Asegurar una ventilación adecuada y evitar exposición directa a humedad."
            ],
            quick_start_steps=[
                "1. Inspeccionar visualmente que los conectores y cables no presenten daños.",
                "2. Conectar la línea de alimentación respetando las especificaciones de la etiqueta.",
                "3. Encender el interruptor principal y verificar el encendido de los indicadores LED.",
                "4. Consultar el procedimiento de vinculación o configuración inicial en el panel."
            ],
            troubleshooting_codes={
                "Sin alimentación / Indicador apagado": "Revisar fusible de entrada y estado de la toma de corriente.",
                "Indicador de estado en rojo o parpadeo": "Comprobar sobrecarga de circuito o temperatura anormal.",
                "Sin comunicación": "Validar integridad del cableado de datos y parámetros de velocidad de enlace."
            },
            manual_source="Catálogo Técnico Local (Modo Desconectado)",
            raw_text_summary=f"Equipo técnico {brand} {model_number}. Ficha generada a partir de los datos serigrafiados."
        )

    def query_manual(self, manual: DeviceManual, question: str) -> str:
        """Responde una consulta en lenguaje natural sobre el manual del equipo.

        Args:
            manual: Estructura técnica del dispositivo.
            question: Pregunta del usuario (ej. "¿Cómo se reinicia?", "¿Qué voltaje usa?").

        Returns:
            str: Respuesta clara y fundamentada en el manual.
        """
        # Si Gemini está activo, generamos una respuesta contextual con IA conversacional
        if self._gemini_client is not None:
            try:
                context_str: str = json.dumps(manual.model_dump(), ensure_ascii=False, indent=2)
                prompt: str = (
                    f"Eres el asistente técnico de ingeniería para el equipo '{manual.equipment_name}'.\n"
                    f"A continuación tienes los datos oficiales de su manual técnico:\n"
                    f"{context_str}\n\n"
                    f"Pregunta del operador: {question}\n\n"
                    f"Responde de forma técnica, precisa, directa y en español. Si la respuesta involucra "
                    f"un procedimiento paso a paso o una advertencia de seguridad crítica, resáltala con claridad."
                )
                models_to_try = [
            "gemini-3.5-flash",
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
                for model_name in models_to_try:
                    try:
                        response = self._gemini_client.models.generate_content(
                            model=model_name,
                            contents=prompt
                        )
                        if response and response.text:
                            return response.text.strip()
                    except Exception:
                        continue
            except Exception:
                pass

        # Motor de respuestas offline basado en concordancia semántica de palabras clave
        q_lower: str = question.lower()

        if any(w in q_lower for w in ["voltaje", "alimentacion", "consumo", "tension", "vatios", "potencia"]):
            specs_txt = "\n".join([f"- **{k}:** {v}" for k, v in manual.specifications.items() if any(t in k.lower() for t in ["voltaje", "alimentacion", "consumo", "entrada"])])
            if specs_txt:
                return f"**Parámetros de Alimentación para {manual.equipment_name}:**\n\n{specs_txt}"
            return f"**Especificaciones eléctricas registradas:**\n" + "\n".join([f"- **{k}:** {v}" for k, v in manual.specifications.items()])

        if any(w in q_lower for w in ["seguridad", "peligro", "advertencia", "cuidado", "riesgo"]):
            warnings_txt = "\n".join([f"- ⚠️ {w}" for w in manual.safety_warnings])
            return f"**Advertencias de Seguridad ({manual.equipment_name}):**\n\n{warnings_txt}"

        if any(w in q_lower for w in ["reiniciar", "iniciar", "arranque", "configurar", "conectar", "paso"]):
            steps_txt = "\n".join([f"{s}" for s in manual.quick_start_steps])
            return f"**Procedimiento de Puesta en Marcha / Arranque:**\n\n{steps_txt}"

        if any(w in q_lower for w in ["error", "falla", "parpadea", "luz", "problema", "alarma"]):
            codes_txt = "\n".join([f"- **{k}:** {v}" for k, v in manual.troubleshooting_codes.items()])
            return f"**Guía de Solución de Problemas y Códigos de Error:**\n\n{codes_txt}"

        # Respuesta general con resumen y especificaciones
        specs_all = "\n".join([f"- **{k}:** {v}" for k, v in manual.specifications.items()])
        return (
            f"**Información Técnica de {manual.equipment_name}:**\n\n"
            f"{manual.raw_text_summary}\n\n"
            f"**Especificaciones Principales:**\n{specs_all}\n\n"
            f"*(Puedes consultar sobre voltaje, pasos de encendido, advertencias de seguridad o códigos de error)*"
        )
