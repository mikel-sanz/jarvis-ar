# 🗺️ Mapa de Arquitectura: Jarvis AR (Spatial Intelligence)

**Versión:** 1.0.0  
**Ecosistema:** NEXO Framework v2.3  
**Stack Tecnológico:** Python 3.12, Streamlit, OpenCV, Google GenAI SDK, Pydantic, Pytest

---

## 🏛️ Flujo General de Datos y Pipeline de Inferencia

```mermaid
sequenceDiagram
    autonumber
    actor Usuario as Usuario (Móvil / Web)
    participant UI as Streamlit Web App (src/app.py)
    participant Camera as CameraManager / Web API (src/camera.py)
    participant AI as VisionEngine (src/vision_ai.py)
    participant AR as AREngine (src/ar_engine.py)

    Usuario->>UI: Accede a http://<IP>:8501 y activa cámara
    UI->>Camera: Captura fotograma (JPG/PNG bytes)
    Camera-->>UI: Retorna buffer de imagen
    UI->>AI: analyze_frame(frame_bgr)
    alt Clave Gemini configurada y conexión activa
        AI->>AI: _analyze_with_gemini()
        Note over AI: Extracción de targets, coordenadas y contexto IA
    else Modo Local / Fallback offline
        AI->>AI: _analyze_locally()
        Note over AI: Detección Canny + contornos geométricos OpenCV
    end
    AI-->>UI: Retorna SceneAnalysisResult (targets, resumen, metadatos)
    UI->>AR: render_overlay(frame_bgr, targets, telemetry)
    AR->>AR: Dibuja HUD, retículas, esquinas y tarjetas holográficas
    AR-->>UI: Retorna frame con overlay AR en alta definición
    UI-->>Usuario: Despliega visor interactivo + tarjetas expandibles en pantalla
```

---

## 🧩 Componentes y Responsabilidades

| Componente | Archivo | Responsabilidad Principal |
| :--- | :--- | :--- |
| **Configuración Segura** | [`src/config.py`](file:///c:/Users/User/Documents/Realidad_AI/src/config.py) | Carga tipada de `.env`, validación de puertos y rutas con `pathlib.Path`. |
| **Captura de Cámara** | [`src/camera.py`](file:///c:/Users/User/Documents/Realidad_AI/src/camera.py) | Context manager para `cv2.VideoCapture` con backend DirectShow y release garantizado. |
| **Motor de IA Multimodal** | [`src/vision_ai.py`](file:///c:/Users/User/Documents/Realidad_AI/src/vision_ai.py) | Análisis multimodal con Gemini API, detección de marcas/modelos y visión artificial. |
| **Motor de Manuales Técnicos** | [`src/manual_engine.py`](file:///c:/Users/User/Documents/Realidad_AI/src/manual_engine.py) | Búsqueda, extracción web, persistencia en JSON y asistente de consultas interactivas. |
| **Motor AR (HUD Holográfico)** | [`src/ar_engine.py`](file:///c:/Users/User/Documents/Realidad_AI/src/ar_engine.py) | Renderizado de retículas sci-fi, corchetes de seguimiento y tarjetas flotantes semitransparentes. |
| **Interfaz de Usuario Web** | [`src/app.py`](file:///c:/Users/User/Documents/Realidad_AI/src/app.py) | Frontend Streamlit con visor AR, visor WebRTC y asistente de chat del manual técnico. |
| **Auditor de Capa 0** | [`nexo_audit.py`](file:///c:/Users/User/Documents/Realidad_AI/nexo_audit.py) | Notario de integridad arquitectónica, SAST y validación de testing automático. |

---

## 🛡️ Protocolo de Seguridad y Resiliencia

1. **Aislamiento de Secretos:** Las API Keys nunca se persisten en código; se inyectan en tiempo de ejecución vía `.env` o variables de entorno del sistema.
2. **Cero-Crasheos por Falta de Conexión:** Si el servicio externo de Gemini no responde o no hay API Key, el sistema continúa funcionando ininterrumpidamente con el motor analítico local.
3. **Manejo Específico de Excepciones:** No se capturan excepciones genéricas sin justificación; se manejan fallos concretos (`ValueError`, `ValidationError`, `OSError`).
