# 👁️ Jarvis AR — Sistema de Fusión: Realidad Aumentada + Inteligencia Artificial

[![NEXO Framework](https://img.shields.io/badge/NEXO-v2.3_Certified-00f0ff.svg)](file:///c:/Users/User/Documents/Realidad_AI/.nexo_status.json)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Mobile--First-green.svg)](file:///c:/Users/User/Documents/Realidad_AI/architecture-map.md)

**Jarvis AR** es una plataforma de inteligencia espacial que combina **Realidad Aumentada (AR)** e **Inteligencia Artificial Multimodal (AI)**. Permite apuntar con la cámara de un teléfono inteligente o PC hacia cualquier objeto del entorno para desplegar en tiempo real retículas holográficas, diagnósticos técnicos y tarjetas interactivas de ayuda paso a paso.

---

## 🛡️ Certificación NEXO (Dashboard de Capa 0)

| Protocolo | Fuente de Verdad | Estado |
| :--- | :--- | :---: |
| **Sincronización de Entorno** | [`.nexo/env_state.json`](file:///c:/Users/User/Documents/Realidad_AI/.nexo/env_state.json) | ✅ |
| **Seguridad (SCA / pip-audit)** | [`nexo_audit.py`](file:///c:/Users/User/Documents/Realidad_AI/nexo_audit.py) | ✅ |
| **Calidad de Código (PEP 8)** | Type Hints & Linting estricto | ✅ |
| **Manejo de Errores Global** | Bloques defensivos `try-except` | ✅ |

---

## 📱 ¿Cómo usarlo desde el Teléfono Móvil?

1. Conecta tu teléfono a la **misma red Wi-Fi** que tu ordenador.
2. Inicia la aplicación en el ordenador (ver sección [Inicio Rápido](#-inicio-rápido)).
3. En la consola del servidor o en la barra lateral de la aplicación, verás tu dirección IP local:
   ```bash
   http://192.168.1.X:8501
   ```
4. Abre esa dirección en el navegador web de tu móvil (Chrome, Safari, Edge).
5. Concede permisos de cámara al navegador y captura los objetos a tu alrededor para ver el visor holográfico AR.

---

## ⚡ Inicio Rápido

### 1. Prerrequisitos
- Python 3.12+ instalado en el equipo.
- Acceso a cámara web o smartphone.

### 2. Activación del Entorno Virtual e Instalación
```powershell
# En Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configuración de Credenciales (Opcional)
El proyecto incluye un motor analítico local autónomo que funciona sin conexión ni claves. Si deseas activar la inferencia multimodal de **Google Gemini**:
1. Copia `.env.example` a `.env`:
   ```powershell
   copy .env.example .env
   ```
2. Agrega tu clave en el archivo `.env`:
   ```env
   GEMINI_API_KEY=tu_api_key_aqui
   ```

### 4. Lanzar la Aplicación
```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app.py
```

---

## 🧪 Pruebas Unitarias y Auditoría de Seguridad

El proyecto incluye una suite completa de pruebas unitarias bajo el patrón **AAA (Arrange, Act, Assert)** y auditorías de seguridad:

```powershell
# 1. Ejecutar tests unitarios:
.\.venv\Scripts\python.exe -m pytest tests/ -v

# 2. Auditoría integral de Capa 0 de NEXO:
.\.venv\Scripts\python.exe nexo_audit.py

# 3. Auditoría de vulnerabilidades de librerías (CVEs):
.\.venv\Scripts\python.exe -m pip_audit
```

---

## 🏛️ Mapa de Módulos

- [`src/config.py`](file:///c:/Users/User/Documents/Realidad_AI/src/config.py): Gestión de configuración con Pydantic y Pathlib.
- [`src/vision_ai.py`](file:///c:/Users/User/Documents/Realidad_AI/src/vision_ai.py): Inferencia multimodal híbrida (Gemini Vision + Local OpenCV).
- [`src/ar_engine.py`](file:///c:/Users/User/Documents/Realidad_AI/src/ar_engine.py): Renderizado de overlays HUD, tarjetas flotantes y corchetes sci-fi.
- [`src/camera.py`](file:///c:/Users/User/Documents/Realidad_AI/src/camera.py): Abstracción segura de cámara con context manager.
- [`src/app.py`](file:///c:/Users/User/Documents/Realidad_AI/src/app.py): Interfaz web Streamlit responsiva para móvil y PC.
