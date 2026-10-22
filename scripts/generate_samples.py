"""Generador de muestras visuales de prueba para el Asistente Jarvis AR.

Propósito:
    Generar fotogramas sintéticos estilizados que simulan objetos reales
    (componente electrónico, taza de café, documento técnico) para que el
    usuario pueda probar inmediatamente el motor de Realidad Aumentada
    incluso si su cámara está ocupada o bloqueada por el navegador.
"""

from pathlib import Path
import cv2
import numpy as np

# Rutas del proyecto
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
SAMPLES_DIR: Path = ROOT_DIR / "assets" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def generate_gadget_sample() -> Path:
    """Crea una imagen de un módulo electrónico/tecnológico con patrones de circuito."""
    w, h = 1024, 768
    img = np.full((h, w, 3), (30, 32, 35), dtype=np.uint8)

    # Placa base verde oscuro
    cv2.rectangle(img, (200, 180), (824, 580), (20, 60, 30), -1)
    cv2.rectangle(img, (200, 180), (824, 580), (0, 200, 100), 3)

    # Chip central integrado (Microcontrolador)
    cv2.rectangle(img, (412, 284), (612, 484), (15, 15, 15), -1)
    cv2.rectangle(img, (412, 284), (612, 484), (180, 180, 180), 2)
    cv2.putText(img, "ARM CORTEX-M4", (430, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

    # Pistas de circuito
    for y_offset in range(220, 560, 40):
        cv2.line(img, (210, y_offset), (400, y_offset), (0, 220, 255), 2)
        cv2.circle(img, (210, y_offset), 4, (0, 255, 255), -1)

    for y_offset in range(220, 560, 40):
        cv2.line(img, (620, y_offset), (810, y_offset), (0, 220, 255), 2)
        cv2.circle(img, (810, y_offset), 4, (0, 255, 255), -1)

    target_path = SAMPLES_DIR / "gadget.jpg"
    cv2.imwrite(str(target_path), img)
    return target_path


def generate_mug_sample() -> Path:
    """Crea una imagen de una taza de café en una mesa de trabajo."""
    w, h = 1024, 768
    img = np.full((h, w, 3), (210, 200, 190), dtype=np.uint8)

    # Textura de mesa de madera
    cv2.rectangle(img, (0, 400), (w, h), (140, 110, 80), -1)

    # Taza (cuerpo principal cilíndrico)
    cv2.ellipse(img, (512, 500), (140, 180), 0, 0, 360, (245, 245, 250), -1)
    cv2.ellipse(img, (512, 500), (140, 180), 0, 0, 360, (100, 100, 100), 3)

    # Asa de la taza
    cv2.ellipse(img, (650, 480), (45, 70), 0, 0, 360, (245, 245, 250), 12)

    # Interior con café
    cv2.ellipse(img, (512, 360), (120, 40), 0, 0, 360, (30, 40, 70), -1)

    target_path = SAMPLES_DIR / "desk.jpg"
    cv2.imwrite(str(target_path), img)
    return target_path


def generate_book_sample() -> Path:
    """Crea una imagen de un libro técnico abierto con diagramas."""
    w, h = 1024, 768
    img = np.full((h, w, 3), (50, 45, 40), dtype=np.uint8)

    # Páginas del libro
    cv2.rectangle(img, (150, 160), (500, 620), (240, 238, 230), -1)
    cv2.rectangle(img, (524, 160), (874, 620), (240, 238, 230), -1)

    # Líneas simuladas de texto
    for line_y in range(220, 580, 25):
        cv2.line(img, (180, line_y), (470, line_y), (100, 100, 100), 2)
        cv2.line(img, (554, line_y), (844, line_y), (100, 100, 100), 2)

    # Título
    cv2.putText(img, "SPATIAL COMPUTING & AI", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)

    target_path = SAMPLES_DIR / "book.jpg"
    cv2.imwrite(str(target_path), img)
    return target_path


if __name__ == "__main__":
    generate_gadget_sample()
    generate_mug_sample()
    generate_book_sample()
    print("Muestras sintéticas generadas exitosamente en assets/samples/")
