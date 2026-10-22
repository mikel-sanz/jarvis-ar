"""Módulo de captura de cámara física para OpenCV.

Propósito:
    Proveer un gestor seguro y robusto para dispositivos de captura de video
    con soporte para context manager (`with`), previniendo fugas de memoria
    y bloqueos de controladores en Windows.

Ejemplo de uso:
    >>> from src.camera import CameraManager
    >>> with CameraManager(camera_index=0) as cam:
    ...     success, frame = cam.read_frame()
    ...     if success:
    ...         print(frame.shape)
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class CameraManager:
    """Gestor de ciclo de vida de captura de video con OpenCV."""

    def __init__(self, camera_index: int = 0) -> None:
        """Inicializa la referencia a la cámara.

        Args:
            camera_index: Índice del dispositivo de video en el sistema operativo.
        """
        self._camera_index: int = camera_index
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "CameraManager":
        """Abre la cámara de forma segura usando el backend DirectShow en Windows.

        Returns:
            CameraManager: Instancia activa del gestor.
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Libera los recursos del dispositivo al salir del contexto."""
        self.release()

    def open(self) -> bool:
        """Inicializa y configura el dispositivo de captura de video.

        Returns:
            bool: True si la cámara se abrió exitosamente, False en caso contrario.
        """
        if self._cap is not None and self._cap.isOpened():
            return True

        try:
            # En Windows cv2.CAP_DSHOW acelera el arranque y evita bloqueos de MSMF
            self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
            if self._cap is None or not self._cap.isOpened():
                # Fallback al backend por defecto
                self._cap = cv2.VideoCapture(self._camera_index)
        except (cv2.error, RuntimeError, OSError):
            self._cap = None
            return False

        if self._cap is not None and self._cap.isOpened():
            try:
                # Configurar resolución recomendada de 1280x720 para equilibrio rendimiento/calidad
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            except (cv2.error, RuntimeError, OSError):
                pass
            return True

        return False

    def is_opened(self) -> bool:
        """Verifica si la cámara está activa y disponible.

        Returns:
            bool: Estado de apertura del dispositivo.
        """
        try:
            return self._cap is not None and self._cap.isOpened()
        except (cv2.error, RuntimeError, OSError):
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Captura un fotograma del dispositivo.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (éxito, fotograma_bgr).
        """
        if not self.is_opened() or self._cap is None:
            return False, None

        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return False, None
            return True, frame
        except (cv2.error, RuntimeError, OSError):
            return False, None

    def release(self) -> None:
        """Libera el descriptor del dispositivo."""
        if self._cap is not None:
            try:
                self._cap.release()
            except (cv2.error, RuntimeError, OSError):
                pass
            finally:
                self._cap = None
