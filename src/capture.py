"""Kamera-Zugriff mit einem Hintergrundthread und latest-frame-wins."""

from __future__ import annotations

import threading
import time
from typing import Any

from .config import Config


class CameraCapture:
    """Liest die Kamera fortlaufend; :meth:`get_frame` liefert nur das neueste Bild.

    ``get_frame()`` ist standardmäßig nicht blockierend und gibt ``None`` zurück,
    wenn noch kein Bild vorliegt. Mit einem positiven ``timeout_s`` wartet es
    höchstens diese Zeit auf das erste bzw. nächste verfügbare Bild.
    Zur Vermeidung einer 4K-Kopie wird das zurückgegebene Frame nicht kopiert;
    Aufrufer behandeln es als unveränderlich.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._condition = threading.Condition()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._camera: Any | None = None
        self._latest_frame: Any | None = None
        self._frame_sequence = 0
        self._delivered_sequence = 0
        self._failure: Exception | None = None

    def start(self) -> None:
        """Öffnet die Kamera und startet den Reader-Thread."""

        if self._thread is not None:
            raise RuntimeError("Kamera-Thread wurde bereits gestartet.")

        import cv2

        camera = cv2.VideoCapture(self._config.camera.index)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError("Kamera kann nicht geöffnet werden.")
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._config.camera.fourcc))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.camera.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.camera.height)
        self._camera = camera
        self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self._thread.start()

    def get_frame(self, timeout_s: float = 0.0) -> Any | None:
        """Gibt das aktuellste Frame zurück, optional nach kurzer Wartezeit."""

        if timeout_s < 0:
            raise ValueError("timeout_s darf nicht negativ sein.")
        with self._condition:
            deadline = time.monotonic() + timeout_s
            while self._frame_sequence == self._delivered_sequence and self._failure is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._failure is not None:
                raise RuntimeError("Kamera-Thread ist fehlgeschlagen.") from self._failure
            self._delivered_sequence = self._frame_sequence
            return self._latest_frame

    def stop(self) -> None:
        """Beendet den Reader und gibt das Kameragerät frei."""

        self._stop_requested.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        self._thread = None

    def _reader(self) -> None:
        assert self._camera is not None
        try:
            while not self._stop_requested.is_set():
                ok, frame = self._camera.read()
                if not ok:
                    raise RuntimeError("Kamera liefert kein Frame.")
                with self._condition:
                    self._latest_frame = frame
                    self._frame_sequence += 1
                    self._condition.notify_all()
        except Exception as error:  # noqa: BLE001 - Fehler gehen kontrolliert an die Hauptschleife.
            with self._condition:
                self._failure = error
                self._condition.notify_all()
