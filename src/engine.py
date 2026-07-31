"""InsightFace-Wrapper: Detection und Embedding.

Siehe docs/mvp-plan.md, M2.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .gallery import Gallery


@dataclass(frozen=True)
class DetectedFace:
    """Ein erkanntes Gesicht in Koordinaten des ursprünglichen Frames."""

    bbox: tuple[float, float, float, float]
    embedding: Any


class FaceEngine:
    """Dünner Wrapper für lokale InsightFace-Detection und -Recognition."""

    def __init__(self, config: Config) -> None:
        from insightface.app import FaceAnalysis

        self._input_width = config.detection.input_width
        self._app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(
            ctx_id=0,
            det_size=(config.detection.det_size, config.detection.det_size),
        )

    def detect(self, frame: Any) -> list[DetectedFace]:
        """Erkennt Gesichter und rechnet Boxen auf die Eingabeauflösung zurück."""

        import cv2

        if frame is None or len(frame.shape) < 2:
            raise ValueError("Frame hat kein gültiges Bildformat.")

        scale = 1.0
        detection_frame = frame
        if frame.shape[1] > self._input_width:
            scale = self._input_width / frame.shape[1]
            detection_frame = cv2.resize(frame, None, fx=scale, fy=scale)

        detected: list[DetectedFace] = []
        for face in self._app.get(detection_frame):
            x1, y1, x2, y2 = (float(coordinate) / scale for coordinate in face.bbox)
            detected.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    embedding=face.normed_embedding,
                )
            )
        return detected


def _format_box(bbox: tuple[float, float, float, float]) -> str:
    coordinates = (round(coordinate) for coordinate in bbox)
    return "(" + ",".join(str(coordinate) for coordinate in coordinates) + ")"


def _run_debug(image_path: Path) -> int:
    import cv2

    try:
        config = load_config()
        gallery = Gallery.load()
    except ValueError as error:
        print(f"Initialisierungsfehler: {error}")
        return 1

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Bild konnte nicht gelesen werden: {image_path}")
        return 1

    try:
        engine = FaceEngine(config)
        faces = engine.detect(frame)
    except Exception as error:  # noqa: BLE001 - Die Debug-CLI soll keinen Traceback ausgeben.
        print(f"Gesichtserkennung fehlgeschlagen: {error}")
        return 1

    if not faces:
        print("Keine Gesichter erkannt.")
        return 0

    for face in faces:
        try:
            person_id, similarity = gallery.match(face.embedding)
        except ValueError as error:
            print(f"Matching fehlgeschlagen: {error}")
            return 1
        if similarity >= config.recognition.threshold:
            label = f"{person_id} / {gallery.name_for(person_id)}"
        else:
            label = "UNBEKANNT"
        print(f"Box {_format_box(face.bbox)}: {label}; Similarity {similarity:.3f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Führt die M2-Debug-Pipeline für ein einzelnes Bild aus."""

    parser = argparse.ArgumentParser(description="Gesichter in einem Bild erkennen und zuordnen.")
    parser.add_argument("--debug", metavar="BILDPFAD", type=Path, help="Bild für die M2-Debug-Pipeline")
    arguments = parser.parse_args(argv)
    if arguments.debug is None:
        parser.error("--debug BILDPFAD ist erforderlich")
    return _run_debug(arguments.debug)


if __name__ == "__main__":
    raise SystemExit(main())
