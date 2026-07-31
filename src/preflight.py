"""CLI-Systemcheck vor dem Vortrag."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Config, load_config

if TYPE_CHECKING:
    from collections.abc import Sequence


MODEL_DIR = Path.home() / ".insightface" / "models" / "buffalo_l"
EMBEDDINGS_PATH = Path("data/embeddings.npz")
_REQUIRED_MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")
_CAMERA_INDEXES_TO_PROBE = range(6)


def _model_pack_check() -> str:
    missing = [name for name in _REQUIRED_MODEL_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        raise RuntimeError("Modellpack buffalo_l ist nicht vollständig lokal vorhanden.")
    return "Modellpack buffalo_l ist lokal vorhanden"


def _coreml_check() -> str:
    import onnxruntime

    if "CoreMLExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError("CoreMLExecutionProvider ist nicht verfügbar.")
    return "CoreMLExecutionProvider verfügbar"


def _open_camera(config: Config) -> Any:
    import cv2

    camera = cv2.VideoCapture(config.camera.index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Kamera kann nicht geöffnet werden.")
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.camera.fourcc))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    return camera


def _available_camera_indices() -> list[int]:
    """Ermittelt die für die Vor-Ort-Auswahl verfügbaren Kamera-Indizes."""

    import cv2

    available: list[int] = []
    for index in _CAMERA_INDEXES_TO_PROBE:
        camera = cv2.VideoCapture(index)
        try:
            if camera.isOpened():
                available.append(index)
        finally:
            camera.release()
    return available


def _camera_check(config: Config) -> str:
    import cv2

    _model_pack_check()
    available_indices = _available_camera_indices()
    camera = _open_camera(config)
    try:
        width = round(camera.get(3))
        height = round(camera.get(4))
        if (width, height) != (config.camera.width, config.camera.height):
            raise RuntimeError(
                f"Kamera liefert {width}x{height} statt {config.camera.width}x{config.camera.height}."
            )
        for _ in range(10):  # Warmup: Autobelichtung braucht ein paar Frames, sonst ist das Bild schwarz.
            ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Kamera liefert kein Frame.")

        window_name = "Preflight: Rohbild (Taste drücken oder 3 Sekunden warten)"
        try:
            started = time.perf_counter()
            while time.perf_counter() - started < 3.0:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(30) != -1:
                    break
        finally:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)

        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(config.detection.det_size, config.detection.det_size))
        faces = app.get(frame)
    finally:
        camera.release()

    available = ", ".join(map(str, available_indices)) or "keine"
    status = f"verfügbare Indizes: {available}; Kamera liefert {width}x{height}"
    # Kein hartes Scheitern: Beim Preflight kann der Bildausschnitt kurz leer sein;
    # das sichtbare Rohbild ist die verlässliche Entscheidungshilfe vor Ort.
    if not faces:
        return f"{status}; WARNUNG: Kein Gesicht im Rohbild erkannt"
    return f"{status}; {len(faces)} Gesicht(er) im Rohbild erkannt"


def _embeddings_check() -> str:
    import numpy as np

    try:
        with np.load(EMBEDDINGS_PATH, allow_pickle=False) as embeddings:
            ids = embeddings["ids"]
            names = embeddings["names"]
            vecs = embeddings["vecs"]
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError("embeddings.npz fehlt oder ist nicht lesbar.") from error

    if ids.ndim != 1 or names.ndim != 1 or vecs.ndim != 2 or vecs.shape[1:] != (512,):
        raise RuntimeError("embeddings.npz hat ein ungültiges Format.")
    if len(ids) != len(names) or len(ids) != len(vecs):
        raise RuntimeError("embeddings.npz enthält unterschiedlich viele Einträge.")
    return f"embeddings.npz ladbar ({len(ids)} Personen)"


def _benchmark_check(config: Config) -> str:
    """Führt 60 lokale Detection-/Recognition-Durchläufe aus, ohne einen Download anzustoßen."""

    _model_pack_check()
    import cv2
    from insightface.app import FaceAnalysis

    camera = _open_camera(config)
    try:
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(config.detection.det_size, config.detection.det_size))

        face_widths: list[float] = []
        processed = 0
        started = time.perf_counter()
        while processed < 60:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Kamera lieferte während des Benchmarks kein Frame.")
            if frame.shape[1] > config.detection.input_width:
                scale = config.detection.input_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            faces = app.get(frame)
            face_widths.extend(float(face.bbox[2] - face.bbox[0]) for face in faces)
            processed += 1
        elapsed = time.perf_counter() - started
    finally:
        camera.release()

    if elapsed <= 0:
        raise RuntimeError("Benchmark-Zeit konnte nicht bestimmt werden.")
    mean_width = sum(face_widths) / len(face_widths) if face_widths else 0.0
    return f"Benchmark: {processed / elapsed:.1f} fps, mittlere Gesichtsbreite {mean_width:.1f} px"


def _run_check(label: str, check: Callable[[], str]) -> bool:
    try:
        print(f"✓ {label}: {check()}")
        return True
    except Exception as error:  # noqa: BLE001 - Die Checkliste soll vollständig durchlaufen.
        print(f"✗ {label}: {error}")
        return False


def main(argv: Sequence[str] | None = None) -> int:
    """Gibt für alle fünf M0-Prüfpunkte genau eine Statuszeile aus."""

    del argv  # Die M0-CLI benötigt noch keine Argumente.
    try:
        config = load_config()
    except Exception as error:  # noqa: BLE001 - Konfigurationsfehler sollen als Checkfehler erscheinen.
        config_error = str(error)
        config = None
    else:
        config_error = ""

    def configuration_failure() -> str:
        raise RuntimeError(config_error)

    checks: list[tuple[str, Callable[[], str]]] = [
        ("1. Modellpack", _model_pack_check),
        ("2. CoreML", _coreml_check),
        (
            "3. Kamera",
            (lambda: _camera_check(config)) if config is not None else configuration_failure,
        ),
        ("4. Embeddings", _embeddings_check),
        (
            "5. Benchmark",
            (lambda: _benchmark_check(config)) if config is not None else configuration_failure,
        ),
    ]
    results = [_run_check(label, check) for label, check in checks]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
