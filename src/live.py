"""CLI-Hauptschleife für die Live-Anwesenheitserkennung."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .attendance import AttendanceRegistry
from .capture import CameraCapture
from .config import load_config
from .engine import DetectedFace, FaceEngine
from .gallery import Gallery
from .overlay import ensure_font_available, render_frame
from .tracker import UNKNOWN_LABEL, Track, Tracker

if TYPE_CHECKING:
    from .config import Config


LOGGER = logging.getLogger(__name__)
WINDOW_NAME = "Live-Anwesenheit"
_CAMERA_INDEXES_TO_PROBE = range(6)


def _available_camera_indices() -> list[int]:
    """Ermittelt die auf dieser Maschine öffnenden Kamera-Indizes.

    Der cv2/AVFoundation-Index derselben physischen Kamera ist auf macOS nicht
    über Sitzungen hinweg stabil (OBS Virtual Camera und echte Webcams können
    die Indizes tauschen) - deshalb wird zur Laufzeit neu geprobt statt sich
    auf einen einmal ermittelten Index zu verlassen.
    """

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


def _filtered_faces(faces: list[DetectedFace], minimum: int) -> list[DetectedFace]:
    return [face for face in faces if min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]) >= minimum]


def _face_for_track(track: Track, faces: list[DetectedFace]) -> DetectedFace | None:
    """Findet die Detection desselben Frames für die aktuelle Track-Box."""

    for face in faces:
        if face.bbox == track.bbox:
            return face
    return None


def _make_tracker(config: Config) -> Tracker:
    return Tracker(
        iou_threshold=config.tracking.iou_threshold,
        max_age_frames=config.tracking.max_age_frames,
        vote_window=config.recognition.vote_window,
        recognition_threshold=config.recognition.threshold,
        reid_interval=config.recognition.reid_interval,
    )


def _display_size(frame: Any, output_width: int) -> tuple[int, int]:
    height, width = frame.shape[:2]
    if width <= output_width:
        return width, height
    return output_width, max(1, round(height * output_width / width))


def _save_screenshot(frame: Any) -> None:
    import cv2

    shots = Path("shots")
    shots.mkdir(exist_ok=True)
    path = shots / f"shot-{time.strftime('%Y%m%d-%H%M%S')}.png"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError("Screenshot konnte nicht geschrieben werden.")
    LOGGER.info("Screenshot gespeichert: %s", path)


def main(argv: Sequence[str] | None = None) -> int:
    """Startet die lokale Kameraanwendung; M4 benötigt keine CLI-Argumente."""

    del argv
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config()
        gallery = Gallery.load()
        ensure_font_available()
        engine = FaceEngine(config)
        tracker = _make_tracker(config)
        attendance = AttendanceRegistry(config.attendance.present_timeout_s)
        camera_indices = _available_camera_indices()
        camera_position = camera_indices.index(config.camera.index) if config.camera.index in camera_indices else 0
        if camera_indices:
            config = replace(config, camera=replace(config.camera, index=camera_indices[camera_position]))
        capture = CameraCapture(config)
        capture.start()
    except Exception as error:  # noqa: BLE001 - Klarer Startfehler statt Traceback im Vortrag.
        LOGGER.error("Initialisierung fehlgeschlagen: %s", error)
        return 1

    LOGGER.info("Aktive Kamera: Index %d (verfügbar: %s)", config.camera.index, camera_indices)

    import cv2

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if config.display.fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    blackout = False
    frozen = False
    debug = False
    displayed_frame: Any | None = None
    fps = 0.0
    try:
        while True:
            if not frozen:
                try:
                    frame = capture.get_frame(timeout_s=0.2)
                except Exception:
                    LOGGER.exception("Kamera-Frame konnte nicht abgerufen werden; Versuch wird wiederholt.")
                    frame = None
                if frame is not None:
                    try:
                        started = time.perf_counter()
                        faces = _filtered_faces(engine.detect(frame), config.detection.min_face_px)
                        visible_tracks = tracker.update([face.bbox for face in faces])
                        for track in tracker.tracks_needing_recognition():
                            face = _face_for_track(track, faces)
                            if face is not None:
                                person_id, similarity = gallery.match(face.embedding)
                                tracker.add_recognition(track.track_id, person_id, similarity)
                        now = time.monotonic()
                        for track in visible_tracks:
                            if track.label != UNKNOWN_LABEL and track.label_similarity is not None and track.label_similarity >= config.recognition.threshold:
                                attendance.update(track.label, now)
                        display_width, display_height = _display_size(frame, config.display.output_width)
                        display = cv2.resize(frame, (display_width, display_height))
                        elapsed = time.perf_counter() - started
                        fps = 1.0 / elapsed if elapsed > 0 else 0.0
                        displayed_frame = render_frame(
                            display,
                            visible_tracks,
                            gallery,
                            attendance,
                            now=now,
                            fps=fps,
                            show_similarity=config.display.show_similarity,
                            debug=debug,
                            source_size=(frame.shape[1], frame.shape[0]),
                            camera_index=config.camera.index,
                        )
                    except Exception:
                        LOGGER.exception("Fehler bei der Verarbeitung eines Frames; Frame wird übersprungen.")
            output = displayed_frame
            if output is not None:
                if blackout:
                    output = output * 0
                cv2.imshow(WINDOW_NAME, output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                blackout = not blackout
            elif key == ord(" "):
                frozen = not frozen
            elif key == ord("d"):
                debug = not debug
            elif key == ord("r"):
                attendance.clear()
            elif key == ord("s") and output is not None:
                try:
                    _save_screenshot(output)
                except Exception:
                    LOGGER.exception("Screenshot fehlgeschlagen.")
            elif key == ord("c") and camera_indices:
                camera_indices = _available_camera_indices() or camera_indices
                next_position = (camera_position + 1) % len(camera_indices)
                candidate_index = camera_indices[next_position]
                candidate_config = replace(config, camera=replace(config.camera, index=candidate_index))
                try:
                    candidate_capture = CameraCapture(candidate_config)
                    candidate_capture.start()
                except Exception:
                    LOGGER.exception("Kamera-Wechsel zu Index %d fehlgeschlagen.", candidate_index)
                else:
                    capture.stop()
                    capture = candidate_capture
                    config = candidate_config
                    camera_position = next_position
                    LOGGER.info("Kamera gewechselt zu Index %d.", candidate_index)
    finally:
        capture.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
