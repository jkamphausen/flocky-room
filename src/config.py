"""Lädt und validiert die Konfiguration der Demo-Anwendung."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraConfig:
    index: int
    width: int
    height: int
    fourcc: str


@dataclass(frozen=True)
class DetectionConfig:
    input_width: int
    det_size: int
    min_face_px: int


@dataclass(frozen=True)
class RecognitionConfig:
    threshold: float
    vote_window: int
    reid_interval: int


@dataclass(frozen=True)
class TrackingConfig:
    iou_threshold: float
    max_age_frames: int


@dataclass(frozen=True)
class AttendanceConfig:
    present_timeout_s: float


@dataclass(frozen=True)
class DisplayConfig:
    fullscreen: bool
    output_width: int
    show_similarity: bool


@dataclass(frozen=True)
class Config:
    camera: CameraConfig
    detection: DetectionConfig
    recognition: RecognitionConfig
    tracking: TrackingConfig
    attendance: AttendanceConfig
    display: DisplayConfig


_SECTIONS = frozenset(
    {"camera", "detection", "recognition", "tracking", "attendance", "display"}
)


def _fail(message: str) -> None:
    raise ValueError(f"Ungültige Konfiguration: {message}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"'{name}' muss ein Abschnitt sein.")
    return value


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"'{name}' muss eine ganze Zahl >= {minimum} sein.")
    return value


def _number(value: Any, name: str, *, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"'{name}' muss eine Zahl sein.")
    result = float(value)
    if not isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        range_description = f"zwischen {minimum} und {maximum}" if maximum is not None else f">= {minimum}"
        _fail(f"'{name}' muss {range_description} sein.")
    return result


def _required(section: dict[str, Any], key: str, section_name: str) -> Any:
    try:
        return section[key]
    except KeyError:
        _fail(f"'{section_name}.{key}' fehlt.")


def load_config(path: str | Path = "config.yaml") -> Config:
    """Lädt *path* und gibt ausschließlich validierte, unveränderliche Werte zurück."""

    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file)
    except OSError as error:
        raise ValueError(f"Konfiguration kann nicht gelesen werden: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError("Konfiguration enthält ungültiges YAML.") from error

    root = _mapping(raw, "root")
    unexpected_sections = set(root) - _SECTIONS
    missing_sections = _SECTIONS - set(root)
    if missing_sections:
        _fail(f"Abschnitt(e) fehlen: {', '.join(sorted(missing_sections))}.")
    if unexpected_sections:
        _fail(f"Unbekannte(r) Abschnitt(e): {', '.join(sorted(unexpected_sections))}.")

    camera = _mapping(root["camera"], "camera")
    detection = _mapping(root["detection"], "detection")
    recognition = _mapping(root["recognition"], "recognition")
    tracking = _mapping(root["tracking"], "tracking")
    attendance = _mapping(root["attendance"], "attendance")
    display = _mapping(root["display"], "display")

    fourcc = _required(camera, "fourcc", "camera")
    if not isinstance(fourcc, str) or len(fourcc) != 4:
        _fail("'camera.fourcc' muss ein FourCC-String mit vier Zeichen sein.")

    fullscreen = _required(display, "fullscreen", "display")
    show_similarity = _required(display, "show_similarity", "display")
    if not isinstance(fullscreen, bool):
        _fail("'display.fullscreen' muss true oder false sein.")
    if not isinstance(show_similarity, bool):
        _fail("'display.show_similarity' muss true oder false sein.")

    return Config(
        camera=CameraConfig(
            index=_integer(_required(camera, "index", "camera"), "camera.index", minimum=0),
            width=_integer(_required(camera, "width", "camera"), "camera.width"),
            height=_integer(_required(camera, "height", "camera"), "camera.height"),
            fourcc=fourcc,
        ),
        detection=DetectionConfig(
            input_width=_integer(
                _required(detection, "input_width", "detection"), "detection.input_width"
            ),
            det_size=_integer(_required(detection, "det_size", "detection"), "detection.det_size"),
            min_face_px=_integer(
                _required(detection, "min_face_px", "detection"), "detection.min_face_px"
            ),
        ),
        recognition=RecognitionConfig(
            threshold=_number(
                _required(recognition, "threshold", "recognition"),
                "recognition.threshold",
                minimum=0.0,
                maximum=1.0,
            ),
            vote_window=_integer(
                _required(recognition, "vote_window", "recognition"), "recognition.vote_window"
            ),
            reid_interval=_integer(
                _required(recognition, "reid_interval", "recognition"), "recognition.reid_interval"
            ),
        ),
        tracking=TrackingConfig(
            iou_threshold=_number(
                _required(tracking, "iou_threshold", "tracking"),
                "tracking.iou_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
            max_age_frames=_integer(
                _required(tracking, "max_age_frames", "tracking"), "tracking.max_age_frames"
            ),
        ),
        attendance=AttendanceConfig(
            present_timeout_s=_number(
                _required(attendance, "present_timeout_s", "attendance"),
                "attendance.present_timeout_s",
                minimum=0.0,
            )
        ),
        display=DisplayConfig(
            fullscreen=fullscreen,
            output_width=_integer(
                _required(display, "output_width", "display"), "display.output_width"
            ),
            show_similarity=show_similarity,
        ),
    )
