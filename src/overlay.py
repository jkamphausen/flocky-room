"""Rendering von Tracks und Anwesenheit mit Pillow-Text."""

from __future__ import annotations

import time
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tracker import UNKNOWN_LABEL

if TYPE_CHECKING:
    from .attendance import AttendanceRegistry
    from .gallery import Gallery
    from .tracker import Track


FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial.ttf")


@cache
def _font(size: int) -> ImageFont.FreeTypeFont:
    """Lädt jeden benötigten macOS-Systemfont genau einmal."""

    if not FONT_PATH.is_file():
        raise RuntimeError(f"Erforderlicher Systemfont fehlt: {FONT_PATH}")
    return ImageFont.truetype(FONT_PATH, size=size)


def ensure_font_available() -> None:
    """Prüft den für dieses lokale macOS-Setup verbindlichen Font frühzeitig."""

    _font(18)


def render_frame(
    frame: Any,
    tracks: tuple[Track, ...],
    gallery: Gallery,
    attendance: AttendanceRegistry,
    *,
    now: float,
    fps: float,
    show_similarity: bool,
    debug: bool,
    source_size: tuple[int, int],
    camera_index: int | None = None,
) -> np.ndarray:
    """Zeichnet ein neues BGR-Ausgabeframe ohne anwendungsweiten Zustand.

    ``source_size`` sind Breite und Höhe des Detection-Frames, damit Track-Boxen
    beim vorherigen Herunterskalieren korrekt auf die Anzeige passen.
    """

    if frame is None or getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
        raise ValueError("Frame muss ein dreikanaliges BGR-Bild sein.")
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source_size muss positive Werte enthalten.")

    image = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    scale_x, scale_y = width / source_width, height / source_height
    header_height = 58
    sidebar_width = min(max(300, width // 4), max(300, width - 120))
    sidebar_left = width - sidebar_width

    draw.rectangle((0, 0, width, header_height), fill=(12, 18, 28))
    present_ids = attendance.present_person_ids(now)
    person_ids = [str(person_id) for person_id in gallery.ids]
    present = sorted(
        (gallery.name_for(person_id) for person_id in person_ids if person_id in present_ids),
        key=str.casefold,
    )
    missing = sorted(
        (gallery.name_for(person_id) for person_id in person_ids if person_id not in present_ids),
        key=str.casefold,
    )
    draw.text((18, 14), f"{len(present)}/{len(person_ids)} erfasst", font=_font(28), fill=(245, 245, 245))
    draw.text(
        (width - 415, 17),
        time.strftime("%H:%M:%S"),
        font=_font(22),
        fill=(190, 205, 220),
    )
    draw.text((width - 245, 17), f"{fps:4.1f} fps", font=_font(22), fill=(190, 205, 220))
    if debug and camera_index is not None:
        draw.text((18, 40), f"Kamera-Index {camera_index} (c zum Wechseln)", font=_font(16), fill=(150, 165, 180))

    draw.rectangle((sidebar_left, header_height, width, height), fill=(18, 24, 35))
    column_width = sidebar_width // 2
    draw.text((sidebar_left + 12, header_height + 12), "ANWESEND", font=_font(19), fill=(80, 215, 130))
    draw.text((sidebar_left + column_width + 8, header_height + 12), "FEHLT", font=_font(19), fill=(180, 185, 195))
    _draw_names(draw, present, sidebar_left + 12, header_height + 43, column_width - 18, (80, 215, 130), height)
    _draw_names(draw, missing, sidebar_left + column_width + 8, header_height + 43,
                column_width - 14, (190, 195, 205), height)

    for track in tracks:
        x1, y1, x2, y2 = (value * scale for value, scale in zip(track.bbox, (scale_x, scale_y, scale_x, scale_y)))
        color = (255, 185, 65) if track.label == UNKNOWN_LABEL else (70, 220, 120)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = UNKNOWN_LABEL if track.label == UNKNOWN_LABEL else gallery.name_for(track.label)
        if show_similarity and track.label_similarity is not None:
            label = f"{label} {track.label_similarity:.0%}"
        if debug:
            label = f"#{track.track_id} {label}"
        text_box = draw.textbbox((0, 0), label, font=_font(22))
        label_top = max(header_height, y1 - (text_box[3] - text_box[1]) - 10)
        draw.rectangle((x1, label_top, x1 + text_box[2] + 12, y1), fill=color)
        draw.text((x1 + 6, label_top + 3), label, font=_font(22), fill=(15, 20, 25))

    return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])


def _draw_names(
    draw: ImageDraw.ImageDraw,
    names: list[str],
    x: int,
    y: int,
    max_width: int,
    color: tuple[int, int, int],
    bottom: int,
) -> None:
    font = _font(18)
    for name in names:
        if y + 24 > bottom:
            return
        clipped = name
        while clipped and draw.textlength(clipped, font=font) > max_width:
            clipped = clipped[:-1]
        if clipped != name and len(clipped) >= 2:
            clipped = f"{clipped[:-1]}…"
        draw.text((x, y), clipped, font=font, fill=color)
        y += 25
