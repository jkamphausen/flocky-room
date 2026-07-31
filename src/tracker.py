"""IoU-Tracker mit Label-Voting und Eindeutigkeitsregel.

Das Modul ist bewusst modellfrei: Der Aufrufer übergibt Bounding-Boxes und
liefert Recognition-Ergebnisse später über :meth:`Tracker.add_recognition`.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

BBox: TypeAlias = tuple[float, float, float, float]
Recognition: TypeAlias = tuple[str, float]
UNKNOWN_LABEL = "UNBEKANNT"


@dataclass
class Track:
    """Ein über mehrere Frames verfolgtes Gesicht."""

    track_id: int
    bbox: BBox
    age: int = 1
    frames_since_seen: int = 0
    votes: deque[Recognition] = field(default_factory=deque)
    last_embedding_frame: int | None = None
    label: str = UNKNOWN_LABEL
    label_similarity: float | None = None


class Tracker:
    """Verfolgt Detections und stabilisiert deren Personenlabels.

    Nach ``update(detections)`` sind die IDs der sichtbaren Tracks mit
    ``tracks_needing_recognition()`` verfügbar. Der Aufrufer berechnet für
    genau diese Track-Boxen bei Bedarf ein Embedding und ruft danach
    ``add_recognition(track_id, person_id, similarity)`` auf.
    """

    def __init__(
        self,
        *,
        iou_threshold: float,
        max_age_frames: int,
        vote_window: int,
        recognition_threshold: float,
        reid_interval: int,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold muss zwischen 0 und 1 liegen.")
        if max_age_frames < 0 or vote_window < 1 or reid_interval < 1:
            raise ValueError("max_age_frames >= 0, vote_window und reid_interval >= 1 erwartet.")
        self.iou_threshold = iou_threshold
        self.max_age_frames = max_age_frames
        self.vote_window = vote_window
        self.recognition_threshold = recognition_threshold
        self.reid_interval = reid_interval
        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1
        self._frame_number = 0

    @property
    def tracks(self) -> tuple[Track, ...]:
        """Alle noch lebenden Tracks, sortiert nach stabiler ID."""

        return tuple(self._tracks[track_id] for track_id in sorted(self._tracks))

    def update(
        self,
        detections: Iterable[BBox],
        recognitions: Mapping[int, Recognition] | None = None,
    ) -> tuple[Track, ...]:
        """Verarbeitet die Boxen eines Frames und optionale frische Matches.

        Die Rückgabe enthält alle sichtbaren Tracks dieses Frames. Das optionale
        ``recognitions`` ist praktisch, wenn der Aufrufer die Track-IDs bereits
        aus dem vorherigen Frame kennt; für neue Tracks folgt die Recognition
        üblicherweise per ``add_recognition`` nach diesem Aufruf.
        """

        self._frame_number += 1
        boxes = list(detections)
        unmatched_track_ids = set(self._tracks)
        unmatched_detection_indices = set(range(len(boxes)))

        candidates = sorted(
            (
                (self._iou(track.bbox, box), track_id, detection_index)
                for track_id, track in self._tracks.items()
                for detection_index, box in enumerate(boxes)
            ),
            reverse=True,
        )
        for score, track_id, detection_index in candidates:
            if score < self.iou_threshold:
                break
            if track_id not in unmatched_track_ids or detection_index not in unmatched_detection_indices:
                continue
            track = self._tracks[track_id]
            track.bbox = boxes[detection_index]
            track.age += 1
            track.frames_since_seen = 0
            unmatched_track_ids.remove(track_id)
            unmatched_detection_indices.remove(detection_index)

        for track_id in unmatched_track_ids:
            track = self._tracks[track_id]
            track.age += 1
            track.frames_since_seen += 1

        for detection_index in sorted(unmatched_detection_indices):
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = Track(
                track_id=track_id,
                bbox=boxes[detection_index],
                votes=deque(maxlen=self.vote_window),
            )

        expired_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.frames_since_seen > self.max_age_frames
        ]
        for track_id in expired_ids:
            del self._tracks[track_id]

        if recognitions:
            for track_id, (person_id, similarity) in recognitions.items():
                self.add_recognition(track_id, person_id, similarity)
        self._refresh_labels()
        return tuple(track for track in self.tracks if track.frames_since_seen == 0)

    def tracks_needing_recognition(self) -> tuple[Track, ...]:
        """Gibt sichtbare Tracks zurück, für die jetzt ein Embedding fällig ist."""

        return tuple(
            track
            for track in self.tracks
            if track.frames_since_seen == 0
            and (
                track.last_embedding_frame is None
                or self._frame_number - track.last_embedding_frame >= self.reid_interval
            )
        )

    def add_recognition(self, track_id: int, person_id: str, similarity: float) -> None:
        """Fügt einem sichtbaren Track ein frisches Galerie-Match hinzu."""

        track = self._tracks.get(track_id)
        if track is None:
            raise KeyError(f"Unbekannte track_id: {track_id}")
        track.votes.append((person_id, similarity))
        track.last_embedding_frame = self._frame_number
        self._refresh_labels()

    @staticmethod
    def _iou(first: BBox, second: BBox) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def _ranked_candidates(self, track: Track) -> list[tuple[str, float]]:
        counts = Counter(person_id for person_id, _ in track.votes)
        similarities = {
            person_id: sum(score for voted_id, score in track.votes if voted_id == person_id) / count
            for person_id, count in counts.items()
        }
        ranked = [
            (person_id, similarities[person_id])
            for person_id in sorted(counts, key=lambda value: (-counts[value], -similarities[value], value))
        ]
        if not ranked or ranked[0][1] < self.recognition_threshold:
            return []
        return [candidate for candidate in ranked if candidate[1] >= self.recognition_threshold]

    def _refresh_labels(self) -> None:
        rankings = {track_id: self._ranked_candidates(track) for track_id, track in self._tracks.items()}
        positions = {track_id: 0 for track_id in self._tracks}

        while True:
            claims: dict[str, list[int]] = {}
            for track_id, candidates in rankings.items():
                position = positions[track_id]
                if position < len(candidates):
                    claims.setdefault(candidates[position][0], []).append(track_id)
            conflicts = [claimants for claimants in claims.values() if len(claimants) > 1]
            if not conflicts:
                break
            for claimants in conflicts:
                winner = max(
                    claimants,
                    key=lambda track_id: (rankings[track_id][positions[track_id]][1], -track_id),
                )
                for track_id in claimants:
                    if track_id != winner:
                        positions[track_id] += 1

        for track_id, track in self._tracks.items():
            position = positions[track_id]
            candidates = rankings[track_id]
            if position < len(candidates):
                track.label, track.label_similarity = candidates[position]
            else:
                track.label = UNKNOWN_LABEL
                track.label_similarity = None
