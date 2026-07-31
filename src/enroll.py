"""CLI zum Erzeugen der lokalen Referenz-Galerie aus Enrollment-Fotos."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from .config import load_config

PEOPLE_PATH = Path("data/people.yaml")
PHOTOS_DIR = Path("data/photos")
EMBEDDINGS_PATH = Path("data/embeddings.npz")
_PHOTO_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def _load_people(path: Path = PEOPLE_PATH) -> dict[str, str]:
    """Lädt die Zuordnung von stabilen IDs zu Anzeigenamen."""

    try:
        with path.open(encoding="utf-8") as people_file:
            raw = yaml.safe_load(people_file)
    except OSError as error:
        raise ValueError("people.yaml kann nicht gelesen werden.") from error
    except yaml.YAMLError as error:
        raise ValueError("people.yaml enthält ungültiges YAML.") from error

    if not isinstance(raw, dict) or not raw:
        raise ValueError("people.yaml muss mindestens eine person_id enthalten.")
    if not all(isinstance(person_id, str) and isinstance(name, str) for person_id, name in raw.items()):
        raise ValueError("people.yaml muss person_id: Anzeigename enthalten.")
    return raw


def _photo_files(person_id: str) -> list[Path]:
    """Gibt die unterstützten Referenzfotos einer Person in fester Reihenfolge zurück."""

    person_dir = PHOTOS_DIR / person_id
    if not person_dir.is_dir():
        return []
    return sorted(path for path in person_dir.iterdir() if path.suffix.lower() in _PHOTO_SUFFIXES)


def _face_area(face: Any) -> float:
    bbox = face.bbox
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _normalise(vector: Any) -> Any:
    import numpy as np

    result = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(result))
    if norm == 0.0:
        raise ValueError("Embedding mit Länge null erhalten.")
    return result / norm


def _build_face_analysis(det_size: int) -> Any:
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def _enroll_person(person_id: str, app: Any) -> tuple[Any, int, list[tuple[str, str]]]:
    import cv2
    import numpy as np

    embeddings: list[Any] = []
    discarded: list[tuple[str, str]] = []
    for photo_path in _photo_files(person_id):
        frame = cv2.imread(str(photo_path))
        if frame is None:
            discarded.append((photo_path.name, "Bild konnte nicht gelesen werden"))
            continue
        try:
            faces = app.get(frame)
        except Exception:  # noqa: BLE001 - Die restlichen Referenzfotos bleiben verwertbar.
            discarded.append((photo_path.name, "Gesichtserkennung fehlgeschlagen"))
            continue
        if not faces:
            discarded.append((photo_path.name, "0 Gesichter erkannt"))
            continue
        if len(faces) > 1:
            discarded.append((photo_path.name, ">1 Gesichter erkannt; größtes verwendet"))
        face = max(faces, key=_face_area)
        embedding = _normalise(face.normed_embedding)
        if embedding.shape != (512,):
            discarded.append((photo_path.name, "Embedding hat nicht 512 Dimensionen"))
            continue
        embeddings.append(embedding)

    if not embeddings:
        raise RuntimeError(f"Keine verwertbaren Referenzfotos für {person_id}.")
    mean_embedding = _normalise(np.mean(embeddings, axis=0)).astype(np.float32)
    return mean_embedding, len(embeddings), discarded


def _cross_check(ids: list[str], vecs: Any, threshold: float) -> None:
    import numpy as np

    if len(ids) < 2:
        print("QA: Nicht genug Personen für Cross-Check.")
        return
    similarities = vecs @ vecs.T
    upper_indices = np.triu_indices(len(ids), k=1)
    best_index = int(np.argmax(similarities[upper_indices]))
    left = int(upper_indices[0][best_index])
    right = int(upper_indices[1][best_index])
    similarity = float(similarities[left, right])
    print(f"QA: Höchste Cross-Person-Similarity: {ids[left]} / {ids[right]} = {similarity:.3f}")
    if similarity > threshold - 0.05:
        print("QA-WARNUNG: Dieses Personenpaar wird live möglicherweise verwechselt.")


def main(argv: Sequence[str] | None = None) -> int:
    """Erstellt ``data/embeddings.npz`` und gibt den Enrollment-Bericht aus."""

    del argv
    import numpy as np

    try:
        config = load_config()
    except ValueError as error:
        print(f"Konfigurationsfehler: {error}")
        return 1

    try:
        people = _load_people()
    except ValueError as error:
        print(f"people.yaml-Fehler: {error}")
        return 1

    try:
        app = _build_face_analysis(config.detection.det_size)
    except Exception as error:  # noqa: BLE001 - Engine-Init darf nicht crashen, sondern klar melden.
        print(f"Gesichtserkennungs-Engine konnte nicht initialisiert werden: {error}")
        return 1

    ids: list[str] = []
    names: list[str] = []
    vecs: list[Any] = []
    reports: list[tuple[str, int, list[tuple[str, str]]]] = []
    skipped: list[str] = []
    for person_id, name in people.items():
        try:
            vector, used_count, discarded = _enroll_person(person_id, app)
        except RuntimeError as error:
            print(f"WARNUNG: {person_id} übersprungen: {error}")
            skipped.append(person_id)
            continue
        ids.append(person_id)
        names.append(name)
        vecs.append(vector)
        reports.append((person_id, used_count, discarded))

    if not ids:
        print("Enrollment fehlgeschlagen: keine Person hat brauchbare Fotos geliefert.")
        return 1

    vectors = np.asarray(vecs, dtype=np.float32)
    np.savez(EMBEDDINGS_PATH, ids=np.asarray(ids), names=np.asarray(names), vecs=vectors)

    for person_id, used_count, discarded in reports:
        print(f"{person_id}: {used_count} Foto(s) verwendet")
        for filename, reason in discarded:
            print(f"  WARNUNG: {filename}: {reason}")
    if skipped:
        print(f"Übersprungen (keine brauchbaren Fotos): {', '.join(skipped)}")
    _cross_check(ids, vectors, config.recognition.threshold)
    print(f"Enrollment abgeschlossen: {len(ids)} Personen gespeichert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
