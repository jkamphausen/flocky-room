"""Embedding-Galerie laden und Cosine-Matching.

Siehe docs/mvp-plan.md, M2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

EMBEDDINGS_PATH = Path("data/embeddings.npz")
_EMBEDDING_DIMENSIONS = 512


@dataclass(frozen=True)
class Gallery:
    """Lokale Referenz-Embeddings und ihre Anzeigenamen."""

    ids: Any
    names: Any
    vecs: Any

    @classmethod
    def load(cls, path: Path = EMBEDDINGS_PATH) -> Gallery:
        """Lädt und validiert die von :mod:`src.enroll` erzeugte Galerie."""

        import numpy as np

        try:
            with np.load(path, allow_pickle=False) as embeddings:
                ids = embeddings["ids"]
                names = embeddings["names"]
                vecs = embeddings["vecs"]
        except FileNotFoundError as error:
            raise ValueError(
                "embeddings.npz fehlt. Bitte zuerst `python -m src.enroll` ausführen."
            ) from error
        except (OSError, ValueError, KeyError) as error:
            raise ValueError("embeddings.npz ist nicht lesbar oder hat ein ungültiges Format.") from error

        if ids.ndim != 1 or names.ndim != 1 or vecs.ndim != 2 or vecs.shape[1:] != (_EMBEDDING_DIMENSIONS,):
            raise ValueError("embeddings.npz hat nicht das erwartete Format.")
        if len(ids) == 0 or len(ids) != len(names) or len(ids) != len(vecs):
            raise ValueError("embeddings.npz enthält keine konsistente Personen-Galerie.")
        if vecs.dtype != np.float32 or not np.isfinite(vecs).all():
            raise ValueError("embeddings.npz enthält ungültige Referenz-Embeddings.")

        return cls(ids=ids, names=names, vecs=vecs)

    def match(self, embedding: Any) -> tuple[str, float]:
        """Gibt die ähnlichste ``person_id`` und ihre Cosine-Similarity zurück."""

        import numpy as np

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (_EMBEDDING_DIMENSIONS,):
            raise ValueError("Embedding muss genau 512 Dimensionen haben.")
        if not np.isfinite(vector).all():
            raise ValueError("Embedding enthält ungültige Werte.")
        similarities = self.vecs @ vector
        index = int(np.argmax(similarities))
        return str(self.ids[index]), float(similarities[index])

    def name_for(self, person_id: str) -> str:
        """Gibt den Anzeigenamen zu einer bekannten stabilen ID zurück."""

        for index, known_id in enumerate(self.ids):
            if str(known_id) == person_id:
                return str(self.names[index])
        raise KeyError(f"Unbekannte person_id in Galerie: {person_id}")
