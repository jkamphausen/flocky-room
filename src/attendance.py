"""Anwesenheitsregistry, ausschliesslich im RAM."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttendanceRecord:
    """Der erste und zuletzt bestätigte Zeitpunkt einer Person."""

    first_seen: float
    last_seen: float


class AttendanceRegistry:
    """Hält Anwesenheit für die Laufzeit der Anwendung fest."""

    def __init__(self, present_timeout_s: float) -> None:
        if present_timeout_s < 0:
            raise ValueError("present_timeout_s darf nicht negativ sein.")
        self.present_timeout_s = present_timeout_s
        self._records: dict[str, AttendanceRecord] = {}

    @property
    def records(self) -> dict[str, AttendanceRecord]:
        """Eine Kopie der Registry für die Anzeige."""

        return self._records.copy()

    def update(self, person_id: str, now: float) -> None:
        """Registriert eine bestätigte Sichtung zum übergebenen Zeitpunkt."""

        record = self._records.get(person_id)
        first_seen = now if record is None else record.first_seen
        self._records[person_id] = AttendanceRecord(first_seen=first_seen, last_seen=now)

    def is_present(self, person_id: str, now: float) -> bool:
        """Gibt zurück, ob die Person innerhalb des Timeouts gesehen wurde."""

        record = self._records.get(person_id)
        return record is not None and now - record.last_seen < self.present_timeout_s

    def present_person_ids(self, now: float) -> set[str]:
        """Gibt alle aktuell anwesenden IDs zurück."""

        return {person_id for person_id in self._records if self.is_present(person_id, now)}

    def clear(self) -> None:
        """Leert die flüchtige Registry."""

        self._records.clear()
