# flocky-room

Live-Anwesenheitserkennung als Einstiegsdemo für einen Seminarvortrag über
Techoligarchie und Palantir. Eine Kamera filmt den Raum, erkannte Teilnehmende
bekommen ihren Namen ins Bild, eine Seitenleiste zeigt "anwesend / fehlt".

Vollständig lokal: keine Cloud-API, kein Netzwerkzugriff zur Laufzeit.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m src.preflight      # lädt beim ersten Lauf ~300 MB Modelldaten
```

## Enrollment (einmalig, vor dem Vortrag)

```bash
# data/people.yaml: person_id -> Anzeigename
# data/photos/<person_id>/*.jpg ablegen (1+ Fotos pro Person)
python -m src.enroll                 # erzeugt data/embeddings.npz
```

Prüft dabei automatisch, ob sich Personen anhand ihrer Fotos zu ähnlich
sind (Similarity-Warnung im Report) — bei einer Warnung Fotos
austauschen und erneut laufen lassen.

## Ablauf am Vortragstag

```bash
python -m src.preflight              # 1. alles grün?
python -m src.live                   # 2. Demo starten (--source camera ist der Standard)
```

Fallback ohne Kamera: `python -m src.live --source data/probe.mp4`
(Video-Datei loopt automatisch für die Dauer des Vortrags, gebremst auf
die Framerate der Datei).

**Tastenbelegung während `live.py` läuft:**

| Taste | Funktion |
|---|---|
| `q` | Beenden |
| `b` | Blackout — Bild sofort schwarz (Panik-Taste, nochmal drücken schaltet zurück) |
| `SPACE` | Standbild einfrieren/fortsetzen |
| `d` | Debug-Overlay (fps, Similarity-Werte, Track-IDs, aktiver Kamera-Index) |
| `r` | Anwesenheitsliste zurücksetzen |
| `s` | Screenshot nach `./shots/` |
| `c` | Nächste verfügbare Kamera durchschalten (nur bei `--source camera`; auf dieser Art Maschine ist der Kamera-Index nicht sitzungsstabil, falls das falsche Bild erscheint hiermit weiterschalten) |

## Datenschutz

Gesichtserkennung erzeugt biometrische Daten nach Art. 9 DSGVO. Vor Nutzung
ist die ausdrückliche Einwilligung aller gefilmten Personen erforderlich.
Die Anwendung speichert keinen Anwesenheitszustand auf Platte.
Enrollment-Daten nach dem Vortrag löschen.

Details siehe `docs/mvp-plan.md`.
