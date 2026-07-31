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

## Ablauf am Vortragstag

```bash
python -m src.preflight              # 1. alles grün?
python -m src.live --source camera   # 2. Demo starten
```

Fallback ohne Kamera: `python -m src.live --source data/probe.mp4`

## Datenschutz

Gesichtserkennung erzeugt biometrische Daten nach Art. 9 DSGVO. Vor Nutzung
ist die ausdrückliche Einwilligung aller gefilmten Personen erforderlich.
Die Anwendung speichert keinen Anwesenheitszustand auf Platte.
Enrollment-Daten nach dem Vortrag löschen.

Details siehe `docs/mvp-plan.md`.
