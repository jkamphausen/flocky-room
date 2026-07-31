# Projekt: Live-Anwesenheitserkennung (Vortragsdemo)

Einmalig genutzte Demo-Anwendung für einen Seminarvortrag. Priorität:
Zuverlässigkeit im Live-Betrieb vor Featureumfang und vor Eleganz.

Der verbindliche Umsetzungsplan steht in `docs/mvp-plan.md`. Arbeite die
Milestones der Reihe nach ab und weise das jeweilige Akzeptanzkriterium
nach, bevor du zum nächsten übergehst.

## Grundregeln
- Python 3.11, Typannotationen durchgehend, `ruff` als Linter
- Keine Netzwerkaufrufe zur Laufzeit. Modelle liegen lokal.
- Keine Persistenz biometrischer Daten. Anwesenheitszustand nur im RAM.
- Keine neuen Abhängigkeiten ohne Notwendigkeit; der Stack ist in
  pyproject.toml abschließend festgelegt.
- Niemals Inhalte aus `data/photos/` oder `data/embeddings.npz` committen,
  loggen oder in Fehlermeldungen ausgeben.

## Fehlerverhalten
Die Anwendung darf im Live-Betrieb unter keinen Umständen abstürzen.
Jede Exception in der Hauptschleife wird geloggt, das Frame wird
übersprungen, die Schleife läuft weiter.

## Tests
Reine Logik (tracker, gallery, attendance) wird unit-getestet.
Kamera- und Rendering-Code nicht - der wird manuell gegen ein
aufgezeichnetes Testvideo verifiziert.
