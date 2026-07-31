# MVP-Umsetzungsplan: Live-Gesichtserkennung als Vortragseinstieg

**Kontext:** Einstiegs-Demo für einen Seminarvortrag über Techoligarchie/Palantir. Eine Kamera filmt den Seminarraum, das Bild läuft live auf den Beamer, erkannte Teilnehmende bekommen ihren Namen ins Bild geschrieben, eine Seitenleiste zeigt "anwesend / fehlt noch". Rhetorischer Zweck: das Publikum erlebt Überwachungstechnik am eigenen Leib, bevor darüber geredet wird.

**Rahmenbedingungen:**
- Max. 24 Personen, Referenzfotos liegen vor
- Alles lokal, keine Cloud-API, kein Netzwerkzugriff zur Laufzeit
- Zielhardware: Apple Silicon Mac + USB-4K-Webcam (Obsbot Meet 2 o.ä.)
- Einmalige Nutzung, ca. 5 Minuten Laufzeit — Robustheit schlägt Featureumfang

---

## 1. Scope

### In Scope (MVP)
1. Enrollment-Skript: Referenzfotos → Embedding-Datenbank
2. Live-Anwendung: Kamerabild + Bounding Boxes + Namen + Anwesenheitsliste
3. Preflight-Check: verifiziert vor dem Vortrag, dass alles läuft
4. Video-Datei als alternative Quelle (Proben + Notfall-Fallback)

### Nicht in Scope
- Persistenz der Anwesenheitsdaten über den Programmlauf hinaus (bewusst: keine Datenspeicherung)
- Web-UI, Netzwerk-Streaming, Multi-Kamera
- Re-Identification über Kleidung/Körper, Alters-/Geschlechtsschätzung
- Erkennung von Personen ohne Referenzfoto (die bekommen `UNBEKANNT`)

---

## 2. Tech-Stack

| Komponente | Wahl | Begründung |
|---|---|---|
| Sprache | Python 3.11 | onnxruntime-Kompatibilität |
| Face Detection + Embedding | `insightface` (Modellpack `buffalo_l`) | SCRFD-Detektor + ArcFace-R50, deutlich robuster als dlib |
| Inference | `onnxruntime-silicon` bzw. `onnxruntime` mit CoreML-Provider | GPU/ANE-Beschleunigung auf Apple Silicon |
| Kamera & Bildverarbeitung | `opencv-python` | UVC-Zugriff, Rendering |
| Textrendering | `Pillow` | OpenCVs `putText` kann keine Umlaute |
| Config | `PyYAML` | |
| Numerik | `numpy` | |

**Wichtig:** `buffalo_l` (~300 MB) wird beim ersten Lauf nach `~/.insightface/models/` heruntergeladen. Muss **vor** dem Vortrag einmal mit Netzwerkzugriff passieren. Der Preflight-Check verifiziert das.

---

## 3. Repo-Struktur

```
attendance-demo/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── config.yaml
├── src/
│   ├── config.py          # Config-Dataclass + Loader
│   ├── engine.py          # InsightFace-Wrapper (Detection + Embedding)
│   ├── enroll.py          # CLI: Fotos → embeddings.npz
│   ├── gallery.py         # Laden/Matchen der Embedding-Galerie
│   ├── tracker.py         # IoU-Tracker + Label-Voting
│   ├── attendance.py      # Anwesenheitszustand
│   ├── overlay.py         # Rendering (Boxen, Namen, Seitenleiste)
│   ├── capture.py         # Kamera-Thread (latest-frame-wins)
│   ├── preflight.py       # CLI: Systemcheck
│   └── live.py            # CLI: Hauptanwendung
├── data/
│   ├── people.yaml        # person_id → Anzeigename
│   ├── photos/<person_id>/*.jpg
│   └── embeddings.npz     # generiert, gitignored
└── tests/
```

`data/photos/` und `data/embeddings.npz` gehören in `.gitignore` — biometrische Daten haben in keinem Repo etwas zu suchen.

---

## 4. Konfiguration (`config.yaml`)

```yaml
camera:
  index: 0
  width: 3840
  height: 2160
  fourcc: MJPG          # zwingend, sonst 5 fps in 4K

detection:
  input_width: 2560     # Frame wird hierauf skaliert, bevor detektiert wird
  det_size: 1280        # SCRFD-Eingabegröße
  min_face_px: 40       # kleinere Boxen werden ignoriert

recognition:
  threshold: 0.45       # Cosine-Similarity ab der ein Match gilt
  vote_window: 5        # Anzahl Embeddings pro Track für Mehrheitsentscheid
  reid_interval: 15     # nur jedes N-te Frame pro Track neu embedden

tracking:
  iou_threshold: 0.3
  max_age_frames: 30    # Track verfällt nach so vielen Frames ohne Match

attendance:
  present_timeout_s: 20 # so lange gilt jemand nach letztem Sichten als anwesend

display:
  fullscreen: true
  output_width: 1920
  show_similarity: true
```

---

## 5. Milestones

### M0 — Setup & Preflight
**Tasks**
- `pyproject.toml`, Abhängigkeiten, Config-Loader mit Dataclasses
- `preflight.py` prüft der Reihe nach und gibt eine klare Checkliste aus:
  1. Modellpack vorhanden (lokal, kein Download nötig)
  2. `CoreMLExecutionProvider` in `onnxruntime.get_available_providers()`
  3. Kamera öffnet sich, liefert die konfigurierte Auflösung
  4. `embeddings.npz` vorhanden und ladbar, Anzahl Personen
  5. Benchmark: 60 Frames durch die volle Pipeline, Ausgabe fps + mittlere Gesichtsbreite in Pixeln

**Akzeptanzkriterium:** `python -m src.preflight` gibt eine Liste mit ✓/✗ pro Punkt aus und beendet sich mit Exit-Code ≠ 0, sobald einer fehlschlägt.

---

### M1 — Enrollment
**Tasks**
- `data/people.yaml` im Format `person_id: "Anzeigename"`
- `enroll.py` iteriert über `data/photos/<person_id>/*.{jpg,jpeg,png}`:
  - Pro Foto Detection; bei 0 Gesichtern warnen und überspringen, bei >1 das größte nehmen und warnen
  - `face.normed_embedding` verwenden (bereits L2-normalisiert, 512-dim)
  - Pro Person alle Embeddings mitteln, Ergebnis erneut L2-normalisieren
- Speichern als `embeddings.npz` mit `ids` (N,), `names` (N,), `vecs` (N,512) float32
- **QA-Ausgabe:** paarweise Similarity-Matrix zwischen allen Personen berechnen, das höchste Cross-Person-Paar ausgeben. Liegt es über `threshold - 0.05`, laut warnen — diese beiden werden live verwechselt.

**Akzeptanzkriterium:** Lauf über den Fotoordner erzeugt `embeddings.npz` mit einem Eintrag pro Person in `people.yaml`, plus einen Report, welche Fotos verworfen wurden.

---

### M2 — Detection- und Matching-Pipeline
**Tasks**
- `engine.py`: dünner Wrapper um `FaceAnalysis`, initialisiert mit
  `providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']`,
  `app.prepare(ctx_id=0, det_size=(cfg.det_size, cfg.det_size))`.
  Nur die Module `detection` und `recognition` laden (`allowed_modules`) — Landmarks, Alter und Geschlecht kosten Zeit und werden nicht gebraucht.
- `gallery.py`: `match(embedding) -> (person_id, similarity)`. Da alle Vektoren L2-normalisiert sind, ist Cosine-Similarity ein einziges Matrixprodukt `vecs @ emb` — argmax davon.
- Skalierungslogik: Frame auf `input_width` verkleinern, detektieren, Bounding-Box-Koordinaten für die Anzeige zurückrechnen.

**Akzeptanzkriterium:** Ein Standbild mit mehreren bekannten Gesichtern wird korrekt zugeordnet; `python -m src.engine --debug bild.jpg` gibt pro Gesicht Box, Match und Similarity aus.

---

### M3 — Tracking, Voting, Anwesenheit
Der Grund für diesen Schritt: Frame-für-Frame-Erkennung flackert sichtbar (Namen springen zwischen Personen), und ein Embedding pro Gesicht pro Frame ist unnötig teuer.

**Tasks**
- `tracker.py`: greedy IoU-Matching zwischen Detections des aktuellen Frames und bestehenden Tracks. Jeder Track hält `track_id`, `bbox`, `age`, `frames_since_seen` und ein `deque(maxlen=vote_window)` mit `(person_id, similarity)`.
- Embedding wird nur berechnet, wenn der Track neu ist oder `reid_interval` Frames seit dem letzten Embedding vergangen sind. Detection läuft weiter jedes Frame.
- Label des Tracks = häufigster `person_id` im Voting-Fenster, sofern dessen mittlere Similarity ≥ `threshold`, sonst `UNBEKANNT`.
- **Eindeutigkeitsregel:** Ist dieselbe `person_id` gleichzeitig zwei Tracks zugeordnet, behält der mit der höheren Similarity das Label, der andere fällt auf seinen zweitbesten Kandidaten oder `UNBEKANNT` zurück. Ohne das steht derselbe Name an zwei Gesichtern, was die Demo sofort unglaubwürdig macht.
- `attendance.py`: Registry `person_id -> (first_seen, last_seen)`. Anwesend, wenn `now - last_seen < present_timeout_s`. Alles nur im RAM, nichts auf Platte.

**Akzeptanzkriterium:** Bei einem Testvideo bleiben Namen über mindestens 5 Sekunden stabil am selben Gesicht, kein Name erscheint doppelt.

---

### M4 — Rendering & Bedienung
**Tasks**
- `capture.py`: Kamera in eigenem Thread, immer nur das jüngste Frame vorhalten (`latest-frame-wins`). Ohne das läuft der OpenCV-interne Puffer voll und das Bild hängt sekundenweise hinter der Realität — bei einer Live-Demo tödlich.
- `overlay.py`:
  - Bounding Box pro Track, Namensplakette darüber
  - Optional Similarity als Balken oder Prozentwert
  - Seitenleiste: zwei Spalten "ANWESEND" (grün) / "FEHLT" (grau), alphabetisch
  - Kopfzeile: `n/24 erfasst`, Uhrzeit, fps
  - **Text via Pillow**, nicht `cv2.putText` — sonst werden Umlaute zu Kästchen. Frame → `PIL.Image` → `ImageDraw.text` mit einer mitgelieferten TTF → zurück zu numpy. Font-Objekt einmalig cachen, nicht pro Frame laden.
- `live.py`: Hauptschleife, Fullscreen-Fenster, Anzeige auf `output_width` herunterskaliert (Detection läuft weiter auf der größeren Variante).
- Tastenbelegung:

| Taste | Funktion |
|---|---|
| `q` | Beenden |
| `b` | Blackout — Bild sofort schwarz (Panik-Taste) |
| `SPACE` | Standbild einfrieren/fortsetzen |
| `d` | Debug-Overlay (fps, Similarity-Werte, Track-IDs) |
| `r` | Anwesenheitsliste zurücksetzen |
| `s` | Screenshot nach `./shots/` |

**Akzeptanzkriterium:** Läuft mit ≥15 fps bei 4K-Eingang auf Apple Silicon, Namen erscheinen mit korrekten Umlauten, `b` schaltet innerhalb eines Frames auf Schwarz.

---

### M5 — Proben & Absicherung
**Tasks**
- `--source path/to/video.mp4` als Alternative zu `--source camera`
- Im echten Raum, unter echter Beleuchtung, mit ein paar Personen einen 60-Sekunden-Clip aufnehmen. Damit die Pipeline testen und `threshold` sowie `input_width` nachjustieren.
- Diesen Clip als Fallback behalten: falls die Kamera am Vortragstag streikt, läuft die Demo eben auf Konserve. Das merkt niemand.
- README mit der Ablaufreihenfolge am Vortragstag.

**Akzeptanzkriterium:** Die Demo lässt sich vollständig ohne angeschlossene Kamera vorführen.

---

## 6. Bekannte Stolpersteine

| Problem | Gegenmaßnahme |
|---|---|
| Auto-Framing der Kamera zoomt selbstständig herum | In der Hersteller-Software abschalten, Preflight zeigt das Rohbild zur Kontrolle |
| macOS fragt beim ersten Start nach Kamerazugriff für das Terminal | Vorher einmal erteilen, nicht vor Publikum |
| USB-Hub drosselt auf USB 2.0 → kein 4K30 | Kamera direkt an den Rechner |
| Gegenlicht von der Leinwand | Kamera so stellen, dass die Leinwand nicht im Bild ist |
| Modell-Download beim ersten Lauf | Preflight schlägt fehl, wenn Modell nicht lokal liegt |
| Zu kleine Gesichter in den hinteren Reihen | `input_width` hoch, Bildausschnitt auf die vorderen zwei Reihen begrenzen |

---

## 7. Rechtlicher Rahmen

Gesichtserkennung erzeugt biometrische Daten im Sinne von Art. 9 DSGVO. Vor der Demo brauchst du eine ausdrückliche, informierte und freiwillige Einwilligung jeder gefilmten Person — freiwillig heißt: wer nicht will, muss ohne Nachteil aussteigen können, und die Demo muss auch dann funktionieren.

Technisch abgesichert durch: keine Persistenz (Registry nur im RAM), kein Netzwerkzugriff zur Laufzeit, Fotos und `embeddings.npz` nicht im Repo, Löschung der Enrollment-Daten nach dem Vortrag.

Der Kontrast zwischen "wir mussten hier 24 Einwilligungen einsammeln" und dem, was im Vortrag über kommerzielle Datenerfassung folgt, ist didaktisch wahrscheinlich der stärkste Moment der ganzen Demo — die Frage "wer von euch hat den Zettel eigentlich gelesen?" trägt den Übergang.

---

## 8. `AGENTS.md` (Inhalt für das Repo-Root)

```markdown
# Projekt: Live-Anwesenheitserkennung (Vortragsdemo)

Einmalig genutzte Demo-Anwendung für einen Seminarvortrag. Priorität:
Zuverlässigkeit im Live-Betrieb vor Featureumfang und vor Eleganz.

## Grundregeln
- Python 3.11, Typannotationen durchgehend, `ruff` als Linter
- Keine Netzwerkaufrufe zur Laufzeit. Modelle liegen lokal.
- Keine Persistenz biometrischer Daten. Anwesenheitszustand nur im RAM.
- Keine neuen Abhängigkeiten ohne Notwendigkeit; der Stack ist in
  README.md abschließend festgelegt.
- Jeder Milestone hat ein Akzeptanzkriterium im Plan. Erst wenn das
  nachweislich erfüllt ist, geht es zum nächsten.

## Fehlerverhalten
Die Anwendung darf im Live-Betrieb unter keinen Umständen abstürzen.
Jede Exception in der Hauptschleife wird geloggt, das Frame wird
übersprungen, die Schleife läuft weiter.

## Tests
Reine Logik (Tracker, Gallery-Matching, Attendance) wird unit-getestet.
Kamera- und Rendering-Code nicht — der wird manuell gegen das Testvideo
verifiziert.
```
