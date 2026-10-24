#  Forward-Once Kennzeichenerkennung – Anleitung

---

## Projektstruktur

```
foreward-once/
├── fo_kennzeichen.py         → Hauptprogramm
├── roi_kalibrierung.py       → Kamerabereich einmalig festlegen
├── forward_once_xor_...py    → Prof-Referenz (nicht anfassen)
├── schranke.ino              → Arduino (nicht anfassen)
├── requirements.txt          → Pakete
├── start.sh                  → Starter
├── fo_model.pkl              → Trainiertes Modell (wird automatisch erstellt)
└── echte_bilder/             → Deine Kamerabilder (wird automatisch erstellt)
    ├── unbekannt/            → Gesicht, Hintergrund, alles ohne Kennzeichen
    ├── MA_AB1234/            → Bilder von Kennzeichen 1
    ├── B_XY9999/             → Bilder von Kennzeichen 2
    ├── HD_ZZ5678/            → Bilder von Kennzeichen 3
    └── KA_TT0001/            → Bilder von Kennzeichen 4
```

---

## Schritt 1 – Einmalige Einrichtung

**Nur einmal beim allerersten Start nötig.**

```bash
# Terminal in VS Code öffnen: Strg + `

# In den Projektordner navigieren
cd /pfad/zu/foreward-once

# Virtuelle Umgebung erstellen
python3 -m venv .venv

# Virtuelle Umgebung aktivieren  ← das musst du JEDES MAL machen wenn du das Terminal neu öffnest
source .venv/bin/activate

# Pakete installieren
pip install -r requirements.txt
```

> ⚠️ **Wichtig:** Jedes Mal wenn du ein neues Terminal öffnest, musst du `source .venv/bin/activate` ausführen bevor du irgendetwas startest.

---

## Schritt 2 – ROI kalibrieren (einmalig)

Die ROI ist der Bereich im Kamerabild wo das Kennzeichen erscheint. Du markierst ihn einmal mit der Maus.

```bash
python3 roi_kalibrierung.py
```

1. Ein Kamerafenster öffnet sich
2. **Mit der Maus** den Bereich ziehen wo das Kennzeichen vor die Kamera gehalten wird
3. **ENTER** drücken → gespeichert als `roi.json`

> Neu kalibrieren: einfach nochmal `python3 roi_kalibrierung.py` ausführen

---

## Schritt 3 – Unbekannt-Bilder sammeln (wichtig!)

Das Netz muss lernen was **kein Kennzeichen** ist — also dein Gesicht, Hände, Hintergrund, leere Szenen.

```bash
python3 fo_kennzeichen.py --auto --dauer 60
```

- Die Kamera läuft **60 Sekunden** und speichert automatisch alle 0.4 Sekunden ein Bild
- Bewege dich vor der Kamera, halte deine Hände rein, zeige verschiedene Objekte
- Du siehst unten einen Fortschrittsbalken und wie viele Bilder gespeichert wurden
- `q` zum vorzeitig beenden

> Tipp: Je mehr verschiedene Situationen du zeigst, desto besser lernt das Netz zu unterscheiden.

---

## Schritt 4 – Echte Kennzeichen-Bilder sammeln (wichtig!)

Das Netz muss lernen wie deine **echten Kennzeichen** aussehen.

```bash
python3 fo_kennzeichen.py --sammeln
```

Im Kamerafenster siehst du unten welche Taste für welches Kennzeichen ist:

| Taste | Kennzeichen |
|-------|-------------|
| `1`   | MA AB1234   |
| `2`   | B XY9999    |
| `3`   | HD ZZ5678   |
| `4`   | KA TT0001   |
| `u`   | Unbekannt   |
| `q`   | Beenden     |

**Vorgehen:**
1. Kennzeichen 1 vor die Kamera halten → **Taste `1` ca. 30-50 mal drücken** (verschiedene Winkel und Abstände)
2. Kennzeichen 2 vor die Kamera halten → **Taste `2` ca. 30-50 mal drücken**
3. usw. für alle Kennzeichen
4. `q` zum Beenden

>  **Je mehr echte Bilder pro Kennzeichen, desto besser die Erkennung.** Mindestens 30 pro Kennzeichen.

---

## Schritt 5 – Modell trainieren

```bash
python3 fo_kennzeichen.py --neu
```

- `--neu` startet von Null (empfohlen nach dem ersten Sammeln)
- Das Training dauert ca. **2-3 Minuten**
- Du siehst die Genauigkeit pro Epoche — sollte gegen Ende über 85% sein

```
[13:42:01]   Epoche   5/80 │ Genauigkeit: 71.3%
[13:42:15]   Epoche  10/80 │ Genauigkeit: 83.6%
[13:42:30]   Epoche  15/80 │ Genauigkeit: 89.1%
...
```

---

## Schritt 6 – Live-Erkennung starten

```bash
python3 fo_kennzeichen.py
```

- Kamerafenster öffnet sich
- **Gelber Rahmen** = deine ROI-Zone → dort das Kennzeichen hinhalten
- **Grüner Rahmen + Text** = Kennzeichen erkannt, Schranke öffnet
- **Oranger Rahmen** = erkannt aber zu unsicher (unter 70% Konfidenz)
- Statusleiste oben zeigt ob Schranke offen oder zu ist
- `q` zum Beenden

---

## Schritt 7 – Nachbessern wenn Erkennung nicht stimmt

**Wenn falsche Kennzeichen erkannt werden:**
```bash
# Mehr echte Bilder von jedem Kennzeichen sammeln
python3 fo_kennzeichen.py --sammeln

# Weitertrainieren (behält bisherigen Fortschritt)
python3 fo_kennzeichen.py --train
```

**Wenn Gesicht/Hintergrund als Kennzeichen erkannt wird:**
```bash
# Mehr Unbekannt-Bilder sammeln
python3 fo_kennzeichen.py --auto --dauer 60

# Weitertrainieren
python3 fo_kennzeichen.py --train
```

**Wenn gar nichts funktioniert – komplett neu starten:**
```bash
# Alle gesammelten Bilder löschen
rm -rf echte_bilder/

# Schritte 3-6 wiederholen
```

---

## Alle Befehle im Überblick

```bash
# Virtuelle Umgebung aktivieren (jedes Mal beim neuen Terminal)
source .venv/bin/activate

# ROI neu kalibrieren
python3 roi_kalibrierung.py

# Unbekannt-Bilder automatisch sammeln (60 Sek.)
python3 fo_kennzeichen.py --auto --dauer 60

# Kennzeichen-Bilder manuell sammeln
python3 fo_kennzeichen.py --sammeln

# Von Null neu trainieren
python3 fo_kennzeichen.py --neu

# Weitertrainieren (Fortschritt bleibt)
python3 fo_kennzeichen.py --train

# Live-Erkennung starten
python3 fo_kennzeichen.py

# Training + sofort Live
python3 fo_kennzeichen.py --train --cam
```

---

## Whitelist ändern

Neue Kennzeichen in `fo_kennzeichen.py` eintragen:

```python
WHITELIST = [
    "MA AB1234",
    "B XY9999",
    "HD ZZ5678",
    "KA TT0001",
]
```

Danach **komplett neu starten:**
```bash
rm -rf echte_bilder/
python3 fo_kennzeichen.py --neu
```

---

## Arduino anschließen

Port herausfinden:
```bash
ls /dev/tty.usb*
```

In `fo_kennzeichen.py` eintragen:
```python
ARDUINO_PORT = "/dev/tty.usbmodem14201"  # deinen Port hier eintragen
```

Das Arduino-Sketch `schranke.ino` in der Arduino IDE öffnen und hochladen.
