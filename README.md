# Forward-Once Kennzeichenerkennung

KI-Schrankensteuerung mit Forward-Once Learning (keine Backpropagation).

## Was ist Forward-Once?

Statt Backpropagation lernt jede Schicht **lokal** anhand einer "Goodness"-Funktion:
- Für das richtige Kennzeichen → hohe Goodness anstreben
- Für falsche Kennzeichen → niedrige Goodness anstreben
- Klassifikation: Klasse mit der höchsten Goodness gewinnt

Die Lernregel ist exakt dieselbe wie in `forward_once_xor_classifying.py`.

## Projektstruktur

```
kennzeichen_projekt/
├── fo_kennzeichen.py        → Hauptprogramm (Forward-Once + Kamera + Arduino)
├── forward_once_xor_classifying.py → Prof-Referenz (unverändert)
├── roi_kalibrierung.py      → ROI einmalig kalibrieren
├── schranke.ino             → Arduino-Sketch (unverändert)
├── requirements.txt         → Python-Abhängigkeiten (kein PyTorch nötig!)
├── start.sh                 → Starter-Skript
├── fo_model.pkl             → Trainiertes Modell (wird automatisch erstellt)
└── README.md                → Diese Datei
```

## Schnellstart

```bash
cd kennzeichen_projekt
chmod +x start.sh
./start.sh
```

## Empfohlener Ablauf für beste Ergebnisse

```bash
# 1. ROI einmalig kalibrieren (Bereich wo Kennzeichen erscheint markieren)
python roi_kalibrierung.py

# 2. Trainieren
python fo_kennzeichen.py --train

# 3. Live starten
python fo_kennzeichen.py

# 4. Optional: Echte Bilder sammeln (verbessert Genauigkeit erheblich)
python fo_kennzeichen.py --sammeln
# → danach nochmal trainieren:
python fo_kennzeichen.py --train
```

## Whitelist anpassen

In `fo_kennzeichen.py`:
```python
WHITELIST = [
    "MA AB1234",
    "B XY9999",
]
```
Nach jeder Änderung neu trainieren: `python fo_kennzeichen.py --train`

## Konfiguration

| Variable          | Standard | Bedeutung                              |
|-------------------|----------|----------------------------------------|
| `HIDDEN_PRO_KLASSE` | 32     | Neuronen pro Klassen-Block (Hidden)    |
| `LR`              | 0.01     | Lernrate                               |
| `THRESHOLD`       | 2.0      | Goodness-Schwellwert θ                 |
| `MAX_EPOCHEN`     | 80       | Trainingsepochen                       |
| `BILDER_PRO_KZ`   | 400      | Synthetische Bilder pro Kennzeichen    |
| `KONFIDENZ_MIN`   | 0.70     | Mindest-Konfidenz für Schranke öffnen  |
| `SCHRANKE_OFFEN`  | 3.0      | Sekunden offen                         |

## Arduino

Port in `fo_kennzeichen.py` anpassen:
```python
ARDUINO_PORT = "/dev/tty.usbmodem14201"
```
Port finden: `ls /dev/tty.usb*`

## Tastenkürzel

| Taste | Funktion |
|-------|----------|
| `q`   | Beenden  |
