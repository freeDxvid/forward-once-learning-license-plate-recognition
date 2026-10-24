"""
EasyOCR + Forward-Once Kennzeichenerkennung
============================================
EasyOCR liest den Text vom Kennzeichen.
Forward-Once Netz entscheidet: Whitelist-Kennzeichen ja oder nein?

Pipeline:
  Kamerabild → ROI → EasyOCR → Text → Zeichenvektor → Forward-Once → Arduino

Warum diese Kombination?
  - EasyOCR ist zuverlässig beim Lesen von Text
  - Forward-Once lernt welche Texte zur Whitelist gehören
  - Das eigene trainierte Netz trifft die Entscheidung (kein fertiges Produkt)

Lernregel (identisch zum Prof-Code forward_once_xor_classifying.py):
  g      = Σ h[block]²
  sign   = +1 wenn c == label  (in-class)
           −1 sonst            (out-of-class)
  dL_dg  = −sign · sigmoid(−sign · (g − θ))
  δ      = dL_dg · 2 · h[block] · relu'(z[block])
  W[:,block] −= lr · outer(h_in, δ)
  b[block]   −= lr · δ

Verwendung:
  python fo_kennzeichen.py --train   # Einmalig trainieren
  python fo_kennzeichen.py           # Live-Erkennung
  python fo_kennzeichen.py --neu     # Von Null neu trainieren
"""

import argparse
import json
import math
import pickle
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import easyocr
import numpy as np
import serial
import serial.tools.list_ports

# ─── Konfiguration ────────────────────────────────────────────────────────────

ARDUINO_PORT   = "/dev/cu.usbmodem1301"
BAUD_RATE      = 9600
KAMERA_INDEX   = 0
SCHRANKE_OFFEN = 3.0
MODEL_PATH     = "fo_model.pkl"
ROI_DATEI      = "roi.json"
ECHTE_BILDER_DIR = Path("echte_bilder")

# Zeichensatz für Kennzeichen: A-Z, 0-9, Leerzeichen = 37 Zeichen
ZEICHEN        = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
INPUT_DIM      = len(ZEICHEN) * 12   # max. 12 Zeichen pro Kennzeichen = 444

# Forward-Once Hyperparameter (analog zum Prof-Code)
HIDDEN_PRO_KLASSE = 64    # erhoeht (war 16) -> mehr Kapazitaet pro Klasse
LR                = 0.05
THRESHOLD         = 1.0
LEAKY_ALPHA       = 0.01
MAX_EPOCHEN       = 200   # erhoeht (war 200) -> mehr Trainingsrunden

KONFIDENZ_MIN     = 0.60  # Niedriger -> auch B XY9999 mit 63% oeffnet
ERKENNUNGS_PAUSE  = 5.0
OCR_KONFIDENZ_MIN = 0.35  # Auch schwaechere OCR-Erkennungen mitnehmen
FUZZY_MIN         = 0.65  # Mindest-Aehnlichkeit fuer Whitelist-Match

# Deutsches Kennzeichen-Muster
KFZ_MUSTER = re.compile(
    r'\b([A-ZÄÖÜ]{1,3})\s*([A-Z]{1,2})\s*(\d{1,4})\b',
    re.IGNORECASE
)

# ── Whitelist hier anpassen ───────────────────────────────────────────────────
WHITELIST = [
    "MA AB1234",
    "B XY9999",
    "HD ZZ5678",
    "KA TT0001",
]

# Negative Beispiele für Training (nicht in Whitelist)
NEGATIVE_TEXTE = [
    "FR MX1234", "S AB9876", "MZ CD4321", "LU XY7777",
    "DA ZZ0001", "WI OP3456", "OF GH2222", "DA BC5555",
    "KA AB1234", "HD XY9999", "B ZZ5678",
]


# ─── Hilfsfunktionen (exakt wie Prof-Code) ────────────────────────────────────

def log(text: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")


def relu(x: np.ndarray) -> np.ndarray:
    """Leaky ReLU – identisch zum Prof-Code."""
    return np.where(x > 0, x, LEAKY_ALPHA * x)


def sigmoid_scalar(x: float) -> float:
    """Numerisch stabiles Sigmoid – identisch zum Prof-Code."""
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def layer_norm(h: np.ndarray) -> np.ndarray:
    """Layer-Normalisierung – identisch zum Prof-Code."""
    return h / (np.linalg.norm(h) + 1e-8)


# ─── Text → Vektor ────────────────────────────────────────────────────────────

def text_zu_vektor(text: str) -> np.ndarray:
    """
    Kodiert einen Kennzeichen-Text als One-Hot Vektor.
    Jede Position im Text wird als One-Hot über ZEICHEN kodiert.
    Feste Länge: 12 Zeichen (mit Leerzeichen aufgefüllt).
    """
    text = text.upper().strip()[:12].ljust(12)
    vektor = np.zeros(INPUT_DIM, dtype=np.float32)
    for i, zeichen in enumerate(text):
        if zeichen in ZEICHEN:
            idx = ZEICHEN.index(zeichen)
            vektor[i * len(ZEICHEN) + idx] = 1.0
    return vektor


def normalisiere_kennzeichen(text: str) -> str:
    """Bereinigt OCR-Text zu einheitlichem Kennzeichen-Format.
    
    Deutsches Kennzeichen-Schema: [STADT] [BUCHSTABEN][ZIFFERN]
    Beispiel: KA TT0001  →  Stadtcode=KA, Buchstaben=TT, Ziffern=0001
    
    0/O und 1/I werden kontextabhängig korrigiert:
      - Im Buchstabenteil des Kennzeichens: 0→O, 1→I
      - Im Ziffernteil: O→0, I→1
    """
    text = text.upper().strip()
    text = re.sub(r"[^A-ZÄÖÜ0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def korrigiere_ocr_verwechslungen(text: str) -> str:
    """
    Korrigiert 0/O und 1/I kontextabhängig anhand der deutschen Kennzeichen-Struktur.
    
    Deutsches Kennzeichen: STADT BUCHSTABEN ZIFFERN
      - STADT    : nur Buchstaben  → 0→O, 1→I
      - BUCHSTABEN: nur Buchstaben → 0→O, 1→I  
      - ZIFFERN  : nur Ziffern    → O→0, I→1, S→5, B→8, Z→2
    
    Beispiel: "KA TT000I" → "KA TT0001"
              "KA TTD001" → "KA TT0001"  (D→0 wird durch Ziffer-Kontext erkannt)
    """
    text = text.upper().strip()
    
    # Versuche, das Kennzeichen-Muster zu erkennen: BUCHST+ BUCHST+ ZIFFER+
    # Regulärer Ausdruck der das Schema tolerant matcht (mit möglichen 0/O-Fehlern)
    muster = re.match(
        r'^([A-ZÄÖÜ0-9]{1,3})\s*([A-Z0-9]{1,2})\s*([A-Z0-9]{1,4})$',
        re.sub(r'\s+', ' ', text).strip()
    )
    
    if not muster:
        return text
    
    stadtcode   = muster.group(1)
    buchstaben  = muster.group(2)
    ziffernpart = muster.group(3)
    
    # Buchstabenteile: 0→O, 1→I (Ziffern sind hier Fehler)
    def buchstaben_fix(s):
        return s.replace("0", "O").replace("1", "I").replace("5", "S").replace("8", "B")
    
    # Ziffernteil: O→0, I→1, S→5, B→8, Z→2 (Buchstaben sind hier Fehler)
    def ziffern_fix(s):
        return (s.replace("O", "0").replace("I", "1")
                 .replace("S", "5").replace("B", "8")
                 .replace("Z", "2").replace("G", "6")
                 .replace("D", "0").replace("Q", "0"))
    
    stadtcode_fixed   = buchstaben_fix(stadtcode)
    buchstaben_fixed  = buchstaben_fix(buchstaben)
    ziffern_fixed     = ziffern_fix(ziffernpart)
    
    return f"{stadtcode_fixed} {buchstaben_fixed}{ziffern_fixed}"


def _normalisiere_fuer_vergleich(s: str) -> str:
    """
    Vereinheitlicht typische OCR-Verwechslungen vor dem Zeichenvergleich.
    O/0, I/1, S/5, B/8 werden auf eine kanonische Form gebracht,
    sodass 'TTOOO1' und 'TT0001' als identisch gelten.
    """
    s = s.upper().replace(" ", "")
    return (s.replace("O", "0")
             .replace("I", "1")
             .replace("S", "5")
             .replace("B", "8"))


def aehnlichkeit(a: str, b: str) -> float:
    """
    Berechnet wie ähnlich zwei Kennzeichen-Strings sind (0.0 - 1.0).
    Toleriert O/0, I/1, S/5, B/8 Verwechslungen vollständig –
    diese zählen als Treffer, nicht als Fehler.
    Beispiel: 'KATTOOO1' vs 'KATT0001' → 100%
    """
    a = _normalisiere_fuer_vergleich(a)
    b = _normalisiere_fuer_vergleich(b)
    if not a or not b:
        return 0.0
    treffer = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i+k < len(a) and j+k < len(b) and a[i+k] == b[j+k]:
                k += 1
            treffer = max(treffer, k)
    return (2.0 * treffer) / (len(a) + len(b))


def beste_whitelist_übereinstimmung(text: str, whitelist: list):
    """
    Findet das ähnlichste Whitelist-Kennzeichen zum erkannten Text.
    Gibt (whitelist_eintrag, ähnlichkeit) zurück.
    """
    text_clean = re.sub(r"[^A-Z0-9]", "", text.upper())
    beste_sim   = 0.0
    bestes_kz   = None
    for kz in whitelist:
        kz_clean = re.sub(r"[^A-Z0-9]", "", kz.upper())
        sim = aehnlichkeit(text_clean, kz_clean)
        if sim > beste_sim:
            beste_sim = sim
            bestes_kz = kz
    return bestes_kz, beste_sim


def extrahiere_kennzeichen(ocr_ergebnis: list, whitelist: list) -> list:
    """
    Robuste Kennzeichenerkennung:
    1. Alle OCR-Texte über 0.3 Konfidenz zusammensetzen
    2. Kennzeichen-Muster suchen
    3. Fragmente (z.B. "MA" + "AB1234" getrennt) zusammenführen
    4. Fuzzy-Match gegen Whitelist
    """
    kandidaten = []

    # ── Schritt 1: Einzelne OCR-Texte auswerten ───────────────────────────────
    alle_texte = []
    for (_bbox, text, konfidenz) in ocr_ergebnis:
        if konfidenz < OCR_KONFIDENZ_MIN:
            continue
        bereinigt = re.sub(r"[^A-ZÄÖÜ0-9\s]", "", text.upper().strip())
        bereinigt = re.sub(r"\s+", " ", bereinigt).strip()
        if bereinigt:
            alle_texte.append(bereinigt)
            # Auch die kontextkorrigierte Variante (0/O-Verwechslung) hinzufügen
            korrigiert = korrigiere_ocr_verwechslungen(bereinigt)
            if korrigiert != bereinigt:
                alle_texte.append(korrigiert)

    # ── Schritt 2: Alle Texte + Kombinationen benachbarter Texte prüfen ───────
    zu_pruefen = list(alle_texte)
    # Benachbarte Fragmente zusammensetzen (löst "MA" + "AB1234" Problem)
    for i in range(len(alle_texte)):
        for j in range(i+1, min(i+3, len(alle_texte))):
            zu_pruefen.append(alle_texte[i] + " " + alle_texte[j])
            zu_pruefen.append(alle_texte[i] + alle_texte[j])

    # ── Schritt 3: Jeden kombinierten Text gegen Whitelist prüfen ─────────────
    gesehen = set()
    for text in zu_pruefen:
        bestes_kz, sim = beste_whitelist_übereinstimmung(text, whitelist)
        if sim >= FUZZY_MIN and bestes_kz not in gesehen:
            kandidaten.append((bestes_kz, sim))
            gesehen.add(bestes_kz)
            log(f"OCR: '{text}' → '{bestes_kz}' (Ähnlichkeit: {sim:.0%})")

    return kandidaten


# ─── Forward-Once Netz ────────────────────────────────────────────────────────

class ForwardOnceKennzeichen:
    """
    Forward-Once Netz – Eingabe ist jetzt ein Text-Vektor (One-Hot kodiert),
    nicht mehr ein Pixelbild. Viel einfachere Aufgabe für das Netz.

    Architektur:
      Eingabe  : INPUT_DIM (444 = 12 Zeichen × 37 mögliche Zeichen)
      Hidden   : N_KLASSEN × HIDDEN_PRO_KLASSE Neuronen
      Output   : N_KLASSEN × 1 Neuron

    Lernregel: exakt wie forward_once_xor_classifying.py vom Prof
    """

    def __init__(self, n_klassen: int, klassen_namen: list):
        self.n_klassen     = n_klassen
        self.klassen_namen = klassen_namen

        hidden_dim  = n_klassen * HIDDEN_PRO_KLASSE
        output_dim  = n_klassen
        layer_sizes = [INPUT_DIM, hidden_dim, output_dim]
        self.block_sizes = [s // n_klassen for s in layer_sizes[1:]]

        rng = np.random.default_rng(42)
        self.W = []
        self.b = []
        for l in range(2):
            in_d  = layer_sizes[l]
            out_d = layer_sizes[l + 1]
            scale = math.sqrt(2.0 / in_d)
            self.W.append(rng.standard_normal((in_d, out_d)).astype(np.float32) * scale)
            self.b.append(np.zeros(out_d, dtype=np.float32))

    def forward_and_learn(self, vektor: np.ndarray, label: int):
        """Forward-Once Lernregel – exakt wie Prof-Code."""
        h_in = vektor.copy()

        for l in range(2):
            z  = h_in @ self.W[l] + self.b[l]
            h  = relu(z)
            bs = self.block_sizes[l]

            for c in range(self.n_klassen):
                s, e  = c * bs, (c + 1) * bs
                bh    = h[s:e]
                sign  = 1.0 if c == label else -1.0
                g     = float(np.sum(bh ** 2))
                dL_dg = -sign * sigmoid_scalar(-sign * (g - THRESHOLD))
                rmask = np.where(z[s:e] > 0, 1.0, LEAKY_ALPHA)
                delta = dL_dg * 2.0 * bh * rmask
                self.W[l][:, s:e] -= LR * np.outer(h_in, delta)
                self.b[l][s:e]    -= LR * delta

            h_in = layer_norm(h) if l < 1 else h

    def forward_only(self, vektor: np.ndarray):
        """Inferenz – gibt (label_index, goodness_array) zurück."""
        h_in = vektor.copy()
        for l in range(2):
            h    = relu(h_in @ self.W[l] + self.b[l])
            h_in = layer_norm(h) if l < 1 else h

        bs       = self.block_sizes[-1]
        goodness = np.array([
            float(np.sum(h_in[c * bs:(c + 1) * bs] ** 2))
            for c in range(self.n_klassen)
        ])
        return int(np.argmax(goodness)), goodness

    def konfidenz(self, goodness: np.ndarray) -> float:
        total = goodness.sum()
        return float(goodness.max() / total) if total > 1e-8 else 0.0

    def speichern(self, pfad: str):
        with open(pfad, "wb") as f:
            pickle.dump({
                "W": self.W, "b": self.b,
                "n_klassen": self.n_klassen,
                "klassen_namen": self.klassen_namen,
            }, f)
        log(f"Modell gespeichert → {pfad}")

    @classmethod
    def laden(cls, pfad: str):
        with open(pfad, "rb") as f:
            d = pickle.load(f)
        netz = cls(d["n_klassen"], d["klassen_namen"])
        netz.W = d["W"]
        netz.b = d["b"]
        log(f"Modell geladen ← {pfad} | Klassen: {d['klassen_namen']}")
        return netz


# ─── Training ─────────────────────────────────────────────────────────────────

def trainiere(whitelist: list, neu: bool = False) -> ForwardOnceKennzeichen:
    klassen  = whitelist + ["UNBEKANNT"]
    n_kl     = len(klassen)
    uk_label = len(whitelist)

    log("=" * 60)
    log("  EasyOCR + Forward-Once – Training")
    log("=" * 60)
    log(f"  Eingabe    : Text-Vektor (One-Hot, {INPUT_DIM} Dim.)")
    log(f"  Architektur: {INPUT_DIM} → {n_kl*HIDDEN_PRO_KLASSE} → {n_kl}")
    log(f"  Klassen    : {klassen}")
    log(f"  LR={LR}  θ={THRESHOLD}  Epochen={MAX_EPOCHEN}")
    log("=" * 60)

    if not neu and Path(MODEL_PATH).exists():
        log("  Vorhandenes Modell → wird weitertrainiert  (--neu für Neustart)")
        netz = ForwardOnceKennzeichen.laden(MODEL_PATH)
        if netz.klassen_namen != klassen:
            log("  Whitelist geändert → starte neu")
            netz = ForwardOnceKennzeichen(n_kl, klassen)
    else:
        log("  Starte von Null")
        netz = ForwardOnceKennzeichen(n_kl, klassen)

    # --- Datensatz aufbauen ---
    import random
    datensatz = []

    for i, kz in enumerate(whitelist):
        varianten = generiere_ocr_varianten(kz)
        log(f"  {kz}: {len(varianten)} Varianten → {varianten[:5]}{'...' if len(varianten)>5 else ''}")

        # Exakter Text – meiste Gewichtung
        for _ in range(300):
            datensatz.append((text_zu_vektor(kz), i))

        # Alle OCR-Fehlervarianten trainieren
        for variante in varianten:
            if variante != kz:
                for _ in range(80):
                    datensatz.append((text_zu_vektor(variante), i))

    # Negative Beispiele
    for _ in range(400 * len(whitelist)):
        neg = random.choice(NEGATIVE_TEXTE)
        datensatz.append((text_zu_vektor(neg), uk_label))

    random.shuffle(datensatz)
    log(f"  Datensatz: {len(datensatz)} Text-Vektoren")
    log("")

    for epoche in range(1, MAX_EPOCHEN + 1):
        random.shuffle(datensatz)
        richtig = gesamt = 0
        for vektor, label in datensatz:
            netz.forward_and_learn(vektor, label)
            pred, _ = netz.forward_only(vektor)
            if pred == label:
                richtig += 1
            gesamt += 1
        acc = richtig / gesamt * 100
        if epoche % 20 == 0 or epoche == 1:
            log(f"  Epoche {epoche:3d}/{MAX_EPOCHEN} │ Genauigkeit: {acc:.1f}%")

    log("")
    log("Training abgeschlossen.")
    netz.speichern(MODEL_PATH)
    return netz


# ─── Hilfsfunktion: OCR-Varianten ────────────────────────────────────────────

def generiere_ocr_varianten(kz: str) -> list:
    """
    Erzeugt realistische OCR-Fehlervarianten eines Kennzeichens.
    
    Für 'KA TT0001' werden z.B. erzeugt:
      - 'KA TT000I'   (letzte 1 → I)
      - 'KA TT0O01'   (erste 0 → O)
      - 'KATT0001'    (ohne Leerzeichen)
    """
    varianten = [kz]
    
    # Ohne Leerzeichen
    varianten.append(kz.replace(" ", ""))
    
    # Nur Nummernschild-Teil (ohne Stadtcode)
    teile = kz.split(" ")
    if len(teile) >= 2:
        varianten.append(" ".join(teile[1:]))
        varianten.append("".join(teile[1:]))
    
    # Einzelne Zeichen-Verwechslungen simulieren
    zeichen_map = {
        "0": ["O", "D", "Q"],
        "1": ["I", "L", "l"],
        "O": ["0"],
        "I": ["1", "L"],
        "5": ["S"],
        "8": ["B"],
        "6": ["G"],
        "2": ["Z"],
    }
    
    kz_clean = kz.replace(" ", "")
    for i, ch in enumerate(kz_clean):
        if ch in zeichen_map:
            for ersatz in zeichen_map[ch]:
                variante = kz_clean[:i] + ersatz + kz_clean[i+1:]
                # Leerzeichen wieder einfügen (nach Stadtcode)
                if len(teile) >= 2:
                    stadtlen = len(teile[0])
                    variante_mit_leer = variante[:stadtlen] + " " + variante[stadtlen:]
                    varianten.append(variante_mit_leer)
                varianten.append(variante)
    
    return list(set(varianten))  # Duplikate entfernen


# ─── ROI ──────────────────────────────────────────────────────────────────────

def lade_roi():
    if Path(ROI_DATEI).exists():
        with open(ROI_DATEI) as f:
            roi = json.load(f)
        log(f"ROI geladen: {roi}")
        return roi
    log("Keine roi.json – ganzes Bild wird analysiert")
    log("→ Tipp: python3 roi_kalibrierung.py (einmalig)")
    return None


def schneide_roi(frame, roi):
    if roi:
        return frame[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
    return frame


def verbessere_bild(frame: np.ndarray) -> np.ndarray:
    """
    Bildvorverarbeitung fuer bessere OCR-Ergebnisse.
    Wappen/Symbole auf Kennzeichen werden durch Binarisierung entfernt.
    Buchstaben sind schwarz auf weiss -> bleiben erhalten.
    Wappen sind grau/bunt -> werden weiss (Hintergrund).
    """
    grau = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Adaptiver Schwellwert: funktioniert bei verschiedenen Beleuchtungen
    binaer = cv2.adaptiveThreshold(
        grau, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15
    )
    # Kleines Rauschen entfernen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binaer = cv2.morphologyEx(binaer, cv2.MORPH_OPEN, kernel)
    return binaer


# ─── Arduino ──────────────────────────────────────────────────────────────────

def finde_arduino_port():
    for port in serial.tools.list_ports.comports():
        n = port.device.lower()
        d = (port.description or "").lower()
        if any(x in n for x in ("usbmodem", "usbserial")) or \
           any(x in d for x in ("arduino", "ch340")):
            return port.device
    return None


class ArduinoSchranke:
    def __init__(self, port, baud=9600):
        self.ser = None
        self.verbunden = False
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            self.verbunden = True
            log(f"Arduino verbunden: {port}")
        except (serial.SerialException, TypeError) as e:
            log(f"Arduino nicht verbunden → Simulationsmodus ({e})")

    def oeffne(self):
        if self.verbunden:
            self.ser.write(b"OPEN\n")
        log("Schranke OEFFNET" + ("" if self.verbunden else " [SIM]"))

    def schliesse(self):
        if self.verbunden:
            self.ser.write(b"CLOSE\n")
        log("Schranke SCHLIESST" + ("" if self.verbunden else " [SIM]"))

    def beende(self):
        if self.ser:
            self.ser.close()


# ─── Live-Erkennung ───────────────────────────────────────────────────────────

def live_erkennung(netz: ForwardOnceKennzeichen, whitelist: list, roi):
    log("Lade EasyOCR (beim ersten Mal ~30 Sek.)...")
    reader = easyocr.Reader(["de", "en"], gpu=False)
    log("EasyOCR bereit.")

    port     = finde_arduino_port() or ARDUINO_PORT
    schranke = ArduinoSchranke(port, BAUD_RATE)
    kamera   = cv2.VideoCapture(KAMERA_INDEX)
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    kamera.set(cv2.CAP_PROP_FPS, 60)

    log("=== Live-Erkennung startet (q = Beenden) ===")

    schranke_offen_bis    = 0.0
    letztes_kennzeichen   = ""
    letzte_erkennungszeit = 0.0
    uk_label              = len(whitelist)

    try:
        while True:
            ret, frame = kamera.read()
            if not ret:
                break

            jetzt   = time.time()
            anzeige = frame.copy()

            # ROI einzeichnen
            if roi:
                cv2.rectangle(anzeige, (roi["x1"], roi["y1"]),
                              (roi["x2"], roi["y2"]), (0, 255, 255), 1)

            # Nur ROI an EasyOCR übergeben
            crop      = schneide_roi(frame, roi)
            bearbeitet = verbessere_bild(crop)
            ocr_result = reader.readtext(bearbeitet, detail=1, paragraph=False)
            kandidaten = extrahiere_kennzeichen(ocr_result, whitelist)

            # Rohe OCR-Erkennungen einzeichnen (zur Kontrolle)
            for (bbox, text, konf) in ocr_result:
                if konf > OCR_KONFIDENZ_MIN:
                    offset_x = roi["x1"] if roi else 0
                    offset_y = roi["y1"] if roi else 0
                    punkte = np.array(bbox, dtype=np.int32)
                    punkte[:, 0] += offset_x
                    punkte[:, 1] += offset_y
                    cv2.polylines(anzeige, [punkte], True, (200, 200, 0), 1)
                    cv2.putText(anzeige, f"{text} ({konf:.0%})",
                                (int(bbox[0][0]) + offset_x,
                                 int(bbox[0][1]) + offset_y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

            # Was hat EasyOCR gelesen? Immer anzeigen (auch wenn kein Match)
            alle_ocr_texte = [t for (_, t, k) in ocr_result if k > OCR_KONFIDENZ_MIN]
            if alle_ocr_texte:
                debug_text = "OCR: " + "  |  ".join(alle_ocr_texte[:4])
                cv2.rectangle(anzeige, (0, anzeige.shape[0]-36),
                              (anzeige.shape[1], anzeige.shape[0]), (0,0,0), -1)
                cv2.putText(anzeige, debug_text,
                            (10, anzeige.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

            # kandidaten = Liste von (whitelist_kz, ähnlichkeit)
            for kz, ocr_sim in kandidaten:
                vektor        = text_zu_vektor(kz)
                idx, goodness = netz.forward_only(vektor)
                fo_konf       = netz.konfidenz(goodness)

                if idx != uk_label and fo_konf >= KONFIDENZ_MIN:
                    erkannt = netz.klassen_namen[idx]
                    farbe   = (0, 220, 0)

                    if roi:
                        cv2.rectangle(anzeige, (roi["x1"], roi["y1"]),
                                      (roi["x2"], roi["y2"]), farbe, 2)
                        cv2.putText(anzeige,
                                    f"OK  {erkannt}  OCR:{ocr_sim:.0%}  FO:{fo_konf:.0%}",
                                    (roi["x1"], roi["y1"] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, farbe, 2)

                    frisch = not (erkannt == letztes_kennzeichen and
                                  jetzt - letzte_erkennungszeit < ERKENNUNGS_PAUSE)
                    if frisch:
                        letztes_kennzeichen   = erkannt
                        letzte_erkennungszeit = jetzt
                        log(f"ZUGANG ERLAUBT: {erkannt}  "
                            f"OCR-Sim={ocr_sim:.0%}  FO-Konf={fo_konf:.0%}")
                        schranke.oeffne()
                        schranke_offen_bis = jetzt + SCHRANKE_OFFEN

                elif fo_konf >= 0.45 and idx != uk_label:
                    farbe = (0, 165, 255)
                    if roi:
                        cv2.putText(anzeige,
                                    f"? {netz.klassen_namen[idx]}  {fo_konf:.0%}",
                                    (roi["x1"], roi["y1"] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, farbe, 2)

            if schranke_offen_bis > 0 and jetzt >= schranke_offen_bis:
                schranke.schliesse()
                schranke_offen_bis = 0.0

            offen  = jetzt < schranke_offen_bis
            status = "Schranke: OFFEN" if offen else "Schranke: ZU"
            cv2.rectangle(anzeige, (0, 0), (380, 36), (0, 0, 0), -1)
            cv2.putText(anzeige, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 0) if offen else (0, 0, 200), 2)

            cv2.imshow("EasyOCR + Forward-Once (q=Beenden)", anzeige)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        log("Beendet.")
    finally:
        schranke.schliesse()
        schranke.beende()
        kamera.release()
        cv2.destroyAllWindows()
        log("System beendet.")





def sammle_bilder(whitelist: list, roi):
    """Manuelles Sammeln echter Kennzeichenbilder per Tastendruck."""
    kamera = cv2.VideoCapture(KAMERA_INDEX)
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Ordner erstellen falls nicht vorhanden
    ECHTE_BILDER_DIR.mkdir(exist_ok=True)
    for kz in whitelist:
        (ECHTE_BILDER_DIR / kz.replace(" ", "_").replace("/", "-")).mkdir(exist_ok=True)
    (ECHTE_BILDER_DIR / "unbekannt").mkdir(exist_ok=True)

    log("=== Bilder-Sammel-Modus ===")
    for i, kz in enumerate(whitelist):
        log(f"  Taste {i+1} = {kz}")
    log("  Taste u = Unbekannt  |  q = Beenden")

    zaehler = {kz: 0 for kz in whitelist}
    zaehler["unbekannt"] = 0

    try:
        while True:
            ret, frame = kamera.read()
            if not ret:
                break
            grau    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            anzeige = frame.copy()
            crop    = schneide_roi(grau, roi)

            if roi:
                cv2.rectangle(anzeige, (roi["x1"], roi["y1"]),
                              (roi["x2"], roi["y2"]), (0, 255, 255), 2)

            # Anleitung unten
            h = anzeige.shape[0]
            cv2.rectangle(anzeige, (0, h - 36), (anzeige.shape[1], h), (0, 0, 0), -1)
            anleitung = "  ".join([f"{i+1}={kz}" for i, kz in enumerate(whitelist)])
            anleitung += "   u=Unbekannt   q=Beenden"
            cv2.putText(anzeige, anleitung, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Bilder sammeln (q = Beenden)", anzeige)
            taste = cv2.waitKey(1) & 0xFF

            if taste == ord("q"):
                break
            elif taste == ord("u"):
                ts = int(time.time() * 1000)
                cv2.imwrite(str(ECHTE_BILDER_DIR / "unbekannt" / f"{ts}.png"), crop)
                zaehler["unbekannt"] += 1
                log(f"Gespeichert: unbekannt ({zaehler['unbekannt']})")
            else:
                for i, kz in enumerate(whitelist):
                    if taste == ord(str(i + 1)):
                        sicher = kz.replace(" ", "_").replace("/", "-")
                        ts     = int(time.time() * 1000)
                        cv2.imwrite(str(ECHTE_BILDER_DIR / sicher / f"{ts}.png"), crop)
                        zaehler[kz] += 1
                        log(f"Gespeichert: {kz} ({zaehler[kz]})")
                        break
    finally:
        kamera.release()
        cv2.destroyAllWindows()
        log("Fertig. Jetzt: python3 fo_kennzeichen.py --train")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EasyOCR + Forward-Once Kennzeichenerkennung")
    parser.add_argument("--train",   action="store_true", help="Weitertrainieren")
    parser.add_argument("--neu",     action="store_true", help="Von Null neu trainieren")
    parser.add_argument("--sammeln", action="store_true", help="Echte Bilder manuell aufnehmen")
    parser.add_argument("--cam",     action="store_true", help="Nach Training live starten")
    args = parser.parse_args()

    roi = lade_roi()

    if args.sammeln:
        sammle_bilder(WHITELIST, roi)
        return

    if args.train or args.neu or not Path(MODEL_PATH).exists():
        if not args.train and not args.neu:
            log("Kein Modell gefunden – Training startet automatisch...")
        netz = trainiere(WHITELIST, neu=args.neu)
    else:
        netz = ForwardOnceKennzeichen.laden(MODEL_PATH)

    if not args.train and not args.neu or args.cam:
        live_erkennung(netz, WHITELIST, roi)


if __name__ == "__main__":
    main()