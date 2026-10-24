#!/bin/bash
# ═══════════════════════════════════════════════════════
#  Forward-Once Kennzeichenerkennung – Starter für Mac
#  Verwendung: ./start.sh
# ═══════════════════════════════════════════════════════

set -e
VENV_DIR=".venv"

echo ""
echo "Forward-Once Kennzeichenerkennung"
echo "=================================="

if ! command -v python3 &>/dev/null; then
  echo "Python3 nicht gefunden. Installiere: brew install python"
  exit 1
fi
echo "Python: $(python3 --version)"

if [ ! -d "$VENV_DIR" ]; then
  echo "Erstelle virtuelle Umgebung..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

if ! python -c "import cv2" &>/dev/null 2>&1; then
  echo "Installiere Pakete (einmalig ~1 Min.)..."
  pip install --upgrade pip -q
  pip install -r requirements.txt -q
  echo "Pakete installiert."
fi

echo ""
echo "Tipp: Kamera nicht erkannt?"
echo "  Systemeinstellungen → Datenschutz → Kamera → Terminal aktivieren"
echo ""
echo "Starte Forward-Once Kennzeichenerkennung..."
echo "  (Kein Modell? Training startet automatisch, ~2 Min.)"
echo "  (q im Kamerafenster = Beenden)"
echo ""
python fo_kennzeichen.py
