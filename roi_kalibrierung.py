"""
roi_kalibrierung.py
===================
Einmalig ausführen, um die Zone zu markieren wo das Kennzeichen erscheint.
Speichert roi.json → wird von fo_kennzeichen.py automatisch geladen.

Steuerung:
  Maus ziehen  → ROI aufziehen
  ENTER        → Speichern & Beenden
  r            → Neu zeichnen
  q            → Abbrechen
"""

import cv2
import json
from pathlib import Path

KAMERA_INDEX = 0
ROI_DATEI    = "roi.json"

ziehen = False
start  = (0, 0)
ende   = (0, 0)
fertig = False


def maus_callback(event, x, y, flags, param):
    global ziehen, start, ende, fertig
    if event == cv2.EVENT_LBUTTONDOWN:
        ziehen = True
        fertig = False
        start  = (x, y)
        ende   = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and ziehen:
        ende = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        ziehen = False
        ende   = (x, y)
        fertig = True


def main():
    global start, ende, fertig

    kamera = cv2.VideoCapture(KAMERA_INDEX)
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not kamera.isOpened():
        print("Kamera nicht gefunden!")
        return

    cv2.namedWindow("ROI Kalibrierung")
    cv2.setMouseCallback("ROI Kalibrierung", maus_callback)

    print()
    print("ROI-Kalibrierung")
    print("================")
    print("  Mit Maus den Bereich ziehen wo das Kennzeichen erscheint")
    print("  ENTER = Speichern   r = Neu   q = Abbrechen")
    print()

    while True:
        ret, frame = kamera.read()
        if not ret:
            break

        anzeige = frame.copy()

        if fertig or ziehen:
            x1 = min(start[0], ende[0])
            y1 = min(start[1], ende[1])
            x2 = max(start[0], ende[0])
            y2 = max(start[1], ende[1])

            maske = anzeige.copy()
            maske[:] = (0, 0, 0)
            maske[y1:y2, x1:x2] = anzeige[y1:y2, x1:x2]
            anzeige = cv2.addWeighted(anzeige, 0.3, maske, 0.7, 0)
            farbe   = (0, 255, 0) if fertig else (0, 200, 255)
            cv2.rectangle(anzeige, (x1, y1), (x2, y2), farbe, 2)

            if fertig:
                info = f"ROI: {x2-x1}x{y2-y1}px  |  ENTER=Speichern  r=Neu"
                cv2.rectangle(anzeige, (0, 0), (560, 36), (0, 0, 0), -1)
                cv2.putText(anzeige, info, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            cv2.rectangle(anzeige, (0, 0), (620, 36), (0, 0, 0), -1)
            cv2.putText(anzeige, "Bereich ziehen wo Kennzeichen erscheint",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        cv2.imshow("ROI Kalibrierung", anzeige)
        taste = cv2.waitKey(1) & 0xFF

        if taste == 13 and fertig:  # ENTER
            x1 = min(start[0], ende[0])
            y1 = min(start[1], ende[1])
            x2 = max(start[0], ende[0])
            y2 = max(start[1], ende[1])
            if (x2 - x1) < 20 or (y2 - y1) < 10:
                print("ROI zu klein – bitte groeßeren Bereich waehlen.")
                fertig = False
                continue
            roi = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            with open(ROI_DATEI, "w") as f:
                json.dump(roi, f)
            print(f"ROI gespeichert: {roi}")
            break
        elif taste == ord("r"):
            fertig = False
            start  = (0, 0)
            ende   = (0, 0)
        elif taste == ord("q"):
            print("Abgebrochen.")
            break

    kamera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
