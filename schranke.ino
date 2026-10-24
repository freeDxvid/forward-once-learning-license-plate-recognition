/*
  Schrankensteuerung – Arduino Micro
  ====================================
  Servo an Pin 9, eingebaute LED an Pin 13.
  Empfängt OPEN / CLOSE über Serial von fo_kennzeichen.py.

  Verkabelung:
    Servo Orange  → Pin 9
    Servo Schwarz → GND
    Servo Rot     → 5V
*/

#include <Servo.h>

const int SERVO_PIN      = 9;
const int LED_BUILTIN_PIN = 13;

const int WINKEL_ZU    = 123;
const int WINKEL_OFFEN = 40;
const int SERVO_PAUSE  = 15;

Servo schranke;
String empfangen = "";
bool schranke_offen = false;
int aktuellerWinkel = WINKEL_ZU;

void fahreZu(int ziel) {
  int schritt = (ziel > aktuellerWinkel) ? 1 : -1;
  while (aktuellerWinkel != ziel) {
    aktuellerWinkel += schritt;
    schranke.write(aktuellerWinkel);
    delay(SERVO_PAUSE);
  }
}

void oeffne() {
  schranke_offen = true;
  digitalWrite(LED_BUILTIN_PIN, HIGH);
  Serial.println(">>> SCHRANKE OEFFNET <<<");
  fahreZu(WINKEL_OFFEN);
  Serial.println("STATUS:OFFEN");
}

void schliesse() {
  schranke_offen = false;
  fahreZu(WINKEL_ZU);
  digitalWrite(LED_BUILTIN_PIN, LOW);
  Serial.println(">>> SCHRANKE SCHLIESST <<<");
  Serial.println("STATUS:ZU");
}

void setup() {
  Serial.begin(9600);

  pinMode(LED_BUILTIN_PIN, OUTPUT);
  digitalWrite(LED_BUILTIN_PIN, LOW);

  schranke.attach(SERVO_PIN);
  schranke.write(WINKEL_ZU);
  aktuellerWinkel = WINKEL_ZU;

  Serial.println("=== Arduino Schrankensteuerung ===");
  Serial.println("Warte auf Befehle: OPEN | CLOSE | STATUS");
  Serial.println("==================================");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      empfangen.trim();

      if (empfangen == "OPEN") {
        oeffne();

      } else if (empfangen == "CLOSE") {
        schliesse();

      } else if (empfangen == "STATUS") {
        Serial.print("Aktueller Zustand: ");
        Serial.println(schranke_offen ? "OFFEN" : "ZU");

      } else if (empfangen.length() > 0) {
        Serial.print("Unbekannter Befehl: '");
        Serial.print(empfangen);
        Serial.println("'");
      }

      empfangen = "";
    } else {
      empfangen += c;
    }
  }
}
