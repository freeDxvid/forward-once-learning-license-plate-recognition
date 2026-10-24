"""
Forward-Once Network – XOR Klassifikation
==========================================
Architektur:   3  →  4  →  2
               ↑     ↑     ↑
             Eingabe Hidden Output
            [bias,a,b]  (4N)  (2N)

Eingabe-Vektor : [bias=1, a, b]
Hidden-Schicht : 4 Neuronen → 2 Klassen-Blöcke à 2 Neuronen
Output-Schicht : 2 Neuronen → 2 Klassen-Blöcke à 1 Neuron

Klasse 0 : a XOR b  = 1   →  Ausgang 1 aktiv   (Eingaben: (0,1), (1,0))
Klasse 1 : NOT(a XOR b)=1 →  Ausgang 2 aktiv   (Eingaben: (0,0), (1,1))

Forward-Once Lernregel pro Klassen-Block c in Schicht l:
  g      = Σ h[block]²
  sign   = +1  wenn c == label   (in-class-Block:  Goodness > θ anstreben)
           −1  sonst             (out-of-class-Block: Goodness < θ anstreben)
  dL/dg  = −sign · sigmoid(−sign · (g − θ))
  δ      = dL/dg · 2 · h[block] · relu'(z[block])
  W[:,block] −= lr · outer(h_in, δ)
  b[block]   −= lr · δ

Klassifikation: argmax( Goodness pro Block ) im Output-Layer.
"""

import numpy as np
import math

SEED = 42
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Architektur & Hyperparameter
# ──────────────────────────────────────────────────────────────────────────────

LAYER_SIZES = [3, 4, 2]                                    # input | hidden | output
N_LAYERS    = len(LAYER_SIZES) - 1                         # 2 trainierbare Schichten
N_CLASSES   = 2
BLOCK_SIZES = [s // N_CLASSES for s in LAYER_SIZES[1:]]   # [2, 1]

LR         = 0.05    # Lernrate
THRESHOLD  = 0.5     # Goodness-Schwellwert θ  (kleines Netz → kleiner θ)
MAX_EPOCHS = 20_000  # Maximale Trainingsepochen

# ──────────────────────────────────────────────────────────────────────────────
# 2.  XOR-Datensatz
# ──────────────────────────────────────────────────────────────────────────────
#
#   Eingabe [bias=1, a, b]          a XOR b    Label
#   ──────────────────────────────────────────────────
#   [1, 0, 0]  →  0 XOR 0 = 0       0          Klasse 1: NOT XOR
#   [1, 0, 1]  →  0 XOR 1 = 1       1          Klasse 0: XOR
#   [1, 1, 0]  →  1 XOR 0 = 1       1          Klasse 0: XOR
#   [1, 1, 1]  →  1 XOR 1 = 0       0          Klasse 1: NOT XOR

XOR_DATA = [
    (np.array([1.0, 0.0, 0.0], dtype=np.float32), 1),   # NOT XOR
    (np.array([1.0, 0.0, 1.0], dtype=np.float32), 0),   # XOR
    (np.array([1.0, 1.0, 0.0], dtype=np.float32), 0),   # XOR
    (np.array([1.0, 1.0, 1.0], dtype=np.float32), 1),   # NOT XOR
]

CLASS_NAMES = ["XOR", "NOT XOR"]

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────

LEAKY_ALPHA = 0.01   # Leaky-ReLU Steigung für negative Eingaben

def relu(x):
    """Leaky ReLU: verhindert 'tote' Neuronen (Gradient ≠ 0 auch für x<0)."""
    return np.where(x > 0, x, LEAKY_ALPHA * x)

def sigmoid_scalar(x):
    """Numerisch stabiles Sigmoid für Skalare."""
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))

def layer_norm(h):
    """Normalisiert auf Einheitsvektor (verhindert explodierendes Wachstum)."""
    return h / (np.linalg.norm(h) + 1e-8)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Forward-Once Netzwerk
# ──────────────────────────────────────────────────────────────────────────────

class ForwardOnceXOR:
    """
    2-schichtiges Forward-Once Netz für XOR vs. NOT-XOR Klassifikation.

    Schicht 1 – Hidden (4 Neuronen):
        Block 0  [Neuronen 0–1]  →  Repräsentation für Klasse 0 (XOR)
        Block 1  [Neuronen 2–3]  →  Repräsentation für Klasse 1 (NOT XOR)

    Schicht 2 – Output (2 Neuronen):
        Block 0  [Neuron 0]  →  Goodness-Ausgang für Klasse 0 (XOR)
        Block 1  [Neuron 1]  →  Goodness-Ausgang für Klasse 1 (NOT XOR)

    Training :  Forward-Once Lernregel (kein Backprop durch Schichten)
    Klassifikation: argmax( g₀, g₁ ) im Output-Layer
    """

    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.W = []
        self.b = []
        for l in range(N_LAYERS):
            in_dim  = LAYER_SIZES[l]
            out_dim = LAYER_SIZES[l + 1]
            scale   = math.sqrt(2.0 / in_dim)
            self.W.append(
                rng.standard_normal((in_dim, out_dim)).astype(np.float32) * scale
            )
            self.b.append(np.zeros(out_dim, dtype=np.float32))

    # ── Forward Pass mit Gewichtsaktualisierung ───────────────────────────────
    def forward_and_learn(self, x: np.ndarray, label: int):
        """
        Propagiert x vorwärts und wendet die Forward-Once Lernregel an.
        Kein Backprop – jede Schicht lernt lokal anhand ihres eigenen Fehlers.
        """
        h_in = x.copy()   # Eingabe: [1, a, b]  – Bias=1 ist bereits enthalten

        for l in range(N_LAYERS):
            z  = h_in @ self.W[l] + self.b[l]   # Linearkombination
            h  = relu(z)                          # Aktivierung
            bs = BLOCK_SIZES[l]

            # ── Forward-Once Update: ein Update pro Klassen-Block ──────────
            for c in range(N_CLASSES):
                s, e  = c * bs, (c + 1) * bs    # Blockgrenzen
                bh    = h[s:e]                   # Block-Aktivierungen

                # In-Class  (c == label): Goodness über θ drücken  (+1)
                # Out-Class (c != label): Goodness unter θ drücken (−1)
                sign  = 1.0 if c == label else -1.0

                g     = float(np.sum(bh ** 2))                          # Goodness
                dL_dg = -sign * sigmoid_scalar(-sign * (g - THRESHOLD)) # ∂L/∂g
                rmask = np.where(z[s:e] > 0, 1.0, LEAKY_ALPHA)  # Leaky-ReLU Ableitung
                delta = dL_dg * 2.0 * bh * rmask                        # Gradient

                self.W[l][:, s:e] -= LR * np.outer(h_in, delta)
                self.b[l][s:e]    -= LR * delta

            # Layer-Normalisierung als Eingabe für die nächste Schicht
            # (letzte Schicht: kein layer_norm, damit Goodness unverfälscht bleibt)
            h_in = layer_norm(h) if l < N_LAYERS - 1 else h

    # ── Inferenz ohne Gewichtsaktualisierung ──────────────────────────────────
    def forward_only(self, x: np.ndarray):
        """
        Gibt (predicted_label, goodness_array) zurück.
        goodness_array[c] = Σ output_block_c²
        """
        h_in = x.copy()

        for l in range(N_LAYERS):
            h    = relu(h_in @ self.W[l] + self.b[l])
            h_in = layer_norm(h) if l < N_LAYERS - 1 else h

        # Goodness pro Block im Output-Layer
        bs       = BLOCK_SIZES[-1]
        goodness = np.array([
            float(np.sum(h_in[c * bs:(c + 1) * bs] ** 2))
            for c in range(N_CLASSES)
        ])
        return int(np.argmax(goodness)), goodness

    def accuracy(self) -> float:
        """Berechnet Accuracy über alle 4 XOR-Beispiele."""
        correct = sum(
            1 for x, label in XOR_DATA
            if self.forward_only(x)[0] == label
        )
        return correct / len(XOR_DATA)

    def hidden_goodness(self, x: np.ndarray) -> np.ndarray:
        """Hilfsfunktion: Gibt Goodness pro Block in der Hidden-Schicht zurück."""
        h = relu(x @ self.W[0] + self.b[0])
        bs = BLOCK_SIZES[0]
        return np.array([
            float(np.sum(h[c * bs:(c + 1) * bs] ** 2))
            for c in range(N_CLASSES)
        ])


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Training
# ──────────────────────────────────────────────────────────────────────────────

def train() -> ForwardOnceXOR:
    print("=" * 64)
    print("  Forward-Once Network  –  XOR Klassifikation")
    print("=" * 64)
    arch = " → ".join(str(s) for s in LAYER_SIZES)
    print(f"  Architektur : {arch}")
    print(f"  Hidden-Blöcke: {N_CLASSES} × {BLOCK_SIZES[0]} Neuronen  "
          f"| Output-Blöcke: {N_CLASSES} × {BLOCK_SIZES[1]} Neuron")
    print(f"  Lernregel   : Forward-Once (kein Backprop)")
    print(f"  LR          : {LR}   θ: {THRESHOLD}   Max-Epochen: {MAX_EPOCHS}")
    print("=" * 64)
    print(f"\n  {'Epoche':>8}  │  {'Accuracy':>8}  │  Status")
    print(f"  {'─'*8}  ┼  {'─'*8}  ┼  {'─'*30}")

    net       = ForwardOnceXOR()
    log_every = MAX_EPOCHS // 20
    converged_epoch = None

    for epoch in range(1, MAX_EPOCHS + 1):
        # Zufällige Reihenfolge der 4 Beispiele pro Epoche
        indices = np.random.permutation(len(XOR_DATA))
        for i in indices:
            x, label = XOR_DATA[i]
            net.forward_and_learn(x, label)

        acc = net.accuracy()

        if epoch % log_every == 0 or epoch == 1:
            status = "konvergiert ✓" if acc == 1.0 else "in Arbeit ..."
            print(f"  {epoch:>8d}  │  {acc*100:>7.1f}%  │  {status}")

        if acc == 1.0 and converged_epoch is None:
            converged_epoch = epoch
            if epoch % log_every != 0:   # noch nicht gedruckt
                print(f"  {epoch:>8d}  │  {acc*100:>7.1f}%  │  konvergiert ✓")
            print(f"\n  ► Erste vollständige Konvergenz: Epoche {epoch}\n")

    if converged_epoch is None:
        print(f"\n  ► Keine Konvergenz in {MAX_EPOCHS} Epochen erreicht.\n")

    return net


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Ergebnisausgabe
# ──────────────────────────────────────────────────────────────────────────────

def print_truth_table(net: ForwardOnceXOR):
    """Druckt die vollständige Wahrheitstafel mit Goodness-Werten."""
    print("=" * 70)
    print("  Wahrheitstafel  –  Ergebnis nach Training")
    print("=" * 70)
    print(f"  {'a':>2}  {'b':>2}  │  {'Erwartet':>9}  {'Vorhergesagt':>13}  "
          f"{'G₀(XOR)':>9}  {'G₁(¬XOR)':>9}  {'✓/✗':>3}")
    print(f"  {'─'*2}  {'─'*2}  ┼  {'─'*9}  {'─'*13}  {'─'*9}  {'─'*9}  {'─'*3}")

    all_ok = True
    for x, label in XOR_DATA:
        a, b           = int(x[1]), int(x[2])
        pred, goodness = net.forward_only(x)
        ok             = pred == label
        all_ok         = all_ok and ok
        mark           = "✓" if ok else "✗"
        print(f"  {a:>2}  {b:>2}  │  {CLASS_NAMES[label]:>9}  "
              f"{CLASS_NAMES[pred]:>13}  "
              f"{goodness[0]:>9.4f}  {goodness[1]:>9.4f}  {mark:>3}")

    print("=" * 70)
    result = "ALLE 4 MUSTER KORREKT KLASSIFIZIERT  ✓" if all_ok else "FEHLER VORHANDEN  ✗"
    print(f"  Ergebnis: {result}")
    print("=" * 70)


def print_goodness_analysis(net: ForwardOnceXOR):
    """Zeigt Goodness-Werte in Hidden- und Output-Schicht für alle Eingaben."""
    print("\n  Goodness-Analyse pro Schicht:")
    print("=" * 70)
    print(f"  {'Eingabe':>14}  │  {'Hidden G₀':>10} {'Hidden G₁':>10}  │  "
          f"{'Out G₀':>8} {'Out G₁':>8}")
    print(f"  {'─'*14}  ┼  {'─'*10} {'─'*10}  ┼  {'─'*8} {'─'*8}")

    for x, label in XOR_DATA:
        a, b            = int(x[1]), int(x[2])
        hg              = net.hidden_goodness(x)
        _,  og          = net.forward_only(x)
        tag             = f"[{a} XOR {b}={a^b}]"
        print(f"  {tag:>14}  │  {hg[0]:>10.4f} {hg[1]:>10.4f}  │  "
              f"{og[0]:>8.4f} {og[1]:>8.4f}")

    print(f"  {'─'*14}  ┴  {'─'*10} {'─'*10}  ┴  {'─'*8} {'─'*8}")
    print(f"  Schwellwert θ  │  {'< '+str(THRESHOLD)+' (out)':>10} {'> '+str(THRESHOLD)+' (in)':>10}  │  "
          f"{'< '+str(THRESHOLD):>8} {'> '+str(THRESHOLD):>8}")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    net = train()
    print_truth_table(net)
    print_goodness_analysis(net)


if __name__ == "__main__":
    main()