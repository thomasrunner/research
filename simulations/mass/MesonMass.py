from dataclasses import dataclass
from typing import Tuple

@dataclass
class Meson:
    name: str
    mass: float
    spin: float
    isospin: float
    isospin3: float
    quarks: Tuple[str, str]
    state: int

mesons = [
    Meson("Pi0", 134.9766, 0.0, 1.0, 0.0, ("u", "d"), 1),
    Meson("Pi+", 139.57039, 0.0, 1.0, 1.0, ("u", "d"), 1),
    Meson("Pi-", 139.57039, 0.0, 1.0, -1.0, ("u", "d"), 1),
    Meson("K+", 493.677, 0.0, 0.5, 0.5, ("u", "s"), 1),
    Meson("K-", 493.677, 0.0, 0.5, -0.5, ("u", "s"), 1),
    Meson("K0", 497.611, 0.0, 0.5, -0.5, ("d", "s"), 1),
    Meson("K̄0", 497.611, 0.0, 0.5, 0.5, ("d", "s"), 1),
    Meson("K0_S", 497.611, 0.0, 0.5, 0.5, ("d", "s"), 1),
    Meson("K0_L", 497.611, 0.0, 0.5, 0.5, ("d", "s"), 1),
    Meson("Rho+", 775.11, 1.0, 1.0, 1.0, ("u", "d"), 1),
    Meson("Rho-", 775.11, 1.0, 1.0, -1.0, ("u", "d"), 1),
    Meson("D0", 1864.841, 0.0, 0.5, -0.5, ("c", "u"), 1),
    Meson("D̄0", 1864.841, 0.0, 0.5, 0.5, ("c", "u"), 1),
    Meson("D+", 1869.65, 0.0, 0.5, 0.5, ("c", "d"), 1),
    Meson("D-", 1869.65, 0.0, 0.5, -0.5, ("c", "d"), 1),
    Meson("Ds+", 1968.34, 0.0, 0.0, 0.0, ("c", "s"), 1),
    Meson("Ds-", 1968.34, 0.0, 0.0, 0.0, ("c", "s"), 1),
    Meson("J/Psi", 3096.9, 1.0, 0.0, 0.0, ("c", "c"), 1),
    Meson("Psi(2S)", 3686.1, 1.0, 0.0, 0.0, ("c", "c"), 2),
    Meson("B+", 5279.1, 0.0, 0.5, 0.5, ("b", "u"), 1),
    Meson("B-", 5279.1, 0.0, 0.5, -0.5, ("b", "u"), 1),
    Meson("B0", 5279.1, 0.0, 0.5, -0.5, ("b", "d"), 1),
    Meson("B̄0", 5279.1, 0.0, 0.5, 0.5, ("b", "d"), 1),
    Meson("Bs0", 5366.88, 0.0, 0.0, 0.0, ("b", "s"), 1),
    Meson("Bs̄0", 5366.88, 0.0, 0.0, 0.0, ("b", "s"), 1),
    Meson("Bc+", 6274.9, 0.0, 0.0, 0.0, ("b", "c"), 1),
    Meson("Bc-", 6274.9, 0.0, 0.0, 0.0, ("b", "c"), 1),
    Meson("Upsilon(1S)", 9460.3, 1.0, 0.0, 0.0, ("b", "b"), 1),
    Meson("Upsilon(2S)", 10023.26, 1.0, 0.0, 0.0, ("b", "b"), 2),
    Meson("Upsilon(3S)", 10355.2, 1.0, 0.0, 0.0, ("b", "b"), 3),
    Meson("Upsilon(4S)", 10579.4, 1.0, 0.0, 0.0, ("b", "b"), 4),
    Meson("Upsilon(5S)", 10885.0, 1.0, 0.0, 0.0, ("b", "b"), 5),
    Meson("Upsilon(6S)", 11000.0, 1.0, 0.0, 0.0, ("b", "b"), 6)
]

# --- Constants ---
E_base = 115.045
E_spin = 635.545
E_iso = 16.0
E_I3 = 4.59
E_state1 = -104.578
E_state_G2 = 483.9
E_state_G3 = 210.319

transitionL1 = {
    "u→d": 3.932,
    "d→s": 356.405,
    "s→c": 1371.35,
    "c→b": 3402.1
}

generation = {
    "u": 1, "d": 1,
    "s": 2, "c": 2,
    "b": 3
}

def level(quark: str) -> int:
    return 1 if quark in ["u", "d"] else 2

def transitionPath(q_from: str, q_to: str, isL1: bool) -> float:
    path = ["u", "d", "s", "c", "b"]
    try:
        start = path.index(q_from)
        end = path.index(q_to)
    except ValueError:
        return 0.0
    if start == end:
        return 0.0

    step_path = path[start:end+1] if start < end else path[end:start+1][::-1]

    cost = 0.0
    for a, b in zip(step_path, step_path[1:]):
        key = f"{a}→{b}"
        base = transitionL1.get(key, 0.0)
        if isL1:
            cost += base
        else:
            if (a, b) == ("u", "d"):
                cost += base
            elif (a, b) == ("d", "s"):
                cost += base / 3.0
            elif (a, b) == ("s", "c"):
                cost += base * 2.0 / 3.0
            elif (a, b) == ("c", "b"):
                cost += base
    return cost

def deltaEBase(initial: Tuple[str, str], target: Tuple[str, str]) -> float:
    q1i, q2i = initial
    q1f, q2f = target
    step1 = transitionPath(q1i, q1f, isL1=(level(q2i) == 1))
    step2 = transitionPath(q2i, q2f, isL1=(level(q1f) == 1))
    return step1 + step2

def stateEnergy(state: int, spin: float, G: float) -> float:
    if spin > 0:
        if G == 2:
            return E_state1 if state == 1 else E_state_G2
        elif G == 3:
            if state == 1:
                return E_state1 * (G + 4/5) + E_state1 * (state * 2/5)
            elif state == 2:
                return E_state_G3 * (state - 1) + E_state1 * state * 2/5
            else:
                return E_state_G3 * (state - 1) - E_state1 * state / 9
    return 0.0

def predictedMass(meson: Meson) -> float:
    q1, q2 = meson.quarks
    G1 = generation.get(q1, 1)
    G2 = generation.get(q2, 1)
    G = max(G1, G2)

    deltaE = deltaEBase(("u", "u"), (q1, q2))
    spinTerm = E_spin * (meson.spin / G)
    isoTerm = E_iso * meson.isospin * G
    i3Term = E_I3 * abs(meson.isospin3)
    stateTerm = stateEnergy(meson.state, meson.spin, G)

    return E_base + deltaE + spinTerm + isoTerm + i3Term + stateTerm

# --- Run Predictions ---
totalError = 0.0

for meson in mesons:
    pred = predictedMass(meson)
    err = ((pred - meson.mass) / meson.mass) * 100.0
    print(f"{meson.name.ljust(16)} | Obs: {meson.mass:7.3f} | Pred: {pred:7.3f} | Δ%: {err:+6.3f}")
    totalError += abs(err)

print(f"Average Δ% error: {totalError / len(mesons):.5f}%")
