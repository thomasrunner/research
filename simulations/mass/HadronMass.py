# hadron_mass.py
# by Thomas Lock
# Based on: https://zenodo.org/records/15596490

from dataclasses import dataclass
from typing import List
import math

# --- Data Classes ---
@dataclass
class Hadron:
    name: str
    mass: float
    spin: float
    isospin: float
    isospin3: float
    quarks: List[str]
    state: int

@dataclass
class Isotope:
    name: str
    Z: int
    N: int
    massObs: float
    halfLife: float

# --- Constants ---
E_quark = 2.79000
E_base = 114.9318111
E_spin_coupling = 251.39
E_isospin_coupling = 17.52

E_state = -104.546
E_state_G2 = E_state * -100**(1/3)
E_state_G3 = E_state * -2.0

E_charm_coupling = 66.45
E_strange_coupling = 9.6

sigma = 0.262

transitionL1 = {
    "u→d": 1.296,
    "d→s": 358.405,
    "s→c": 1371.35,
    "c→b": 3403.1,
    "b→t": 167388.145
}

generation = {
    "u": 1, "d": 1,
    "s": 2, "c": 2,
    "b": 3
}

def level(quark: str) -> int:
    return 1 if quark in ["u", "d"] else 2

def transitionPath(q_from: str, q_to: str, isL1: bool) -> float:
    path = ["u", "d", "s", "c", "b", "t"]
    if q_from not in path or q_to not in path or q_from == q_to:
        return 0.0
    start = path.index(q_from)
    end = path.index(q_to)
    step_path = path[start:end+1] if start < end else list(reversed(path[end:start+1]))

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
            elif (a, b) == ("b", "t"):
                cost += base
    return cost

def deltaEbase(quarks=List[str]) -> float:
    if len(quarks) == 2:
        q1i, q2i = "u", "u"
        q1f, q2f = quarks[0], quarks[1]
        step1 = transitionPath(q1i, q1f, level(q2i) == 1)
        step2 = transitionPath(q2i, q2f, level(q1f) == 1)
        return step1 + step2
    else:
        total = 0.0
        for q in quarks:
            if q == "u":
                total += 0
            elif q == "d":
                total += transitionL1["u→d"]
            elif q == "s":
                total += transitionL1["u→d"] + transitionL1["d→s"] / 2
            elif q == "c":
                total += transitionL1["u→d"] + 6 * transitionL1["d→s"] / 13 + transitionL1["s→c"]
            elif q == "b":
                total += transitionL1["u→d"] + transitionL1["d→s"] + 38 * transitionL1["s→c"] / 56 + transitionL1["c→b"]
            elif q == "t":
                total += transitionL1["u→d"] + transitionL1["d→s"] + transitionL1["s→c"] + transitionL1["c→b"] + transitionL1["b→t"]
        return total

def stateEnergy(state:int, spin:float, G: float, Q: float) -> float:
    if spin > 0 and Q < 3:
        if G == 2:
            if state == 1:
                return E_state
            elif state == 2:
                return E_state_G2
        elif G == 3:
            if state == 1:
                return E_state * 21 / 5
            elif state == 2:
                return E_state_G2 / 4
            else:
                return E_state_G3 * (state - 1) - E_state * state / 9
    return 0.0

def nQuarks(quarks=List[str]) -> List[float]:
    Q = [0.0] * 7  # [u, d, s, c, b, t, extra]
    for q in quarks:
        if q == "u":
            Q[0] += 1.0
        elif q == "d":
            Q[1] += 1.0
        elif q == "s":
            Q[2] += 1.0
        elif q == "c":
            Q[3] += 1.0
        elif q == "b":
            Q[4] += 1.0
        elif q == "t":
            Q[5] += 1.0
    return Q

def gate(condition: bool) -> float:
    return 1.0 if condition else 0.0

def predictedMass(hadron: Hadron) -> float:
    q = hadron.quarks + ["u", "u"]  # pad to length 3
    q1, q2, q3 = q[0], q[1], q[2]
    G1 = generation.get(q1, 1.0)
    G2 = generation.get(q2, 1.0)
    G3 = generation.get(q3, 1.0)
    Q = float(len(hadron.quarks))
    G = max(G1, G2, G3)

    Qu = 1.0 if "u" in hadron.quarks else 0.0
    Qd = 1.0 if "d" in hadron.quarks else 0.0
    Qs = 1.0 if "s" in hadron.quarks else 0.0
    Qc = 1.0 if "c" in hadron.quarks else 0.0
    Qb = 1.0 if "b" in hadron.quarks else 0.0
    Tri = 1.0 if len(hadron.quarks) == 3 else 0.0

    NQ = nQuarks(hadron.quarks)
    Nu, Nd, Ns = NQ[0], NQ[1], NQ[2]

    if Q == 1:
        deltaE = deltaEbase(hadron.quarks)
        return E_quark + deltaE
    else:
        a = (7/2) * ((Q**2 - Q) / 2.0) - (5/2)
        E_base_a = E_base * a

        iso = E_isospin_coupling * hadron.isospin * G * (Q - 1.0) + \
              abs(Tri - 1.0) * (2.0 * E_isospin_coupling / 7.0) ** (
                gate(hadron.isospin == 1.0 and abs(hadron.isospin3) == 1.0) * G1)

        spin_coupling = E_spin_coupling * abs(Q - 4.0)**(4.0 / 3.0) * \
                        (hadron.spin - ((Q - 2)/2)) * \
                        (5.0 - G)**abs(2.0 - Q) / \
                        ((4.0)**abs(2 - Q) * G**abs(Q - 3.0))

        state_excitation = stateEnergy(hadron.state, hadron.spin, G, Q)

        flavor_coupling = (E_strange_coupling * G)**(Qs * Tri) * (-1)**Qc - \
                          (E_strange_coupling * 9)**(Qb * Tri * Qs * (Qu + Qd)) - \
                          Qc * Tri * (
                              E_charm_coupling +
                              (E_charm_coupling * 2.0)**Qs -
                              E_charm_coupling**gate(hadron.isospin == 0) +
                              (E_charm_coupling * 2.7)**(gate(hadron.isospin == 0) * Qd)
                          )

        deltaE = deltaEbase(hadron.quarks)
        return E_base_a + deltaE + iso + state_excitation + spin_coupling + flavor_coupling

# Sample hadron and isotope data
hadrons = [
    Hadron(name="Proton", mass=938.272089, spin=0.5, isospin=0.5, isospin3=0.5, quarks=["u", "u", "d"], state=1),
    Hadron(name="Neutron", mass=939.565421, spin=0.5, isospin=0.5, isospin3=-0.5, quarks=["u", "d", "d"], state=1),

    # Quarks
    Hadron(name="u", mass=2.79000, spin=0.5, isospin=0.5, isospin3=0.5, quarks=["u"], state=1),

    Hadron(name="d", mass=4.08600, spin=0.5, isospin=0.5,  isospin3=-0.5, quarks=["d"], state=1),
    Hadron(name="s", mass=183.28850, spin=0.5, isospin=0.0,  isospin3=0.0, quarks=["s"], state=1),
    Hadron(name="c", mass=1540.85369, spin=0.5, isospin=0.0,  isospin3=0.0, quarks=["c"], state=1),
    Hadron(name="b", mass=4696.14993, spin=0.5, isospin=0.0,  isospin3=0.5, quarks=["b"], state=1),
    Hadron(name="t", mass=172525.08600, spin=0.5, isospin=0.0,  isospin3=0.5, quarks=["t"], state=1),

    # Mesons
    Hadron(name="Pi0", mass=134.9766, spin=0.0, isospin=1.0,  isospin3=0.0, quarks=["u", "d"], state=1),
    Hadron(name="Pi+", mass=139.57039, spin=0.0, isospin=1.0,  isospin3=1.0, quarks=["u", "d"], state=1),
    Hadron(name="Pi-", mass=139.57039, spin=0.0, isospin=1.0,  isospin3=-1.0, quarks=["u", "d"], state=1),
    Hadron(name="K+", mass=493.677, spin=0.0, isospin=0.5,  isospin3=0.5, quarks=["u", "s"], state=1),
    Hadron(name="K-", mass=493.677, spin=0.0, isospin=0.5,  isospin3=-0.5, quarks=["u", "s"], state=1),
    Hadron(name="K0", mass=497.611, spin=0.0, isospin=0.5,  isospin3=-0.5, quarks=["d", "s"], state=1),
    Hadron(name="K̄0", mass=497.611, spin=0.0, isospin=0.5,  isospin3=0.5, quarks=["d", "s"], state=1),
    Hadron(name="K0_S", mass=497.611, spin=0.0, isospin=0.5,  isospin3=0.5, quarks=["d", "s"], state=1),
    Hadron(name="K0_L", mass=497.611, spin=0.0, isospin=0.5,  isospin3=0.5, quarks=["d", "s"], state=1),

    Hadron(name="D0", mass=1864.841, spin=0.0, isospin=0.5,  isospin3=-0.5, quarks=["c", "u"], state=1),
    Hadron(name="D̄0", mass=1864.841, spin=0.0, isospin=0.5,  isospin3=0.5, quarks=["c", "u"], state=1),
    Hadron(name="D+", mass=1869.65, spin=0.0, isospin=0.5,  isospin3=0.5, quarks=["c", "d"], state=1),
    Hadron(name="D-", mass=1869.65, spin=0.0, isospin=0.5,  isospin3=-0.5, quarks=["c", "d"], state=1),
    Hadron(name="Ds+", mass=1968.34, spin=0.0, isospin=0.0,  isospin3=0.0, quarks=["c", "s"], state=1),
    Hadron(name="Ds-", mass=1968.34, spin=0.0, isospin=0.0,  isospin3=0.0, quarks=["c", "s"], state=1),

    Hadron(name="B+", mass=5279.1, spin=0.0, isospin=0.5,  isospin3=0.5, quarks=["b", "u"], state=1),
    Hadron(name="B-", mass=5279.1, spin=0.0, isospin=0.5,  isospin3=-0.5, quarks=["b", "u"], state=1),
    Hadron(name="B0", mass=5279.1, spin=0.0, isospin=0.5,  isospin3=-0.5, quarks=["b", "d"], state=1),
    Hadron(name="B̄0", mass=5279.1, spin=0.0, isospin=0.5,  isospin3=0.5, quarks=["b", "d"], state=1),
    Hadron(name="Bs0", mass=5366.88, spin=0.0, isospin=0.0,  isospin3=0.0, quarks=["b", "s"], state=1),
    Hadron(name="Bs̄0", mass=5366.88, spin=0.0, isospin=0.0,  isospin3=0.0, quarks=["b", "s"], state=1),
    Hadron(name="Bc+", mass=6274.9, spin=0.0, isospin=0.0,  isospin3=0.0, quarks=["b", "c"], state=1),
    Hadron(name="Bc-", mass=6274.9, spin=0.0, isospin=0.0,  isospin3=0.0, quarks=["b", "c"], state=1),


    Hadron(name="J/Psi",     mass=3096.9,    spin=1.0, isospin=0.0,  isospin3=0.0, quarks=["c", "c"], state=1),
    Hadron(name="Psi(2S)",   mass=3686.1,    spin=1.0, isospin=0.0,  isospin3=0.0, quarks=["c", "c"], state=2),
    Hadron(name="Rho+",      mass=775.11,    spin=1.0, isospin=1.0,  isospin3=1.0, quarks=["u", "d"], state=1),
    Hadron(name="Rho-",      mass=775.11,    spin=1.0, isospin=1.0,  isospin3=-1.0, quarks=["u", "d"], state=1),
    Hadron(name="Upsilon(1S)", mass=9460.3,  spin=1.0, isospin=0.0,  isospin3=0.0, quarks=["b", "b"], state=1),
    Hadron(name="Upsilon(2S)", mass=10023.26, spin=1.0, isospin=0.0, isospin3=0.0, quarks=["b", "b"], state=2),
    Hadron(name="Upsilon(3S)", mass=10355.2, spin=1.0, isospin=0.0,  isospin3=0.0, quarks=["b", "b"], state=3),
    Hadron(name="Upsilon(4S)", mass=10579.4, spin=1.0, isospin=0.0,  isospin3=0.0, quarks=["b", "b"], state=4),
    Hadron(name="Upsilon(5S)", mass=10885.0, spin=1.0, isospin=0.0,  isospin3=0.0, quarks=["b", "b"], state=5),
    Hadron(name="Upsilon(6S)", mass=11000.0, spin=1.0, isospin=0.0,  isospin3=0.0, quarks=["b", "b"], state=6),

    # Baryons
    Hadron(name="Lambda0",      mass=1115.683,   spin=0.5, isospin=0.0, isospin3=0.0,  quarks=["u", "d", "s"], state=1),
    Hadron(name="Sigma+",       mass=1189.370,   spin=0.5, isospin=1.0, isospin3=1.0,  quarks=["u", "u", "s"], state=1),
    Hadron(name="Sigma0",       mass=1192.642,   spin=0.5, isospin=1.0, isospin3=0.0,  quarks=["u", "d", "s"], state=1),
    Hadron(name="Sigma−",       mass=1197.449,   spin=0.5, isospin=1.0, isospin3=-1.0, quarks=["d", "d", "s"], state=1),
    Hadron(name="Delta++",      mass=1232.000,   spin=1.5, isospin=1.5, isospin3=1.5,  quarks=["u", "u", "u"], state=1),
    Hadron(name="Delta+",       mass=1232.000,   spin=1.5, isospin=1.5, isospin3=0.5,  quarks=["u", "u", "d"], state=1),
    Hadron(name="Delta0",       mass=1232.000,   spin=1.5, isospin=1.5, isospin3=-0.5, quarks=["u", "d", "d"], state=1),
    Hadron(name="Delta−",       mass=1232.000,   spin=1.5, isospin=1.5, isospin3=-1.5, quarks=["d", "d", "d"], state=1),

    Hadron(name="Xi0",          mass=1314.860,   spin=0.5, isospin=0.5, isospin3=0.5,  quarks=["u", "s", "s"], state=1),
    Hadron(name="Xi−",          mass=1321.710,   spin=0.5, isospin=0.5, isospin3=-0.5, quarks=["d", "s", "s"], state=1),
    Hadron(name="Omega−",       mass=1672.450,   spin=1.5, isospin=0.0, isospin3=0.0,  quarks=["s", "s", "s"], state=1),

    Hadron(name="Lambda+c",     mass=2286.460,   spin=0.5, isospin=0.0, isospin3=0.0,  quarks=["u", "d", "c"], state=1),
    Hadron(name="Sigma++c",     mass=2453.970,   spin=0.5, isospin=1.0, isospin3=1.0,  quarks=["u", "u", "c"], state=1),
    Hadron(name="Sigma+c",      mass=2452.900,   spin=0.5, isospin=1.0, isospin3=0.0,  quarks=["u", "d", "c"], state=1),
    Hadron(name="Sigma0c",      mass=2453.750,   spin=0.5, isospin=1.0, isospin3=-1.0, quarks=["d", "d", "c"], state=1),
    Hadron(name="Xi0c",         mass=2470.850,   spin=0.5, isospin=0.5, isospin3=-0.5, quarks=["d", "s", "c"], state=1),
    Hadron(name="Xi+c",         mass=2468.000,   spin=0.5, isospin=0.5, isospin3=0.5,  quarks=["u", "s", "c"], state=1),
    Hadron(name="Omega0c",      mass=2695.200,   spin=0.5, isospin=0.0, isospin3=0.0,  quarks=["s", "s", "c"], state=1),

    Hadron(name="Lambda0b",     mass=5619.600,   spin=0.5, isospin=0.0, isospin3=0.0,  quarks=["u", "d", "b"], state=1),
    Hadron(name="Xi0b",         mass=5787.800,   spin=0.5, isospin=0.5, isospin3=0.5,  quarks=["u", "s", "b"], state=1),
    Hadron(name="Xi−b",         mass=5794.400,   spin=0.5, isospin=0.5, isospin3=-0.5, quarks=["d", "s", "b"], state=1),
    Hadron(name="Omega−b",      mass=6045.100,   spin=0.5, isospin=0.0, isospin3=0.0,  quarks=["s", "s", "b"], state=1),

    #Top Predictions Meson
    Hadron(name="Dt0", mass=172646.98781, spin=0.0, isospin=0.5,  isospin3=-0.5, quarks=["t", "u"], state=1),
    Hadron(name="D̄t0", mass=172646.98781, spin=0.0, isospin=0.5,  isospin3=0.5, quarks=["t", "u"], state=1),
    Hadron(name="Dt+", mass=172648.28381, spin=0.0, isospin=0.5,  isospin3=0.5, quarks=["t", "d"], state=1),
    Hadron(name="Dt-", mass=172648.28381, spin=0.0, isospin=0.5,  isospin3=-0.5, quarks=["t", "d"], state=1),

    #Top Predictions Baryon
    Hadron(name="Sigma++t",     mass=173476.79049,   spin=0.5, isospin=1.0, isospin3=1.0,  quarks=["u", "u", "t"], state=1),
    Hadron(name="Sigma+t",      mass=173478.08649,   spin=0.5, isospin=1.0, isospin3=0.0,  quarks=["u", "d", "t"], state=1),
    Hadron(name="Sigma0t",      mass=173479.38249,   spin=0.5, isospin=1.0, isospin3=-1.0, quarks=["d", "d", "t"], state=1),
]

isotopes = [
    Isotope(name="Protium", Z=1, N=0, massObs=938.272, halfLife=float("inf")),
    Isotope(name="Deuterium", Z=1, N=1, massObs=1875.612, halfLife=float("inf")),
    Isotope(name="Tritium",       Z=1,  N=2,   massObs=2808.921,    halfLife=3.888e8),
    Isotope(name="Helium-3",      Z=2,  N=1,   massObs=2808.391,    halfLife=0),
    Isotope(name="Helium-4",      Z=2,  N=2,   massObs=3727.379,    halfLife=0),
    Isotope(name="Lithium-6",     Z=3,  N=3,   massObs=5601.519,    halfLife=0),
    Isotope(name="Lithium-7",     Z=3,  N=4,   massObs=6533.832,    halfLife=0),
    Isotope(name="Beryllium-9",   Z=4,  N=5,   massObs=8394.795,    halfLife=0),
    Isotope(name="Boron-10",      Z=5,  N=5,   massObs=9310.194,    halfLife=0),
    Isotope(name="Boron-11",      Z=5,  N=6,   massObs=10254.545,   halfLife=0),
    Isotope(name="Carbon-12",     Z=6,  N=6,   massObs=11177.928,   halfLife=0),
    Isotope(name="Carbon-14",     Z=6,  N=8,   massObs=13044.570,   halfLife=1.8e11),
    Isotope(name="Nitrogen-14",   Z=7,  N=7,   massObs=13043.780,   halfLife=0),
    Isotope(name="Oxygen-16",     Z=8,  N=8,   massObs=14900.383,   halfLife=0),
    Isotope(name="Oxygen-18",     Z=8,  N=10,   massObs=16791.228,   halfLife=0),
    Isotope(name="Iron-56",       Z=26, N=30,  massObs=52048.736,   halfLife=0),
    Isotope(name="Nickel-62",     Z=28, N=34,  massObs=57674.0,     halfLife=0),
    Isotope(name="Lead-206",      Z=82, N=124,  massObs=191902.4,    halfLife=0),
    Isotope(name="Lead-208",      Z=82, N=126,  massObs=193681.3,    halfLife=0),
    Isotope(name="Uranium-235",   Z=92, N=143,  massObs=218876.7,    halfLife=2.22e16),
    Isotope(name="Uranium-238",   Z=92, N=146,  massObs=221695.8,    halfLife=1.41e17),
    Isotope(name="Plutonium-239", Z=94, N=145,  massObs=223888.0,    halfLife=7.6e11)
]

# Constants
M_p = 938.27049
M_n = 939.56649
m_e = 0.5109989461

# Run predictions
totalError = 0.0

for hadron in hadrons:
    pred = predictedMass(hadron)
    err = ((pred - hadron.mass) / hadron.mass) * 100.0
    print(f"{hadron.name:<16} | Obs: {hadron.mass:9.5f} | Pred: {pred:9.5f} | Δ%: {err:+6.5f}")

    totalError += abs(err)

print(f"Average Δ% error: {totalError / len(hadrons):.5f}%")

# Isotope mass predictions
isotopeError = 0.0
for iso in isotopes:
    Z = float(iso.Z)
    N = float(iso.N)
    A = Z + N

    baseMass = Z * M_p + N * M_n + Z * m_e
    kappa = 0.9995 - 0.00525 * (A / 239.0)**0.0154 * N / Z
    predicted = baseMass * kappa
    error = ((predicted - iso.massObs) / iso.massObs) * 100.0

    print(f"{iso.name:<16} | Obs: {iso.massObs:9.5f} | Pred: {predicted:9.5f} | Δ%%: {error:+9.5f}")
    totalError += abs(error)
    isotopeError += abs(error)

print(f"Atomic Isotope Δ%% error: {isotopeError / len(isotopes):.5f}%")
print(f"Total Hadron + Atomic Δ%% error: {totalError / (len(hadrons) + len(isotopes)):.5f}%")

