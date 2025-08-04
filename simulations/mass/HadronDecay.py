# hadron_decay.py
# Exact Python mirror of HadronDecay.swift by Thomas Lock
# Based on: https://zenodo.org/records/15596490

from dataclasses import dataclass
from typing import List
from math import pow

# Constants
hbar = 6.582119569e-22  # Planck constant / 2π

deltaE = {
    "u": 2.49296e-25,
    "d": 2.49299e-25,
    "s": 6.330e-14,
    "c": 1.291e-9,
    "b": 4.478e-10,
    "t": 1.316e3
}

def gate(condition: bool) -> float:
    return 1.0 if condition else 0.0

@dataclass
class Hadron:
    name: str
    quarks: List[str]
    spin: float
    mass: float
    isospin: float
    isospin3: float
    state: int
    decayTime: float
    cp: float
    superposition: float
    charge: float

def deltaEbase(quarks: List[str]) -> float:
    total = 0.0
    for q in quarks:
        if q == "u": total += deltaE["u"]
        elif q == "d": total += deltaE["d"]
        elif q == "s": total += deltaE["s"] if len(quarks) == 2 else deltaE["s"] * 1e2
        elif q == "c": total += deltaE["c"] if len(quarks) == 2 else deltaE["c"] * 16 / 3
        elif q == "b": total += deltaE["b"]
        elif q == "t": total += deltaE["t"]
    return total

def nQuarks(quarks: List[str]) -> List[float]:
    Q = [0.0] * 6  # u, d, s, c, b, t
    for q in quarks:
        if q == "u": Q[0] += 1.0
        elif q == "d": Q[1] += 1.0
        elif q == "s": Q[2] += 1.0
        elif q == "c": Q[3] += 1.0
        elif q == "b": Q[4] += 1.0
        elif q == "t": Q[5] += 1.0
    return Q

def predictedDecay(hadron):
    NQ = nQuarks(hadron.quarks)
    Nu, Nd, Ns, Nc, Nb = NQ[0:5]

    Gu = gate("u" in hadron.quarks)
    Gd = gate("d" in hadron.quarks)
    Gs = gate("s" in hadron.quarks)
    Gc = gate("c" in hadron.quarks)
    Gb = gate("b" in hadron.quarks)

    G = 3.0 if Gb else (2.0 if (Gs or Gc) else 1.0)
    G1 = 1.0 if G == 1.0 else 0.0
    G2 = 1.0 if G == 2.0 else 0.0
    G3 = 1.0 if G == 3.0 else 0.0

    Tri = 1.0 if len(hadron.quarks) == 3 else 0.0
    ΔE = deltaEbase(hadron.quarks)
    κ = pow(9.83e-11, gate(1 != 1))  # always 1

    # Spin Decay
    s = hadron.spin - abs((2.0 - len(hadron.quarks)) / 2.0)
    s0 = gate(hadron.spin == 0.0)

    Spin = [6.55e-27, 4.2575e-9, 6.55e-8, 1.97203e-11]
    spin_decay = pow(Spin[0] * pow(0.523, abs(Tri - 1)), s * G1)
    spin_decay *= pow(Spin[1], s * G2 * abs(Tri - 1))
    spin_decay *= pow(Spin[2], s * G3 * abs(Tri - 1))
    spin_decay *= pow(Spin[3], s0 * G1 * abs(Tri - 1))

    # State Decay
    States = [6.6167, 2.9661, 2.53005e-1, 3.20247, 3.4485, 9.99986e-1, 1.450986]
    state_decay = pow(States[0], Gc * gate(hadron.state >= 2))
    state_decay *= pow(States[1], Gc * gate(hadron.state == 3))
    state_decay *= pow(States[2], Gb * gate(hadron.state >= 2))
    state_decay *= pow(States[3], Gb * gate(hadron.state == 3))
    state_decay *= pow(States[4], Gb * gate(hadron.state == 4))
    state_decay *= pow(States[5], Gb * gate(hadron.state == 5))
    state_decay *= pow(States[6], Gb * gate(hadron.state == 6))

    # Superspin Decay
    Superspin = [4.790e-11, 3.093e-13, 3.336e1, 1.74512e-3, 6.74e1]
    superspin = pow(Superspin[0], hadron.superposition * abs(hadron.spin - 1) * abs(hadron.cp - 1))
    superspin *= pow(Superspin[1], hadron.superposition * abs(hadron.spin - 1) * hadron.cp)
    superspin *= pow(Superspin[2], hadron.superposition * hadron.spin * abs(hadron.cp - 1))
    superspin *= pow(Superspin[3], abs(hadron.superposition - 1) * abs(hadron.spin - 1) * hadron.cp)
    superspin *= pow(Superspin[4], hadron.superposition * abs(hadron.spin - 1) * abs(hadron.cp - 1) * Nd)

    flavor_decay = 1.0

    if Tri == 1:
        # Baryons
        if hadron.isospin == 1.5:
            ΔEg1 = 1.04
            flavor_decay *= pow(ΔEg1, 1)

        elif hadron.isospin == 1.0:
            ΔEi = 2.066e-9
            ΔEu = 0.38525
            ΔEd = 0.725
            ΔEc = 1.05e-8

            if Ns >= 1:
                flavor_decay *= pow(ΔEu * ΔEd * ΔEi, gate(Nd * Nu == 1))
                flavor_decay *= pow(2.0 * ΔEd, gate(Nd == 2))
                flavor_decay *= pow(2.0 * ΔEu, gate(Nu == 2))
            elif Nc == 1:
                flavor_decay *= pow(ΔEc, 1)

        elif hadron.isospin == 0.5:
            ΔEi = 2.435
            ΔEu = 3.99
            ΔEd = 1.0
            ΔEs = 1.155
            ΔEb = 1.07

            if Ns == 2:
                flavor_decay *= pow(ΔEu * ΔEs / 2 * ΔEi, gate(Nu == 1.0))
                flavor_decay *= pow(ΔEd * ΔEs / 2 * ΔEi * 9/4, gate(Nd == 1.0))
            elif Ns == 1:
                if Nc == 1:
                    flavor_decay *= pow(ΔEu * ΔEs, gate(Nu == 1.0))
                    flavor_decay *= pow(ΔEd * ΔEs, gate(Nd == 1.0))
                elif Nb == 1:
                    flavor_decay *= pow(ΔEb, 1)

        elif hadron.isospin == 0.0:
            ΔEi = 0.246712
            ΔEj = 7.095
            ΔEg1 = 5.12744
            ΔEs = 1.0
            ΔEc = 0.203806
            ΔEb = 0.256152

            if Ns == 1.0:
                flavor_decay *= pow(2.0 * ΔEg1 * ΔEi, gate(Nu * Nd == 1.0))
            elif Ns > 1:
                flavor_decay *= pow(ΔEs / 3.0 * ΔEj, gate(Ns / 3.0 == 1.0))
                flavor_decay *= pow((ΔEb * ΔEs) / 2.0 * ΔEj, gate((Nb * Ns) / 2.0 == 1.0))
                flavor_decay *= pow((ΔEs * ΔEc) / 2.0 * ΔEj, gate((Ns * Nc) / 2.0 == 1.0))
            elif Nc == 1:
                flavor_decay *= pow(2.0 * ΔEg1 * ΔEc, gate(Nu * Nd * Nc == 1.0))

    else:
        # Mesons
        ΔEu = 0.8115
        ΔEd = 2.0368
        ΔEs = 1.451
        ΔEc = 0.986
        ΔEb = 1.3682

        if hadron.isospin == 0.5:
            flavor_decay *= pow(ΔEu * ΔEs, gate(Gu * Gs == 1))
            flavor_decay *= pow(ΔEd * ΔEc, gate(Gd * Gc == 1))
            flavor_decay *= pow(ΔEu * ΔEb, gate(Gu * Gb == 1))
            flavor_decay *= pow(ΔEd * ΔEs * 5/3, gate(Gd * Gs == 1))
            flavor_decay *= pow(ΔEu * ΔEc, gate(Gu * Gc == 1))
        elif hadron.isospin == 0.0:
            flavor_decay *= pow(ΔEc * ΔEb, gate(Gc * Gb == 1))
            flavor_decay *= pow(ΔEs * ΔEb * 7/15, gate(Gs * Gb == 1))

    return (κ * (hbar / ΔE)) * spin_decay * state_decay * superspin * flavor_decay



# Sample usage:
if __name__ == "__main__":
    hadrons = [
        Hadron("Neutron", ["u", "d", "d"], 0.5, 939.565421, 0.5, -0.5, 1, 880.0, 0, 0, 0.0),
        Hadron("K0", ["d", "s"], 0.0, 497.611, 0.5, -0.5, 1, 5.116e-8, 0, 0, 0.0),
        Hadron("K̄0",       ["d", "s"], 0.0, 497.611,   0.5,  0.5, 1, 5.116e-8,   0, 0,  0.0),
        Hadron("K0_L",      ["d", "s"], 0.0, 497.611,   0.5,  0.5, 1, 5.116e-8,   0, 0,  0.0),
        Hadron("Pi+",       ["u", "d"], 0.0, 139.57039, 1.0,  1.0, 1, 2.6033e-8,  0, 0,  1.0),
        Hadron("Pi−",       ["u", "d"], 0.0, 139.57039, 1.0, -1.0, 1, 2.6033e-8,  0, 0, -1.0),
        Hadron("K+",        ["u", "s"], 0.0, 493.677,   0.5,  0.5, 1, 1.2380e-8,  0, 0,  1.0),
        Hadron("K−",        ["u", "s"], 0.0, 493.677,   0.5, -0.5, 1, 1.2380e-8,  0, 0, -1.0),
        Hadron("Xi0",       ["u", "s", "s"], 0.5, 1314.860, 0.5, 0.5, 1, 2.9e-10, 0, 0, 0.0),
        Hadron("Lambda0",   ["u", "d", "s"], 0.5, 1115.683, 0.0, 0.0, 1, 2.631e-10, 0, 0, 0.0),
        Hadron("Xi−",       ["d", "s", "s"], 0.5, 1321.710, 0.5, -0.5, 1, 1.64e-10, 0, 0, -1.0),
        Hadron("Sigma−",    ["d", "d", "s"], 0.5, 1197.449, 1.0, -1.0, 1, 1.5e-10, 0, 0, -1.0),
        Hadron("K0_S",      ["d", "s"], 0.0, 497.611,   0.5,  0.5, 1, 8.9598e-11, 1, 0,  0.0),
        Hadron("Omega−",    ["s", "s", "s"], 1.5, 1672.450, 0.0, 0.0, 1, 8.2e-11, 0, 0, -1.0),
        Hadron("Sigma+",    ["u", "u", "s"], 0.5, 1189.370, 1.0, 1.0, 1, 8.0e-11, 0, 0,  1.0),
        Hadron("B+",        ["b", "u"], 0.0, 5279.1,    0.5,  0.5, 1, 1.638e-12,  0, 0,  1.0),
        Hadron("B−",        ["b", "u"], 0.0, 5279.1,    0.5, -0.5, 1, 1.638e-12,  0, 0, -1.0),
        Hadron("Xi−b",      ["d", "s", "b"], 0.5, 5794.400, 0.5, -0.5, 1, 1.6e-12,  0, 0, -1.0),
        Hadron("Xi0b",      ["u", "s", "b"], 0.5, 5787.800, 0.5,  0.5, 1, 1.5e-12,  0, 0,  0.0),
        Hadron("B0",        ["b", "d"], 0.0, 5279.1,    0.5, -0.5, 1, 1.5053e-12, 0, 0,  0.0),
        Hadron("B̄0",       ["b", "d"], 0.0, 5279.1,    0.5,  0.5, 1, 1.5053e-12, 0, 0,  0.0),
        Hadron("Lambda0b",  ["u", "d", "b"], 0.5, 5619.600, 0.0, 0.0, 1, 1.409e-12, 0, 0, 0.0),
        Hadron("Bs0",       ["b", "s"], 0.0, 5366.88,   0.0,  0.0, 1, 1.360e-12,  0, 0,  0.0),
        Hadron("Bs̄0",      ["b", "s"], 0.0, 5366.88,   0.0,  0.0, 1, 1.360e-12,  0, 0,  0.0),
        Hadron("Omega−b",   ["s", "s", "b"], 0.5, 6045.100, 0.0, 0.0, 1, 1.3e-12,  0, 0, -1.0),
        Hadron("D+",        ["c", "d"], 0.0, 1869.65,   0.5,  0.5, 1, 1.040e-12,  0, 0,  1.0),
        Hadron("D−",        ["c", "d"], 0.0, 1869.65,   0.5, -0.5, 1, 1.040e-12,  0, 0, -1.0),
        Hadron("Bc+",       ["b", "c"], 0.0, 6274.9,    0.0,  0.0, 1, 5.094e-13,  0, 0,  1.0),
        Hadron("Bc−",       ["b", "c"], 0.0, 6274.9,    0.0,  0.0, 1, 5.094e-13,  0, 0, -1.0),
        Hadron("Ds+",       ["c", "s"], 0.0, 1968.34,   0.0,  0.0, 1, 5.074e-13,  0, 0,  1.0),
        Hadron("Ds−",       ["c", "s"], 0.0, 1968.34,   0.0,  0.0, 1, 5.074e-13,  0, 0, -1.0),
        Hadron("Xi+c",      ["u", "s", "c"], 0.5, 2468.000, 0.5, 0.5, 1, 4.4e-13,  0, 0,  1.0),
        Hadron("D0",        ["c", "u"], 0.0, 1864.841,  0.5, -0.5, 1, 4.101e-13,  0, 0,  0.0),
        Hadron("D̄0",       ["c", "u"], 0.0, 1864.841,  0.5,  0.5, 1, 4.101e-13,  0, 0,  0.0),
        Hadron("Lambda+c",  ["u", "d", "c"], 0.5, 2286.460, 0.0, 0.0, 1, 2.0e-13,  0, 0,  1.0),
        Hadron("Xi0c",      ["d", "s", "c"], 0.5, 2470.850, 0.5, -0.5, 1, 1.1e-13,  0, 0,  0.0),
        Hadron("Omega0c",   ["s", "s", "c"], 0.5, 2695.200, 0.0, 0.0, 1, 6.9e-14,  0, 0,  0.0),
        Hadron("Pi0",       ["u", "d"], 0.0, 134.9766,  1.0,  0.0, 1, 8.52e-17,    0, 1,  0.0),
        Hadron("Eta",       ["s", "u"], 0.0, 547.862,   0.0,  0.0, 1, 5.0e-19,     0, 1,  0.0),
        Hadron("Sigma0",    ["u", "d", "s"], 0.5, 1192.642, 1.0, 0, 1, 6.0e-20,   0, 0,  0.0),
        Hadron("Upsilon(3S)", ["b", "b"], 1.0, 10355.2, 0.0,  0.0, 4, 4.2e-20,     0, 0,  0.0),
        Hadron("Upsilon(2S)", ["b", "b"], 1.0, 10023.26, 0.0, 0.0, 3, 3.9e-20,    0, 0,  0.0),
        Hadron("Psi(2S)",   ["c", "c"], 1.0, 3686.1,    0.0,  0.0, 3, 2.13e-20,    0, 0,  0.0),
        Hadron("Upsilon(5S)", ["b", "b"], 1.0, 10885.0, 0.0,  0.0, 6, 1.76e-20,    0, 0,  0.0),
        Hadron("Upsilon(1S)", ["b", "b"], 1.0, 9460.3,  0.0,  0.0, 2, 1.21e-20,    0, 0,  0.0),
        Hadron("Upsilon(4S)", ["b", "b"], 1.0, 10579.4, 0.0,  0.0, 5, 1.21e-20,    0, 0,  0.0),
        Hadron("J/Psi",     ["c", "c"], 1.0, 3096.9,    0.0,  0.0, 2, 7.2e-21,     0, 0,  0.0),
        Hadron("EtaPrime",  ["u", "s"], 0.0, 957.78,    0.0,  0.0, 1, 3.2e-21,     1, 1,  0.0),
        Hadron("Sigma++c",  ["u", "u", "c"], 0.5, 2453.970, 1.0, 1, 1, 1.0e-21,   0, 0,  2.0),
        Hadron("Sigma+c",   ["u", "d", "c"], 0.5, 2452.900, 1.0, 0, 1, 1.0e-21,   0, 0,  1.0),
        Hadron("Sigma0c",   ["d", "d", "c"], 0.5, 2453.750, 1.0,-1, 1, 1.0e-21,   0, 0,  0.0),
        Hadron("Phi",       ["u", "d"], 1.0, 1019.46,   0.0,  0.0, 1, 1.5e-22,     0, 1,  0.0),
        Hadron("Delta++",   ["u", "u", "u"], 1.5, 1232.000, 1.5,  1.5, 1, 6.0e-24, 0, 0,  2.0),
        Hadron("Delta+",    ["u", "u", "d"], 1.5, 1232.000, 1.5,  0.5, 1, 6.0e-24, 0, 0,  1.0),
        Hadron("Delta0",    ["u", "d", "d"], 1.5, 1232.000, 1.5, -0.5, 1, 6.0e-24, 0, 0,  0.0),
        Hadron("Delta−",    ["d", "d", "d"], 1.5, 1232.000, 1.5, -1.5, 1, 6.0e-24, 0, 0, -1.0),
        Hadron("Rho+",      ["u", "d"], 1.0, 775.11,    1.0,  1.0, 1, 4.5e-24,     0, 0,  1.0),
        Hadron("Rho−",      ["u", "d"], 1.0, 775.11,    1.0, -1.0, 1, 4.5e-24,     0, 0, -1.0),
        Hadron("Rho0",      ["u", "d"], 1.0, 775.11,    1.0,  0.0, 1, 4.4e-24,     0, 0,  0.0)
    ]

    totalError = 0.0
    for h in hadrons:
        pred = predictedDecay(h)
        error = ((pred - h.decayTime) / h.decayTime) * 100.0
        totalError += abs(error)
        print(f"{h.name:12s} | Obs: {h.decayTime:.5e}s | Pred: {pred:.5e}s | Error: {error:+9.5f}%")

print(f"Total Hadron Decay Δ%% error: {totalError / len(hadrons):.5f}%")
