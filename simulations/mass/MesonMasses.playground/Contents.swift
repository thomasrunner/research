//
//  MesonMasses.swift
//
//  Created by Thomas Lock on Jun 2 2025
//


import Foundation

let mesons: [Meson] = [
    Meson(name: "Pi0",       mass: 134.9766,  spin: 0.0, isospin: 1.0,  isospin3: 0.0,   quarks: ("u", "d"), state: 1),
    Meson(name: "Pi+",       mass: 139.57039, spin: 0.0, isospin: 1.0,  isospin3: 1.0,   quarks: ("u", "d"), state: 1),
    Meson(name: "Pi-",       mass: 139.57039, spin: 0.0, isospin: 1.0,  isospin3: -1.0,  quarks: ("u", "d"), state: 1),
    Meson(name: "K+",        mass: 493.677,   spin: 0.0, isospin: 0.5,  isospin3: 0.5,   quarks: ("u", "s"), state: 1),
    Meson(name: "K-",        mass: 493.677,   spin: 0.0, isospin: 0.5,  isospin3: -0.5,  quarks: ("u", "s"), state: 1),
    Meson(name: "K0",        mass: 497.611,   spin: 0.0, isospin: 0.5,  isospin3: -0.5,  quarks: ("d", "s"), state: 1),
    Meson(name: "K̄0",       mass: 497.611,   spin: 0.0, isospin: 0.5,  isospin3: 0.5,   quarks: ("d", "s"), state: 1),
    Meson(name: "K0_S",      mass: 497.611,   spin: 0.0, isospin: 0.5,  isospin3: 0.5,   quarks: ("d", "s"), state: 1),
    Meson(name: "K0_L",      mass: 497.611,   spin: 0.0, isospin: 0.5,  isospin3: 0.5,   quarks: ("d", "s"), state: 1),
    Meson(name: "Rho+",      mass: 775.11,    spin: 1.0, isospin: 1.0,  isospin3: 1.0,   quarks: ("u", "d"), state: 1),
    Meson(name: "Rho-",      mass: 775.11,    spin: 1.0, isospin: 1.0,  isospin3: -1.0,  quarks: ("u", "d"), state: 1),
    Meson(name: "D0",        mass: 1864.841,  spin: 0.0, isospin: 0.5,  isospin3: -0.5,  quarks: ("c", "u"), state: 1),
    Meson(name: "D̄0",       mass: 1864.841,  spin: 0.0, isospin: 0.5,  isospin3: 0.5,   quarks: ("c", "u"), state: 1),
    Meson(name: "D+",        mass: 1869.65,   spin: 0.0, isospin: 0.5,  isospin3: 0.5,   quarks: ("c", "d"), state: 1),
    Meson(name: "D-",        mass: 1869.65,   spin: 0.0, isospin: 0.5,  isospin3: -0.5,  quarks: ("c", "d"), state: 1),
    Meson(name: "Ds+",       mass: 1968.34,   spin: 0.0, isospin: 0.0,  isospin3: 0.0,   quarks: ("c", "s"), state: 1),
    Meson(name: "Ds-",       mass: 1968.34,   spin: 0.0, isospin: 0.0,  isospin3: 0.0,   quarks: ("c", "s"), state: 1),
    Meson(name: "J/Psi",     mass: 3096.9,    spin: 1.0, isospin: 0.0,  isospin3: 0.0,   quarks: ("c", "c"), state: 1),
    Meson(name: "Psi(2S)",   mass: 3686.1,    spin: 1.0, isospin: 0.0,  isospin3: 0.0,   quarks: ("c", "c"), state: 2),
    Meson(name: "B+",        mass: 5279.1,    spin: 0.0, isospin: 0.5,  isospin3: 0.5,   quarks: ("b", "u"), state: 1),
    Meson(name: "B-",        mass: 5279.1,    spin: 0.0, isospin: 0.5,  isospin3: -0.5,  quarks: ("b", "u"), state: 1),
    Meson(name: "B0",        mass: 5279.1,    spin: 0.0, isospin: 0.5,  isospin3: -0.5,  quarks: ("b", "d"), state: 1),
    Meson(name: "B̄0",       mass: 5279.1,    spin: 0.0, isospin: 0.5,  isospin3: 0.5,   quarks: ("b", "d"), state: 1),
    Meson(name: "Bs0",       mass: 5366.88,   spin: 0.0, isospin: 0.0,  isospin3: 0.0,   quarks: ("b", "s"), state: 1),
    Meson(name: "Bs̄0",      mass: 5366.88,   spin: 0.0, isospin: 0.0,  isospin3: 0.0,   quarks: ("b", "s"), state: 1),
    Meson(name: "Bc+",       mass: 6274.9,    spin: 0.0, isospin: 0.0,  isospin3: 0.0,   quarks: ("b", "c"), state: 1),
    Meson(name: "Bc-",       mass: 6274.9,    spin: 0.0, isospin: 0.0,  isospin3: 0.0,   quarks: ("b", "c"), state: 1),
    Meson(name: "Upsilon(1S)", mass: 9460.3,  spin: 1.0, isospin: 0.0,  isospin3: 0.0,   quarks: ("b", "b"), state: 1),
    Meson(name: "Upsilon(2S)", mass: 10023.26, spin: 1.0, isospin: 0.0, isospin3: 0.0,   quarks: ("b", "b"), state: 2),
    Meson(name: "Upsilon(3S)", mass: 10355.2, spin: 1.0, isospin: 0.0,  isospin3: 0.0,   quarks: ("b", "b"), state: 3),
    Meson(name: "Upsilon(4S)", mass: 10579.4, spin: 1.0, isospin: 0.0,  isospin3: 0.0,   quarks: ("b", "b"), state: 4),
    Meson(name: "Upsilon(5S)", mass: 10885.0, spin: 1.0, isospin: 0.0,  isospin3: 0.0,   quarks: ("b", "b"), state: 5),
    Meson(name: "Upsilon(6S)", mass: 11000.0, spin: 1.0, isospin: 0.0,  isospin3: 0.0,   quarks: ("b", "b"), state: 6)
]

import Foundation

struct Meson {
    let name: String
    let mass: Double
    let spin: Double
    let isospin: Double
    let isospin3: Double
    let quarks: (String, String)
    let state: Int
}

// --- Constants ---
let E_base = 115.045
let E_spin = 635.545
let E_iso = 16.0
let E_I3 = 4.59
let E_state1 = -104.578
let E_state_G2 = 483.9
let E_state_G3 = 210.319


let transitionL1: [String: Double] = [
    "u→d": 3.932,
    "d→s": 356.405,
    "s→c": 1371.35,
    "c→b": 3402.1
]

let generation: [String: Int] = [
    "u": 1, "d": 1,
    "s": 2, "c": 2,
    "b": 3
]

// Return 1 if quark is light (u, d), else 2
func level(of quark: String) -> Int {
    return ["u", "d"].contains(quark) ? 1 : 2
}

// Returns transition cost from → to using L1 or L2 rules
func transitionPath(from: String, to: String, isL1: Bool) -> Double {
    let path = ["u", "d", "s", "c", "b"]
    guard let start = path.firstIndex(of: from),
          let end = path.firstIndex(of: to),
          start != end else { return 0.0 }

    let stepPath: [String] = start < end
        ? Array(path[start...end])
        : Array(path[end...start].reversed())

    var cost = 0.0
    for (a, b) in zip(stepPath, stepPath.dropFirst()) {
        let key = "\(a)→\(b)"
        let base = transitionL1[key] ?? 0.0
        if isL1 {
            cost += base
        } else {
            switch (a, b) {
            case ("u", "d"): cost += base
            case ("d", "s"): cost += base / 3.0
            case ("s", "c"): cost += base * 2.0 / 3.0
            case ("c", "b"): cost += base
            default: break
            }
        }
    }

    return cost
}

// Correct two-stage deltaEBase calculation
func deltaEBase(initial: (String, String), target: (String, String)) -> Double {
    let (q1i, q2i) = initial
    let (q1f, q2f) = target

    let step1 = transitionPath(from: q1i, to: q1f, isL1: level(of: q2i) == 1)
    let step2 = transitionPath(from: q2i, to: q2f, isL1: level(of: q1f) == 1)
    return step1 + step2
}

// E_state(S) = sum of 452 / (i - 1) for i = 2 to S
func stateEnergy(state: Int, spin: Double, G: Double) -> Double {
    if spin > 0 {
        if G == 2 {
            if state == 1 {
                return E_state1
            } else {
                return E_state_G2
            }
        } else if G == 3 {
            if state == 1 {
                return E_state1 * (G + 4/5) + E_state1 * (Double(state) * 2/5)
            } else if state == 2 {
                return E_state_G3 * Double(state - 1) + E_state1 * Double(state) * 2/5
            } else {
                return E_state_G3 * Double(state - 1) - E_state1 * Double(state) / 9
            }
        }
    }
    return 0
}

// Full mass prediction function
func predictedMass(meson: Meson) -> Double {
    let (q1, q2) = meson.quarks
    let G1 = generation[q1] ?? 1
    let G2 = generation[q2] ?? 1
    let G = max(G1, G2)

    let ΔE = deltaEBase(initial: ("u", "u"), target: (q1, q2))
    let spinTerm = E_spin * (meson.spin / Double(G))
    let isoTerm = E_iso * meson.isospin * Double(G)
    let i3Term = E_I3 * abs(meson.isospin3)
    let stateTerm = stateEnergy(state: meson.state, spin: meson.spin, G: Double(G))

    return E_base + ΔE + spinTerm + isoTerm + i3Term + stateTerm
}

// --- Run Predictions ---
var totalError = 0.0

for meson in mesons {
    let pred = predictedMass(meson: meson)
    let err = ((pred - meson.mass) / meson.mass) * 100.0
    print("\(meson.name.padding(toLength: 14, withPad: " ", startingAt: 0)) | Obs: \(String(format: "%7.3f", meson.mass)) | Pred: \(String(format: "%7.3f", pred)) | Δ%: \(String(format: "%+6.3f", err))")
    totalError += abs(err)
}

print(String(format: "Average Δ%% error: %.5f%%", totalError / Double(mesons.count)))
