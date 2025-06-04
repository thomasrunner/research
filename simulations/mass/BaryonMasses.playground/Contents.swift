//
//  BaryonMasses.swift
//
//  Created by Thomas Lock on Jun 2 2025
//  Based on Paper > https://zenodo.org/records/15596490


import UIKit
import Foundation

struct Baryon {
    let name: String
    let quarkContent: [String]
    let spin: Double
    let mass: Double
    let isospin: Double
    let isospin3: Double
    let decayTime: String
}


let baryons: [Baryon] = [
    Baryon(name: "Proton",       quarkContent: ["u", "u", "d"], spin: 0.5, mass: 938.272089,  isospin: 0.5, isospin3: 1/2, decayTime: "Stable"),
    Baryon(name: "Neutron",      quarkContent: ["u", "d", "d"], spin: 0.5, mass: 939.565421,  isospin: 0.5, isospin3: -1/2, decayTime: "~880"),
    Baryon(name: "Lambda0",      quarkContent: ["u", "d", "s"], spin: 0.5, mass: 1115.683, isospin: 0.0, isospin3: 0,    decayTime: "2.6e-10"),
    Baryon(name: "Sigma+",       quarkContent: ["u", "u", "s"], spin: 0.5, mass: 1189.370, isospin: 1.0, isospin3: 1,   decayTime: "8.0e-11"),
    Baryon(name: "Sigma0",       quarkContent: ["u", "d", "s"], spin: 0.5, mass: 1192.642, isospin: 1.0, isospin3: 0,    decayTime: "6.0e-20"),
    Baryon(name: "Sigma−",       quarkContent: ["d", "d", "s"], spin: 0.5, mass: 1197.449, isospin: 1.0, isospin3: -1,   decayTime: "1.5e-10"),
    Baryon(name: "Delta++",      quarkContent: ["u", "u", "u"], spin: 1.5, mass: 1232.000, isospin: 1.5, isospin3: 3/2, decayTime: "6.0e-24"),
    Baryon(name: "Delta+",       quarkContent: ["u", "u", "d"], spin: 1.5, mass: 1232.000, isospin: 1.5, isospin3: 1/2, decayTime: "6.0e-24"),
    Baryon(name: "Delta0",       quarkContent: ["u", "d", "d"], spin: 1.5, mass: 1232.000, isospin: 1.5, isospin3: -1/2, decayTime: "6.0e-24"),
    Baryon(name: "Delta−",       quarkContent: ["d", "d", "d"], spin: 1.5, mass: 1232.000, isospin: 1.5, isospin3: -3/2, decayTime: "6.0e-24"),
    Baryon(name: "Xi0",          quarkContent: ["u", "s", "s"], spin: 0.5, mass: 1314.860, isospin: 0.5, isospin3: 1/2, decayTime: "2.9e-10"),
    Baryon(name: "Xi−",          quarkContent: ["d", "s", "s"], spin: 0.5, mass: 1321.710, isospin: 0.5, isospin3: -1/2, decayTime: "1.64e-10"),
    Baryon(name: "Omega−",       quarkContent: ["s", "s", "s"], spin: 1.5, mass: 1672.450, isospin: 0.0, isospin3: 0,    decayTime: "8.2e-11"),
    Baryon(name: "Lambda+c",     quarkContent: ["u", "d", "c"], spin: 0.5, mass: 2286.460, isospin: 0.0, isospin3: 0,    decayTime: "2.0e-13"),
    Baryon(name: "Sigma++c",     quarkContent: ["u", "u", "c"], spin: 0.5, mass: 2453.970, isospin: 1.0, isospin3: 1,   decayTime: "1.0e-21"),
    Baryon(name: "Sigma+c",      quarkContent: ["u", "d", "c"], spin: 0.5, mass: 2452.900, isospin: 1.0, isospin3: 0,    decayTime: "1.0e-21"),
    Baryon(name: "Sigma0c",      quarkContent: ["d", "d", "c"], spin: 0.5, mass: 2453.750, isospin: 1.0, isospin3: -1,   decayTime: "1.0e-21"),
    Baryon(name: "Xi0c",         quarkContent: ["d", "s", "c"], spin: 0.5, mass: 2470.850, isospin: 0.5, isospin3: -1/2, decayTime: "1.1e-13"),
    Baryon(name: "Xi+c",         quarkContent: ["u", "s", "c"], spin: 0.5, mass: 2468.000, isospin: 0.5, isospin3: 1/2, decayTime: "4.4e-13"),
    Baryon(name: "Omega0c",      quarkContent: ["s", "s", "c"], spin: 0.5, mass: 2695.200, isospin: 0.0, isospin3: 0,    decayTime: "6.9e-14"),
    Baryon(name: "Lambda0b",     quarkContent: ["u", "d", "b"], spin: 0.5, mass: 5619.600, isospin: 0.0, isospin3: 0,    decayTime: "1.2e-12"),
    Baryon(name: "Xi0b",         quarkContent: ["u", "s", "b"], spin: 0.5, mass: 5787.800, isospin: 0.5, isospin3: 1/2, decayTime: "1.5e-12"),
    Baryon(name: "Xi−b",         quarkContent: ["d", "s", "b"], spin: 0.5, mass: 5794.400, isospin: 0.5, isospin3: -1/2, decayTime: "1.6e-12"),
    Baryon(name: "Omega−b",      quarkContent: ["s", "s", "b"], spin: 0.5, mass: 6045.100, isospin: 0.0, isospin3: 0,    decayTime: "1.3e-12")
]


// --- Constants from the model ---
let E_base = 922.00075
let E_spin = 255.39
let E_iso = 17.8
let E_i3 = 1.75
let E_bottom = 51.3
let E_charm = 79.7
let A_sin = -6.7
let T_sin = 98.65

let generation: [String: Int] = [
    "u": 1, "d": 1,
    "s": 2, "c": 2,
    "b": 3
]

let transitionCost: [String: Double] = [
    "u→d": 3.08,
    "d→s": 181.1,
    "s→c": 1237.3,
    "c→b": 3253.29425
]

func deltaEbase(for quarks: [String]) -> Double {
    var total = 0.0
    for q in quarks {
        switch q {
        case "u":
            total += 0
        case "d":
            total += 3.08
        case "s":
            total += 3.08 + 181.1
        case "c":
            total += 3.08 + 181.1 + 1237.3
        case "b":
            total += 3.08 + 181.1 + 1237.3 + 3253.29425
        default:
            total += 0
        }
    }
    return total
}


// Compute the predicted mass
func predictedMass(for baryon: Baryon) -> Double {
    let G = baryon.quarkContent.compactMap { generation[$0] }.max() ?? 1

    let Gmin = baryon.quarkContent.compactMap { generation[$0] }.min() ?? 1

    let charmed: Double = baryon.quarkContent.contains("c") ? 1.0 : 0.0
    let bottom: Double = baryon.quarkContent.contains("b") ? 1.0 : 0.0

    let deltaE = deltaEbase(for: baryon.quarkContent)
    let spinTerm = E_spin * (baryon.spin - 1/2) * ((5.0 - Double(G))/4.0)
    let isoTerm = E_iso * 2.0 * (baryon.isospin) * Double(G)

    var sinTerm = A_sin * sin((2.0 * .pi * (spinTerm + isoTerm - 2.75)/T_sin)) * Double(G)

    let i3Term = E_i3 * baryon.isospin3 / Double(G)

    let charmTerm = charmed * cos((.pi * isoTerm + 2/3)) * E_charm
    let bottomTerm = bottom * sin((.pi * isoTerm + 3/4)) * E_bottom


    return E_base + deltaE + spinTerm  + i3Term  + sinTerm + isoTerm - charmTerm + bottomTerm
}


var totalError = 0.0

// Run model and compare
for baryon in baryons {
    let predicted = predictedMass(for: baryon)
    let error = ((predicted - baryon.mass) / baryon.mass) * 100.0
    print(String(format: "%@ | Obs: %7.5f | Pred: %7.5f | Δ%%: %+5.5f%%",baryon.name.padding(toLength: 12, withPad: " ", startingAt: 0), baryon.mass, predicted,error))
    totalError += abs(error)
}

print((totalError)/24)
