import math

# --- Constants from the model ---
E_base   = 922.00075
E_spin   = 255.39
E_iso    = 17.8
E_i3     = 1.75
E_bottom = 51.3
E_charm  = 79.7
A_sin    = -6.7
T_sin    = 98.65

# Quark generation mapping
generation = {
    "u": 1, "d": 1,
    "s": 2, "c": 2,
    "b": 3
}

# Compute deltaE_base from quark content
def deltaEbase(quarks):
    total = 0.0
    for q in quarks:
        if q == "u":
            total += 0
        elif q == "d":
            total += 3.08
        elif q == "s":
            total += 3.08 + 181.1
        elif q == "c":
            total += 3.08 + 181.1 + 1237.3
        elif q == "b":
            total += 3.08 + 181.1 + 1237.3 + 3253.29425
    return total

# Predict baryon mass from quantum properties
def predicted_mass(quarks, spin, isospin, isospin3):
    G = max(generation[q] for q in quarks)
    charmed = 1.0 if "c" in quarks else 0.0
    bottom  = 1.0 if "b" in quarks else 0.0

    deltaE  = deltaEbase(quarks)

    spinTerm = E_spin * (spin - 0.5) * ((5.0 - G) / 4.0)
    isoTerm  = E_iso * 2.0 * isospin * G
    i3Term   = E_i3 * isospin3 / G

    sinTerm = A_sin * math.sin((2 * math.pi * (spinTerm + isoTerm - 2.75) / T_sin)) * G

    charmTerm  = charmed * math.cos((math.pi * isoTerm + 2/3)) * E_charm
    bottomTerm = bottom * math.sin((math.pi * isoTerm + 3/4)) * E_bottom

    return E_base + deltaE + spinTerm + i3Term + sinTerm + isoTerm - charmTerm + bottomTerm

# Baryon dataset: name, quarks, spin, mass, isospin, isospin3
baryons = [
    ("Proton",      ["u", "u", "d"], 0.5,  938.272089, 0.5,  0.5),
    ("Neutron",     ["u", "d", "d"], 0.5,  939.565421, 0.5, -0.5),
    ("Lambda0",     ["u", "d", "s"], 0.5, 1115.683,    0.0,  0.0),
    ("Sigma+",      ["u", "u", "s"], 0.5, 1189.370,    1.0,  1.0),
    ("Sigma0",      ["u", "d", "s"], 0.5, 1192.642,    1.0,  0.0),
    ("Sigma-",      ["d", "d", "s"], 0.5, 1197.449,    1.0, -1.0),
    ("Delta++",     ["u", "u", "u"], 1.5, 1232.000,    1.5,  1.5),
    ("Delta+",      ["u", "u", "d"], 1.5, 1232.000,    1.5,  0.5),
    ("Delta0",      ["u", "d", "d"], 1.5, 1232.000,    1.5, -0.5),
    ("Delta-",      ["d", "d", "d"], 1.5, 1232.000,    1.5, -1.5),
    ("Xi0",         ["u", "s", "s"], 0.5, 1314.860,    0.5,  0.5),
    ("Xi-",         ["d", "s", "s"], 0.5, 1321.710,    0.5, -0.5),
    ("Omega-",      ["s", "s", "s"], 1.5, 1672.450,    0.0,  0.0),
    ("Lambda+c",    ["u", "d", "c"], 0.5, 2286.460,    0.0,  0.0),
    ("Sigma++c",    ["u", "u", "c"], 0.5, 2453.970,    1.0,  1.0),
    ("Sigma+c",     ["u", "d", "c"], 0.5, 2452.900,    1.0,  0.0),
    ("Sigma0c",     ["d", "d", "c"], 0.5, 2453.750,    1.0, -1.0),
    ("Xi0c",        ["d", "s", "c"], 0.5, 2470.850,    0.5, -0.5),
    ("Xi+c",        ["u", "s", "c"], 0.5, 2468.000,    0.5,  0.5),
    ("Omega0c",     ["s", "s", "c"], 0.5, 2695.200,    0.0,  0.0),
    ("Lambda0b",    ["u", "d", "b"], 0.5, 5619.600,    0.0,  0.0),
    ("Xi0b",        ["u", "s", "b"], 0.5, 5787.800,    0.5,  0.5),
    ("Xi-b",        ["d", "s", "b"], 0.5, 5794.400,    0.5, -0.5),
    ("Omega-b",     ["s", "s", "b"], 0.5, 6045.100,    0.0,  0.0)
]

# Run predictions and evaluate accuracy
total_error = 0.0
for name, quarks, spin, mass, isospin, isospin3 in baryons:
    pred = predicted_mass(quarks, spin, isospin, isospin3)
    error = ((pred - mass) / mass) * 100.0
    print(f"{name.ljust(12)} | Obs: {mass:9.6f} | Pred: {pred:9.6f} | Δ%: {error:+7.5f}%")
    total_error += abs(error)

print("Average | Δ% error: {:.5f}%".format(total_error / len(baryons)))
