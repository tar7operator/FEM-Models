
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from show_geometry import show_geometry

# Model
MODEL = {
    "name": "JET ENGINE ROTOR",
    "material": "Structural Steel",
    "density": 7850,
    "youngs_modulus": 200e9,
    "poisson_ratio": 0.30
}

# Geometry
GEOMETRY = pd.read_csv("geometry.csv").to_dict("records")
show_geometry(GEOMETRY)

E, RHO = MODEL["youngs_modulus"], MODEL["density"]

# Shaft sections
PARTS, x_s = [], 0.0

for p in GEOMETRY:
    L, d = p["length"] / 1000, p["diameter"] / 1000
    PARTS.append({"name": p["name"], "start": x_s, "end": x_s + L, "d": d, "type": p["type"]})
    x_s += L

TOTAL_LENGTH = x_s

# FEM nodes
BEARING_POSITIONS = [(p["start"] + p["end"]) / 2 for p in PARTS if p["type"] == "bearing"]
x_nodes = np.array(sorted(set([p["start"] for p in PARTS] + [PARTS[-1]["end"]] + BEARING_POSITIONS))) #getting all section ends and also the bearing
n_nodes, n_dof = len(x_nodes), 2 * len(x_nodes)

K, M = np.zeros((n_dof, n_dof)), np.zeros((n_dof, n_dof))

# Beam element assembly
for i, (x1, x2) in enumerate(zip(x_nodes[:-1], x_nodes[1:])):
    L, xm = x2 - x1, (x1 + x2) / 2
    d = next(p["d"] for p in PARTS if p["start"] <= xm < p["end"])
    A, I = np.pi * d**2 / 4, np.pi * d**4 / 64

    k = E * I / L**3 * np.array([
        [12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]
    ])

    m = RHO * A * L / 420 * np.array([
        [156, 22*L, 54, -13*L], [22*L, 4*L**2, 13*L, -3*L**2],
        [54, 13*L, 156, -22*L], [-13*L, -3*L**2, -22*L, 4*L**2]
    ])

    dofs = [2*i, 2*i+1, 2*i+2, 2*i+3]
    K[np.ix_(dofs, dofs)] += k
    M[np.ix_(dofs, dofs)] += m


# Bearing constraints
fixed = [2 * np.argmin(abs(x_nodes - xb)) for xb in BEARING_POSITIONS]
free = np.setdiff1d(np.arange(n_dof), fixed)

# Eigenvalue solution
values, vectors = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
valid = values > 1e-8
values, vectors = values[valid][:10], vectors[:, valid][:, :10]
frequencies = np.sqrt(values) / (2 * np.pi)

# Mode shapes
MODES = []

for i, f in enumerate(frequencies):
    mode = np.zeros(n_dof)
    mode[free] = vectors[:, i]
    disp = mode[0::2]
    disp /= np.max(np.abs(disp))
    MODES.append({"number": i + 1, "frequency": f, "displacement": disp})

# Results
print("\nMODEL:", MODEL["name"], "| MATERIAL:", MODEL["material"])
print("TOTAL LENGTH:", TOTAL_LENGTH * 1000, "mm")
print("BEARING POSITIONS:", np.array(BEARING_POSITIONS) * 1000, "mm")
print("\nNATURAL FREQUENCIES")
for m in MODES:
    print(f"Mode {m['number']}: {m['frequency']:.3f} Hz")

# Frequency plot
plt.figure(figsize=(9, 4))
plt.plot([m["number"] for m in MODES], frequencies, "o-")
plt.xlabel("Mode Number")
plt.ylabel("Natural Frequency (Hz)")
plt.title("Rotor Natural Frequencies")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# Displacement subplots
fig, axes = plt.subplots(5, 2, figsize=(13, 14), sharex=True)

for ax, mode in zip(axes.flat, MODES):
    ax.plot(x_nodes * 1000, mode["displacement"], linewidth=1.5)
    for xb in BEARING_POSITIONS:
        ax.axvline(xb * 1000, color="red", linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Mode {mode['number']} | {mode['frequency']:.2f} Hz")
    ax.set_ylabel("Normalized Disp.")
    ax.grid(alpha=0.2)

for ax in axes[-1]:
    ax.set_xlabel("Rotor Length (mm)")

fig.suptitle("First 10 Rotor Mode Shapes")
fig.tight_layout()
plt.show()
