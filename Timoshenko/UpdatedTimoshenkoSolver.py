
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.linalg import eigh

# Model properties
E, G, RHO = 200e9, 81.2e9, 7850
KAPPA, ELEMENTS, MODES = 6/7, 16, 10

# Load geometry
GEOMETRY = pd.read_csv("geometry.csv").to_dict("records")

for s in GEOMETRY:
    s["length"] = float(s["length"]) / 1000
    s["diameter"] = float(s["diameter"]) / 1000
    s["type"] = str(s["type"]).strip().lower()

# Interpolate shaft diameter through bearing sections
for i, s in enumerate(GEOMETRY):
    if s["type"] == "bearing":
        before = next((GEOMETRY[j]["diameter"] for j in range(i-1, -1, -1)
                       if GEOMETRY[j]["type"] == "shaft"), None)
        after = next((GEOMETRY[j]["diameter"] for j in range(i+1, len(GEOMETRY))
                      if GEOMETRY[j]["type"] == "shaft"), None)
        values = [d for d in (before, after) if d is not None]
        if not values:
            raise ValueError(f"No adjacent shaft diameter for {s['name']}")
        s["diameter"] = np.mean(values)

# Timoshenko beam matrices
def beam_matrices(d, L):
    A = np.pi*d**2/4
    I = np.pi*d**4/64
    phi = 12*E*I/(KAPPA*G*A*L**2)

    k = E*I/(L**3*(1+phi))*np.array([
        [12, 6*L, -12, 6*L],
        [6*L, (4+phi)*L**2, -6*L, (2-phi)*L**2],
        [-12, -6*L, 12, -6*L],
        [6*L, (2-phi)*L**2, -6*L, (4+phi)*L**2]
    ])

    m = RHO*A*L/420*np.array([
        [156, 22*L, 54, -13*L],
        [22*L, 4*L**2, 13*L, -3*L**2],
        [54, 13*L, 156, -22*L],
        [-13*L, -3*L**2, -22*L, 4*L**2]
    ])

    m[np.ix_([1, 3], [1, 3])] += RHO*I*L/6*np.array([
        [2, 1], [1, 2]
    ])

    return k, m

# Build FEM mesh
nodes = [0.0]
elements = []
bearings = []
x = 0.0

for s in GEOMETRY:
    L = s["length"]/ELEMENTS
    start = len(nodes)-1

    for j in range(ELEMENTS):
        elements.append((start+j, start+j+1, s["diameter"], L))
        x += L
        nodes.append(x)

    if s["type"] == "bearing":
        bearings.append(start+ELEMENTS//2)

nodes = np.array(nodes)
ndof = 4*len(nodes)
K = np.zeros((ndof, ndof))
M = np.zeros((ndof, ndof))

# Assemble FEM matrices
for n1, n2, d, L in elements:
    ke, me = beam_matrices(d, L)

    for offset in (0, 2):
        dofs = [4*n1+offset, 4*n1+offset+1,
                4*n2+offset, 4*n2+offset+1]
        K[np.ix_(dofs, dofs)] += ke
        M[np.ix_(dofs, dofs)] += me

# Apply bearing constraints
fixed = [4*n+d for n in bearings for d in (0, 2)]
free = np.setdiff1d(np.arange(ndof), fixed)

# Solve modal eigenvalue problem
count = min(MODES, len(free))
values, vectors = eigh(
    K[np.ix_(free, free)],
    M[np.ix_(free, free)],
    subset_by_index=[0, count-1]
)

frequencies = np.sqrt(np.maximum(values, 0))/(2*np.pi)

# Save results
with open("output.txt", "w") as f:
    f.write("ROTOR - TIMOSHENKO MODAL ANALYSIS\n")
    f.write("======================\n\n")
    f.write(f"Nodes: {len(nodes)}\n")
    f.write(f"Elements: {len(elements)}\n")
    f.write(f"Bearing nodes: {bearings}\n\n")
    f.write("MODE       NATURAL FREQUENCY (Hz)\n")
    f.write("--------------------------------\n")

    for i, freq in enumerate(frequencies):
        f.write(f"{i+1:<11}{freq:.4f}\n")

print("\nNatural Frequencies")
for i, freq in enumerate(frequencies):
    print(f"Mode {i+1}: {freq:.3f} Hz")

print("\nResults saved to output.txt")

# Create geometry display in millimetres
plotter = pv.Plotter(window_size=(1400, 950))
plotter.set_background("white")

x_mm = 0.0
max_radius_mm = 0.0

for s in GEOMETRY:
    L_mm = s["length"]*1000
    d_mm = s["diameter"]*1000
    max_radius_mm = max(max_radius_mm, d_mm/2)

    cylinder = pv.Cylinder(
        center=(x_mm+L_mm/2, 0, 0),
        direction=(1, 0, 0),
        radius=d_mm/2,
        height=L_mm,
        resolution=64
    )

    color = "orange" if s["type"] == "bearing" else "steelblue"

    plotter.add_mesh(
        cylinder,
        color=color,
        smooth_shading=True,
        show_edges=True,
        line_width=0.5
    )

    x_mm += L_mm

# Set camera focal point and scale
total_length_mm = x_mm
plotter.camera.focal_point = (total_length_mm/2, 0, 0)
plotter.camera.position = (
    total_length_mm/2,
    -total_length_mm*1.4,
    total_length_mm*0.9
)
plotter.camera.up = (0, 0, 1)
plotter.camera.zoom(1.15)

# Display grid in millimetres
margin = max_radius_mm*1.5
bounds = (
    0, total_length_mm,
    -margin, margin,
    -margin, margin
)

plotter.show_bounds(
    bounds=bounds,
    grid="back",
    location="outer",
    ticks="outside",
    xlabel="Length (mm)",
    ylabel="Width (mm)",
    zlabel="Height (mm)",
    font_size=10,
    fmt="%.0f"
)

# Frequency table
table = "ROTOR - MODAL ANALYSIS\n\n"
table += "MODE       NATURAL FREQUENCY (Hz)\n"
table += "--------------------------------\n"

for i, freq in enumerate(frequencies):
    table += f"{i+1:<11}{freq:.4f}\n"

table += "\nBlue: Shaft   Orange: Bearing"

plotter.add_text(
    table,
    position="upper_left",
    font_size=8,
    color="black",
    viewport=True,
    shadow=False
)

# Show model
plotter.show()
