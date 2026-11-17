
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ross as rs
from pathlib import Path

# MODEL SETTINGS
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = "geometry.csv"
OUTPUT_FILE = "output-ross.txt"

NUMBER_OF_MODES = 20
ELEMENTS_PER_SECTION = 16
ROTOR_SPEED = 0.0

BEAM_THEORY = "Timoshenko"
BEARING_STIFFNESS = 1e15

# MATERIAL
MATERIAL = rs.Material(
    name="Steel",
    rho=7850,
    E=200e9,
    G_s=81.2e9
)

print(MATERIAL)

# LOAD GEOMETRY
geometry = pd.read_csv(CSV_FILE)

geometry["name"] = geometry["name"].str.strip()
geometry["type"] = geometry["type"].str.strip().str.lower()

required_columns = {"name", "length", "diameter", "type"}

if not required_columns.issubset(geometry.columns):
    raise ValueError("Invalid geometry.csv columns.")

if not geometry["type"].isin(["shaft", "bearing"]).all():
    raise ValueError("Type must be shaft or bearing.")

if (geometry[["length", "diameter"]] <= 0).any().any():
    raise ValueError("Length and diameter must be positive.")

lengths = geometry["length"].to_numpy() / 1000
diameters = geometry["diameter"].to_numpy() / 1000

positions = np.concatenate(([0], np.cumsum(lengths)))

shaft_indices = np.where(
    geometry["type"].to_numpy() == "shaft"
)[0]

if len(shaft_indices) < 2:
    raise ValueError("At least two shaft sections are required.")

# CREATE SHAFT AND BEARINGS
shaft_elements = []
bearing_elements = []
bearing_info = []

shear_effects = BEAM_THEORY == "Timoshenko"
rotary_inertia = BEAM_THEORY == "Timoshenko"

for i, row in geometry.iterrows():

    start_node = i * ELEMENTS_PER_SECTION
    element_length = lengths[i] / ELEMENTS_PER_SECTION

    if row["type"] == "shaft":

        d_left = diameters[i]
        d_right = diameters[i]

    else:

        previous = shaft_indices[shaft_indices < i]
        following = shaft_indices[shaft_indices > i]

        if len(previous) == 0 or len(following) == 0:
            raise ValueError(
                f"Cannot determine shaft diameter through {row['name']}."
            )

        # Constant diameter throughout bearing section
        d_bearing = (
            diameters[previous[-1]] + diameters[following[0]]
        ) / 2

        d_left = d_bearing
        d_right = d_bearing

    # CREATE SHAFT ELEMENTS
    for j in range(ELEMENTS_PER_SECTION):

        f1 = j / ELEMENTS_PER_SECTION
        f2 = (j + 1) / ELEMENTS_PER_SECTION

        od_left = d_left + (d_right - d_left) * f1
        od_right = d_left + (d_right - d_left) * f2

        element = rs.ShaftElement(
            L=element_length,
            idl=0,
            odl=od_left,
            idr=0,
            odr=od_right,
            material=MATERIAL,
            shear_effects=shear_effects,
            rotary_inertia=rotary_inertia,
            gyroscopic=False
        )

        shaft_elements.append(element)

    # BEARING SUPPORT AT MIDPOINT
    if row["type"] == "bearing":

        bearing_node = start_node + ELEMENTS_PER_SECTION // 2

        bearing = rs.BearingElement(
            n=bearing_node,
            kxx=BEARING_STIFFNESS,
            kyy=BEARING_STIFFNESS,
            cxx=0.0,
            cyy=0.0
        )

        bearing_elements.append(bearing)

        bearing_info.append({
            "name": row["name"],
            "node": bearing_node,
            "position": (positions[i] + lengths[i]/2)*1000
        })

# ASSEMBLE ROTOR
rotor = rs.Rotor(
    shaft_elements=shaft_elements,
    bearing_elements=bearing_elements
)

print("\nROTOR MODEL")
print("=" * 40)
print(f"Total length     : {positions[-1]*1000:.2f} mm")
print(f"Shaft elements   : {len(shaft_elements)}")
print(f"Bearing supports : {len(bearing_elements)}")
print(f"Beam theory      : {BEAM_THEORY}")

for bearing in bearing_info:
    print(
        f"{bearing['name']}: "
        f"Node {bearing['node']}, "
        f"Position {bearing['position']:.2f} mm"
    )

# DISPLAY ROTOR
rotor.plot_rotor().show()

# UNDAMPED MODAL ANALYSIS
modal = rotor.run_modal(
    speed=ROTOR_SPEED,
    num_modes=NUMBER_OF_MODES
)

frequencies = np.asarray(modal.wn)/(2*np.pi)
number_of_modes = min(NUMBER_OF_MODES, len(frequencies))

# EXTRACT MODE INFORMATION
mode_types = []

for i in range(number_of_modes):

    mode_type = "Unknown"

    if hasattr(modal, "shapes"):
        mode_type = getattr(
            modal.shapes[i],
            "mode_type",
            "Unknown"
        )

    mode_types.append(mode_type)

# SAVE RESULTS
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    file.write("JET ENGINE ROTORDYNAMICS - ROSS\n")
    file.write("=" * 65 + "\n\n")

    file.write("UNDAMPED MODAL ANALYSIS\n")
    file.write(f"Material        : Steel\n")
    file.write(f"Density         : {MATERIAL.rho} kg/m3\n")
    file.write(f"Young's Modulus : {MATERIAL.E:.3e} Pa\n")
    file.write(f"Shear Modulus   : {MATERIAL.G_s:.3e} Pa\n")
    file.write(f"Beam Theory     : {BEAM_THEORY}\n")
    file.write(f"Rotor Speed     : {ROTOR_SPEED} rad/s\n")
    file.write(f"Total Length    : {positions[-1]*1000:.2f} mm\n")

    file.write("\nBEARING LOCATIONS\n")
    file.write("-" * 65 + "\n")

    for bearing in bearing_info:
        file.write(
            f"{bearing['name']}: "
            f"Node {bearing['node']}, "
            f"{bearing['position']:.2f} mm\n"
        )

    file.write("\nUNDAMPED NATURAL FREQUENCIES\n")
    file.write("-" * 65 + "\n")
    file.write(f"{'Mode':<10}{'Frequency (Hz)':<20}{'Mode Type'}\n")

    for i in range(number_of_modes):
        file.write(
            f"{i+1:<10}"
            f"{frequencies[i]:<20.4f}"
            f"{mode_types[i]}\n"
        )

# PRINT FREQUENCIES
print("\nUNDAMPED NATURAL FREQUENCIES")
print("=" * 65)

for i in range(number_of_modes):
    print(
        f"Mode {i+1:02d} | "
        f"{frequencies[i]:10.3f} Hz | "
        f"{mode_types[i]}"
    )

print(f"\nResults saved: {OUTPUT_FILE}")

# FREQUENCY GRAPH
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    np.arange(1, number_of_modes+1),
    frequencies[:number_of_modes],
    "o-"
)

ax.set_xlabel("Mode Number")
ax.set_ylabel("Undamped Natural Frequency (Hz)")
ax.set_title("ROSS - Undamped Natural Frequencies")
ax.set_xticks(np.arange(1, number_of_modes+1))
ax.grid(alpha=0.25)

fig.tight_layout()

# ALL MODE DEFORMATIONS IN ONE WINDOW
fig, axes = plt.subplots(
    5, 2,
    figsize=(15, 18),
    sharex=True
)

axes = axes.flatten()

for i in range(number_of_modes):

    # Obtain ROSS mode-shape plot data
    shape = modal.shapes[i]

    # ROSS mode shape displacement components
    x_disp = np.asarray(shape.x)
    y_disp = np.asarray(shape.y)

    node_positions = np.asarray(shape.nodes_pos)*1000

    # Normalize displacement for visual comparison
    scale = max(
        np.max(np.abs(x_disp)),
        np.max(np.abs(y_disp)),
        1e-15
    )

    x_disp = x_disp / scale
    y_disp = y_disp / scale

    axes[i].plot(
        node_positions,
        x_disp,
        label="X displacement"
    )

    axes[i].plot(
        node_positions,
        y_disp,
        label="Y displacement"
    )

    axes[i].set_title(
        f"Mode {i+1} - {frequencies[i]:.2f} Hz"
    )

    axes[i].set_ylabel("Normalized displacement")
    axes[i].grid(alpha=0.25)
    axes[i].legend(fontsize=8)

    for bearing in bearing_info:
        axes[i].axvline(
            bearing["position"],
            color="gray",
            linestyle="--",
            alpha=0.5
        )

for ax in axes[-2:]:
    ax.set_xlabel("Rotor Length (mm)")

fig.suptitle(
    "ROSS - First 10 Undamped Mode Shapes",
    fontsize=15
)

fig.tight_layout(rect=[0, 0, 1, 0.98])

# SHOW BOTH WINDOWS
plt.show()
