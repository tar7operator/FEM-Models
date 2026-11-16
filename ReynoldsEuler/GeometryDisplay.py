
import numpy as np
import pyvista as pv

# Model
MODEL = {
    "name": "JET ENGINE ROTOR",
    "material": "Structural Steel",
    "density": 7850,
    "youngs_modulus": 210e9,
    "poisson_ratio": 0.30
}

# Geometry: Length (mm), Diameter (mm), Type
GEOMETRY = [
    {"name": "Front Stator",  "length": 100, "diameter": 80,  "type": "shaft"},
    {"name": "Front Bearing", "length": 40,  "diameter": 60,  "type": "bearing"},
    {"name": "Combustor",     "length": 200, "diameter": 100, "type": "shaft"},
    {"name": "Shaft 1",       "length": 200, "diameter": 30,  "type": "shaft"},
    {"name": "Shaft 2",       "length": 300, "diameter": 40,  "type": "shaft"},
    {"name": "Turbine",       "length": 120, "diameter": 90,  "type": "shaft"},
    {"name": "Rear Bearing",  "length": 40,  "diameter": 60,  "type": "bearing"},
    {"name": "Shaft 3",       "length": 200, "diameter": 30,  "type": "shaft"}
]

# Geometry and visualization
plotter = pv.Plotter(window_size=[1600, 900])

x_s = 0

for part in GEOMETRY:
    plotter.add_mesh(
        pv.Cylinder(
            center=(x_s + part["length"] / 2, 0, 0),
            direction=(1, 0, 0),
            radius=part["diameter"] / 2,
            height=part["length"],
            resolution=100
        ),
        color="#D69B45" if part["type"] == "bearing" else "steelblue",
        smooth_shading=True
    )
    part["position"] = x_s
    x_s += part["length"]

BEARING_POSITIONS = [
    p["position"] + p["length"] / 2
    for p in GEOMETRY if p["type"] == "bearing"
]

# Display
plotter.set_background("#20252B")
plotter.add_text("GEOMETRY", position="upper_left", font_size=8, color="white")
plotter.show_grid(
    xtitle="Length (mm)", ytitle="Width (mm)", ztitle="Height (mm)",
    color="#50565C", font_size=8, bold=False
)
plotter.add_axes()
plotter.view_isometric()
plotter.reset_camera()
plotter.show()

print("TOTAL LENGTH:", x_s, "mm")
print("BEARING POSITIONS:", BEARING_POSITIONS, "mm")
