
import pyvista as pv

def show_geometry(GEOMETRY):

    plotter = pv.Plotter(window_size=[1600, 900])
    plotter.set_background("#20252B")

    x_s = 0

    for part in GEOMETRY:
        length = part["length"]
        diameter = part["diameter"]

        plotter.add_mesh(
            pv.Cylinder(
                center=(x_s + length / 2, 0, 0),
                direction=(1, 0, 0),
                radius=diameter / 2,
                height=length,
                resolution=100
            ),
            color="#D69B45" if part["type"] == "bearing" else "steelblue",
            smooth_shading=True
        )

        part["position"] = x_s
        x_s += length

    BEARING_POSITIONS = [
        p["position"] + p["length"] / 2
        for p in GEOMETRY if p["type"] == "bearing"
    ]

    plotter.add_text(
        "Rotor with Bearings",
        position="upper_left",
        font_size=8,
        color="white"
    )

    plotter.show_grid(
        xtitle="Length (mm)",
        ytitle="Width (mm)",
        ztitle="Height (mm)",
        color="#50565C",
        font_size=8,
        bold=False
    )

    plotter.add_axes()
    plotter.view_isometric()
    plotter.reset_camera()
    plotter.show()