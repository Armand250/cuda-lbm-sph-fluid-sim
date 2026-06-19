import glob
import os
import argparse
import re
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Command-Line argument parsing
parser = argparse.ArgumentParser(description='LBM Fluid Dynamics Dashboard Stitcher')
parser.add_argument('--mode', type=str, required=True, choices=['cpu', 'gpu'], 
                    help='Simulation execution mode (cpu or gpu)')
args = parser.parse_args()

# Set base paths based on execution mode
DATA_DIR = os.path.join("output", args.mode)
master_output_path = os.path.join("output", f"fluid_simulation_{args.mode}.gif")

# Obstacle configurations
SCENE_KEYS = [
    "Circle",
    "Vertical_Plate",
    "Airfoil_Wing",
    "V_Cup_Trap",
    "All_Shapes_Combined",
]

scene_files = {}
scene_metadata = {}

for scene in SCENE_KEYS:
    current_path = os.path.join(DATA_DIR, scene)
    search_path = os.path.join(current_path, "frame_*.csv")

    files = glob.glob(search_path)
    files.sort(key=lambda f: int(re.sub(r"\D", "", f)))

    if not files:
        exit(1)

    scene_files[scene] = files

    # Extract time data from the first frame's header comment block
    execution_time_str = "0.000 s"
    mlups_str = "0.00 MLUPS"
    with open(files[0], "r") as f:
        first_line = f.readline()
        if first_line.startswith("#"):
            metadata = first_line.replace("#", "").strip().split(",")
            if len(metadata) >= 2:
                execution_time_str = metadata[0].strip()
                mlups_str = metadata[1].strip()

    scene_metadata[scene] = (execution_time_str, mlups_str)

# Ensure frame counts match perfectly across all channels
num_frames = min(len(scene_files[s]) for s in SCENE_KEYS)

if args.mode == "gpu":
    mode_title = "LBM Fluid Dynamics Dashboard - Optimized Parallel GPU"
else:
    mode_title = "LBM Fluid Dynamics Dashboard - Sequential CPU Baseline"

# Grid plot layout figure
fig, axs = plt.subplots(2, 3, figsize=(18, 9.5))
fig.suptitle(mode_title, fontsize=18, fontweight="bold", y=0.98)

axs = axs.flatten()

images = []
ax_titles = []

for idx, scene in enumerate(SCENE_KEYS):
    ax = axs[idx]

    # Load initial frame configuration
    initial_data = np.loadtxt(scene_files[scene][0], delimiter=",", comments="#")
    im = ax.imshow(initial_data, cmap="jet", origin="lower", aspect="equal")
    images.append(im)

    # Format axis labels
    clean_name = scene.replace("_", " ")
    time_val, mlups_val = scene_metadata[scene]
    t = ax.set_title(
        f"{clean_name}\nTime: {time_val} | Throughput: {mlups_val}",
        fontsize=10,
        fontweight="bold",
    )
    ax_titles.append(t)

    ax.set_xticks([])
    ax.set_yticks([])

ax_summary = axs[5]
ax_summary.axis("off")
summary_text = (
    "Simulation Experiment Metrics\n"
    "--------------------------------------\n"
    f"Grid Lattice Resolution: {initial_data.shape[1]} x {initial_data.shape[0]}\n"
    f"Total Evaluated Time Steps: 1000\n"
    f"Engine Infrastructure: {args.mode.upper()}\n"
    "Collision Model: D2Q9 BGK Relaxation\n"
    "Inflow Condition: Zou-He Velocity\n"
    "Outflow Condition: Convective Wave"
)
ax_summary.text(
    0.05,
    0.5,
    summary_text,
    fontsize=11,
    fontfamily="monospace",
    verticalalignment="center",
)

plt.tight_layout(rect=[0, 0.02, 1, 0.94])


# Synchronized animation step
def update_dashboard(frame_idx):
    updated_artists = []
    for idx, scene in enumerate(SCENE_KEYS):
        frame_file = scene_files[scene][frame_idx]
        data = np.loadtxt(frame_file, delimiter=",", comments="#")

        images[idx].set_array(data)
        images[idx].set_clim(vmin=0, vmax=0.15)
        updated_artists.append(images[idx])

    return updated_artists


# Generate the GIF
ani = animation.FuncAnimation(
    fig, update_dashboard, frames=num_frames, blit=True
)

ani.save(master_output_path, writer="pillow", fps=10)
print(f"Animation generated successfully.")
plt.close(fig)