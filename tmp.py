import numpy as np
import os
from glob import glob

# Path to directory
folder = "/Volumes/velten/Research_Users/David/gated_project_data"

# Get all .npz files in the directory
npz_files = glob(os.path.join(folder, "*.npz"))

for path in npz_files:
    if path.endswith("exp1.npz"):
        continue
    try:
        # Load the .npz file
        file = np.load(path)

        # Convert to dict
        params = {k: v for k, v in file.items()}

        # Add freq
        params["gate_width"] = 7  # 10 MHz
        params["freq"] = 10_000_000

        # Resave (overwrite original)
        np.savez(path, **params)

        print(f"Updated: {os.path.basename(path)}")
    except Exception as e:
        print(f"Failed to update {path}: {e}")
