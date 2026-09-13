# sweep.py
import subprocess
import numpy as np

from illum_config import get_illum

SCRIPT = "single_pixel_capture.py"

K = 4  # capture K; must have matching rows in illum_config

BASE = [
    "python", SCRIPT,
    "--k", str(K),
    "--im_width", "512",
    "--burst_time", "10",
    "--int_time", "30",
    "--split_acquisition", "1",
    "--bit_depth", "12",
    "--ground_truth_int_time", "30", #40
    "--ground_truth", "0",
    "--rep_rate", "10000000",
    "--save_into_file", "1",
    "--iterations", "1",
    "--current", "16",
    "--trials", "100",
]

# sweeps
# (capture_type, illum_type) pairs to run; illumination pulled from illum_config
RUNS = [("coarse", "gaussian"), ("ham", "pulse")]
phase_shifts = np.arange(20, 340, 30).tolist()  # <-- set whatever you want (degrees or whatever your script expects)

print(phase_shifts)
print(len(phase_shifts))

run_id = 0

# INNER LOOP = capture types share the SAME run_id folder
for typ, illum_typ in RUNS:

    illum = get_illum(K, typ, illum_typ)

    cmd = BASE + [
        "--phase_shifts", ",".join(str(item) for item in phase_shifts),
        "--capture_type", typ,
        "--gate_shrinkage", str(illum["gate_shrinkage"]),
        "--duty", str(illum["duty"]),
        "--illum_type", illum_typ,
        "--high_level_amplitude", str(illum["high_level_amplitude"]),
        "--low_level_amplitude", str(illum["low_level_amplitude"]),
        "--exp_path", f"exp_{run_id}",
    ]

    print( ",".join(str(item) for item in phase_shifts))
    print(f"  -> running capture_type={typ} illum={illum_typ} {illum} \n \n")
    subprocess.run(cmd, check=True)
    run_id += 1
