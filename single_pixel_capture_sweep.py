# sweep.py
import subprocess
import numpy as np
from itertools import product

SCRIPT = "single_pixel_capture.py"

BASE = [
    "python", SCRIPT,
    "--k", "3",
    "--im_width", "512",
    "--burst_time", "4800",
    "--bit_depth", "12",
    "--int_time", "100",
    "--ground_truth_int_time", "10_000",
    "--ground_truth", "1",
    "--rep_rate", "5000000",
    "--save_into_file", "1",
    "--iterations", "1",
    "--current", "50"
]

# sweeps
capture_types = ["coarse", "ham"]
phase_shifts = np.arange(60, 300, 30).tolist()  # <-- set whatever you want (degrees or whatever your script expects)

print(phase_shifts)

run_id = 8

# INNER LOOP = capture types share the SAME run_id folder
for typ in capture_types:

    high_level_amp=  "5.0" if typ == "ham" else "4.4"
    low_level_amps = "-5.0"
    illum_typ = "square" if typ == "ham" else "gaussian"
    gate_shrinkage = "25" #"25" if typ == "ham" else "10"
    duty = "20" if typ == "ham" else "30"

    cmd = BASE + [
        "--phase_shifts", ",".join(str(item) for item in phase_shifts),
        "--capture_type", typ,
        "--gate_shrinkage", str(gate_shrinkage),
        "--duty", str(duty),
        "--illum_type", illum_typ,
        "--high_level_amplitude", str(high_level_amp),
        "--low_level_amplitude", str(low_level_amps),

        "--exp_path", f"exp_{run_id}",
    ]

    print( ",".join(str(item) for item in phase_shifts))
    print(f"  -> running capture_type={typ} \n \n")
    subprocess.run(cmd, check=True)
