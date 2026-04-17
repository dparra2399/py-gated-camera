# sweep.py
import subprocess
from itertools import product

SCRIPT = "depth_map_capture.py"

BASE = [
    "python", SCRIPT,
    "--k", "3",
    "--im_width", "512",
    "--burst_time", "100",
    "--bit_depth", "12",
    "--int_time", "500",
    "--ground_truth_int_time", "10000",
    "--ground_truth", "1",
    "--rep_rate", "10000000",
    "--save_into_file", "1",
    "--iterations", "1",
    "--current", "50"
]

# sweeps
capture_types = ["coarse", "ham"]
phases = [180]   # <-- set whatever you want (degrees or whatever your script expects)

run_id = 3

# OUTER LOOP = things that define a "run folder"
for phase in phases:


    # INNER LOOP = capture types share the SAME run_id folder
    for typ in capture_types:

        high_level_amp=  "4.0" if typ == "ham" else "3.4"
        low_level_amps = "-4.0"
        illum_typ = "square" if typ == "ham" else "gaussian"
        gate_shrinkage = "5" #20" if typ == "ham" else "10"
        duty = "20" if typ == "ham" else "30"

        cmd = BASE + [
            "--phase", str(phase),
            "--capture_type", typ,
            "--gate_shrinkage", str(gate_shrinkage),
            "--duty", str(duty),
            "--illum_type", illum_typ,
            "--high_level_amplitude", str(high_level_amp),
            "--low_level_amplitude", str(low_level_amps),

            "--exp_path", f"exp_{run_id}",
        ]

        print(f"  -> running capture_type={typ}")
        subprocess.run(cmd, check=True)

    # increment ONCE per outer sweep combo (not per capture type)
    run_id += 1