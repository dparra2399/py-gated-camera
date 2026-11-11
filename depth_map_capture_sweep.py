# sweep.py
import subprocess
from itertools import product

from illum_config import get_illum

SCRIPT = "depth_map_capture.py"
K = 3


BASE = [
    "python", SCRIPT,
    "--k", str(K),
    "--im_width", "512",
    "--bit_depth", "12",
    "--split_acquisition", "1",
    "--int_time", "100",
    "--burst_time", "100",
    "--max_trials", "50",
    "--ground_truth_int_time", "2000",
    "--ground_truth", "1",
    "--rep_rate", "10000000",
    "--save_into_file", "1",
    "--iterations", "1",
    "--current", "16"
]

# sweeps
# (capture_type, illum_type) pairs to run; illumination pulled from illum_config
RUNS = [('ham', 'square')] #[("coarse", "gaussian"), ("ham", "pulse")]
phases = [60]   # <-- set whatever you want

# sliding only: gate width in ns (k sets the shift, the width is independent)
SLIDING_GATE_WIDTH = 50

run_id = 0

# OUTER LOOP = things that define a "run folder"
for phase in phases:


    # INNER LOOP = capture types share the SAME run_id folder
    for typ, illum_typ in RUNS:

        illum = get_illum(K, typ, illum_typ)

        cmd = BASE + [
            "--phase", str(phase),
            "--capture_type", typ,
            "--gate_shrinkage", str(illum["gate_shrinkage"]),
            "--duty", str(illum["duty"]),
            "--illum_type", illum_typ,
            "--high_level_amplitude", str(illum["high_level_amplitude"]),
            "--low_level_amplitude", str(illum["low_level_amplitude"]),

            "--exp_path", f"exp_{run_id}",
        ]
        if typ == "sliding":
            cmd += ["--sliding_gate_width", str(SLIDING_GATE_WIDTH)]

        print(f"  -> running capture_type={typ} illum={illum_typ} {illum}")
        subprocess.run(cmd, check=True)

    # increment ONCE per outer sweep combo (not per capture type)
    run_id += 1