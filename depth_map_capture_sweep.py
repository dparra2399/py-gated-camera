# sweep.py
import subprocess
from itertools import product

SCRIPT = "depth_map_capture.py"

BASE = [
    "python", SCRIPT,
    "--k", "3",
    "--im_width", "512",
    "--burst_time", "4800",
    "--bit_depth", "12",
    "--int_time", "1000",
    "--ground_truth_int_time", "500000",
    "--ground_truth", "1",
    "--rep_rate", "5000000",
    "--save_into_file", "1",
    "--iterations", "1",
]

# sweeps
capture_types = ["coarse", "ham"]
high_level_amps = [4.0]
low_level_amps  = [-4.0]
currs = [50]
phases = [45, 90, 135, 180]   # <-- set whatever you want (degrees or whatever your script expects)

run_id = 3

# OUTER LOOP = things that define a "run folder"
for phase, high_lvl, low_lvl, cur in product(phases, high_level_amps, low_level_amps, currs):

    print("=" * 70)
    print(f"[{run_id:03d}] phase={phase}  high={high_lvl:4.1f}  low={low_lvl:5.1f}  cur={cur:3d}")
    print("=" * 70)

    # INNER LOOP = capture types share the SAME run_id folder
    for typ in capture_types:
        illum_typ = "square" if typ == "ham" else "gaussian"
        gate_shrinkage = "20" if typ == "ham" else "10"
        duty = "20" if typ == "ham" else "30"

        cmd = BASE + [
            "--phase", str(phase),
            "--capture_type", typ,
            "--gate_shrinkage", str(gate_shrinkage),
            "--duty", str(duty),
            "--illum_type", illum_typ,
            "--high_level_amplitude", str(high_lvl),
            "--low_level_amplitude", str(low_lvl),
            "--current", str(cur),

            # IMPORTANT: if exp_path is typed as int in argparse, pass just the int
            "--exp_path", str(run_id),
        ]

        print(f"  -> running capture_type={typ}")
        subprocess.run(cmd, check=True)

    # increment ONCE per outer sweep combo (not per capture type)
    run_id += 1