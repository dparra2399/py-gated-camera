# sweep.py
import numpy as np
import subprocess

amps = np.arange(0.5, 3.1, 0.5)
currs = np.arange(50, 80, 10)

BASE = [
    "python", "capture_correlation_functions.py",
    "--k", "3",
    "--im_width", "512",
    "--bit_depth", "12",
    "--int_time", "200",
    "--shift", "2500",
    "--rep_rate", "5000000",
    "--plot_correlations", "false",
    "--save_into_file", "true",
]

run_id = 0
for typ in ["ham"]:
    for amp in amps:
        for cur in currs:
            cmd = BASE + [
                "--capture_type", typ,
                "--amplitude", str(amp),
                "--current", str(cur),
            ]
            run_id += 1
            print("==============================================================")
            print(f"[{run_id:03d}] type={typ:6s}  amp={amp:4.1f}  current={cur:3d}")
            print("==============================================================")

            subprocess.run(cmd)
