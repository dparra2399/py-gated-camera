# sweep.py
import numpy as np
import subprocess


high_level_amps = [4.0]
low_level_amps = [-4.0]
currs = [50]

BASE = [
    "python", "correlations_single_capture.py",
    "--k", "3",
    "--im_width", "512",
    "--burst_time", "480",
    "--bit_depth", "12",
    "--int_time", "200",
    "--gate_step_size", "2500",
    "--rep_rate", "5000000",
    "--plot_correlations", "false",
    "--save_into_file", "true",
    "--timeout", "0"
]

run_id = 0
for typ in ["coarse", "ham"]:
    for high_lvl in high_level_amps:
        for low_lvl in low_level_amps:
            for cur in currs:
                illum_typ = 'square' if typ == 'ham' else 'gaussian'
                gate_shrinkage = '20' if typ == 'ham' else '10'
                duty = '20' if typ == 'ham' else '30'
                cmd = BASE + [
                    "--capture_type", typ,
                    "--gate_shrinkage", gate_shrinkage,
                    "--duty", duty,
                    "--illum_type", illum_typ,
                    "--high_level_amplitude", str(high_lvl),
                    "--low_level_amplitude", str(low_lvl),
                    "--current", str(cur),
                ]
                run_id += 1
                print("==============================================================")
                print(f"[{run_id:03d}] type={typ:6s}  amp={high_lvl:4.1f}  current={cur:3d}")
                print("==============================================================")

                subprocess.run(cmd)
