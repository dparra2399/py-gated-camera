# sweep.py
import numpy as np
import subprocess


BASE = [
    "python", "correlations_single_capture.py",
    "--k", "3",
    "--im_width", "512",
    "--burst_time", "100",
    "--bit_depth", "12",
    "--int_time", "500",
    "--gate_step_size", "300",
    "--rep_rate", "10000000",
    "--plot_correlations", "false",
    "--save_into_file", "true",
    "--timeout", "0",
    "--current", "50",
    "--low_level_amplitude", "-4.0",

]

for typ in ["coarse", "ham"]:
        illum_typ = 'square' if typ == 'ham' else 'gaussian'
        gate_shrinkage = '5' #'20' if typ == 'ham' else '10'
        duty = '20' if typ == 'ham' else '30'
        high_lvl_tmp = '4.0' if typ == 'ham' else '3.4'
        cmd = BASE + [
            "--capture_type", typ,
            "--gate_shrinkage", gate_shrinkage,
            "--duty", duty,
            "--illum_type", illum_typ,
            "--high_level_amplitude", str(high_lvl_tmp),
        ]
        print("==============================================================")
        print(f"type={typ}  amp={high_lvl_tmp}")
        print("==============================================================")

        subprocess.run(cmd)
