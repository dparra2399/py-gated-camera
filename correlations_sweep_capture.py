# sweep.py
import numpy as np
import subprocess


BASE = [
    "python", "correlations_single_capture.py",
    "--k", "4",
    "--im_width", "512",
    "--burst_time", "100",
    "--bit_depth", "12",
    "--int_time", "1000",
    "--gate_step_size", "600",
    "--rep_rate", "10000000",
    "--plot_correlations", "false",
    "--save_into_file", "true",
    "--timeout", "0",
    "--current", "16",
    "--low_level_amplitude", "-0.5",

]

for typ in ["coarse", "ham", "trapcoarse"]:
        illum_typ = 'pulse' if typ == 'ham' else 'gaussian'
        gate_shrinkage = '5' #'20' if typ == 'ham' else '10'
        #duty = '20' if typ == 'ham' else '30'
        duty = "15" if typ == "ham" else "23" #"30"
        high_level_amp=  "0.77" if typ == "ham" else "0.54"
        #high_level_amp=  "0.5" if typ == "ham" else "0.42"


        cmd = BASE + [
            "--capture_type", typ,
            "--gate_shrinkage", gate_shrinkage,
            "--duty", duty,
            "--illum_type", illum_typ,
            "--high_level_amplitude", str(high_level_amp),
        ]
        print("==============================================================")
        print(f"type={typ}  amp={high_level_amp}")
        print("==============================================================")

        subprocess.run(cmd)
