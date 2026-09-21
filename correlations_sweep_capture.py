# sweep.py
import numpy as np
import subprocess

from illum_config import get_illum

K = 16

BASE = [
    "python", "correlations_single_capture.py",
    "--k", str(K),
    "--im_width", "128",
    "--burst_time", "0.05",
    "--bit_depth", "12",
    "--int_time", "2",
    "--gate_step_size", "1200",
    "--rep_rate", "10000000",
    "--plot_correlations", "false",
    "--save_into_file", "true",
    "--timeout", "0",
    "--current", "16",
    "--low_level_amplitude", "-0.5",

]

# (capture_type, illum_type) pairs to run; illumination pulled from illum_config
RUNS = [("coarse", "gaussian")] #[("ham", "pulse")]
#("ham", "square"), ("coarse", "gaussian"),
for typ, illum_typ in RUNS:
        illum = get_illum(K, typ, illum_typ)

        cmd = BASE + [
            "--capture_type", typ,
            "--gate_shrinkage", str(illum["gate_shrinkage"]),
            "--duty", str(illum["duty"]),
            "--illum_type", illum_typ,
            "--high_level_amplitude", str(illum["high_level_amplitude"]),
        ]
        print("==============================================================")
        print(f"type={typ}  illum={illum_typ}  amp={illum['high_level_amplitude']}")
        print("==============================================================")

        subprocess.run(cmd)
