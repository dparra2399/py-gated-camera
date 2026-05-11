# sweep.py
import subprocess
import numpy as np


SCRIPT = "single_pixel_capture.py"

BASE = [
    "python", SCRIPT,
    "--k", "16",
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
capture_types = ["timeslicing"]
phase_shifts = np.arange(20, 340, 30).tolist()  # <-- set whatever you want (degrees or whatever your script expects)

print(phase_shifts)
print(len(phase_shifts))

run_id = 0

# INNER LOOP = capture types share the SAME run_id folder
for typ in capture_types:

    #high_level_amp=  "0.5"  if typ == "ham" else "0.42"
    #high_level_amp=  "0.5"  if typ == "ham" else "0.54"
    #high_level_amp = "0.42" if typ == "trapcoarse" else "0.23"
    high_level_amp = "1.2"

    low_level_amps = "-0.5"
    #illum_typ = "square" if typ == "ham" else "gaussian"
    #illum_typ = "gaussian" if typ == "trapcoarse" else "pulse"
    illum_typ = "gaussian"
    gate_shrinkage = "0" #"25" #"25" if typ == "ham" else "10"
    #duty = "15" if typ == "ham" else "23" #"30"
    #duty = "20" if typ == "ham" else "30"  # "30"
    duty = "12"

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
    run_id += 1
