# sweep.py
import subprocess
import numpy as np

from illum_config import get_illum, ILLUM_CONFIG

SCRIPT = "single_pixel_capture.py"

K_VALUES = [4]  # <-- sweep these; illum combos are resolved from illum_config per K

BASE = [
    "python", SCRIPT,
    "--im_width", "128",
    "--burst_time", "0.05",
    "--int_time", "2",
    "--split_acquisition", "1",
    "--bit_depth", "12",
    "--ground_truth_int_time", "2",  # 40
    "--ground_truth", "0",
    "--rep_rate", "10000000",
    "--save_into_file", "1",
    "--iterations", "1",
    "--current", "16",
    "--trials", "10",
]

# Capture types to run. The illum_type for each is looked up from illum_config
# per K, so ham -> square at k=3, ham -> pulse at k=4, and ham is skipped
# wherever no row is configured (e.g. k=8).
CAPTURE_TYPES = ['ham'] #'["coarse", "trapcoarse"]

# sliding only: gate width in ns. k is the number of shifts, so the shift is tau/k and the gate
# width is independent of it -- wide overlapping gates with fine shifts is the whole point.
SLIDING_GATE_WIDTH = 50

phase_shifts = np.arange(20, 340, 30).tolist()

print(phase_shifts)
print(len(phase_shifts))


def runs_for_k(k):
    """(capture_type, illum_type) combos configured for this k, in CAPTURE_TYPES order."""
    combos = []
    for typ in CAPTURE_TYPES:
        # kk is None for schemes whose illumination does not depend on k (sliding)
        for illum_typ in [it for (kk, ct, it) in ILLUM_CONFIG if kk in (k, None) and ct == typ]:
            combos.append((typ, illum_typ))
    return combos


run_id = 5

# OUTER LOOP = K             -> each K gets its own run_id folder
# INNER LOOP = capture types -> share the SAME run_id folder
for K in K_VALUES:
    for typ, illum_typ in runs_for_k(K):

        illum = get_illum(K, typ, illum_typ)

        cmd = BASE + [
            "--k", str(K),
            "--phase_shifts", ",".join(str(item) for item in phase_shifts),
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

        print(f"  -> running K={K} capture_type={typ} illum={illum_typ} {illum} \n \n")
        subprocess.run(cmd, check=True)

    #run_id += 1  # new folder per K