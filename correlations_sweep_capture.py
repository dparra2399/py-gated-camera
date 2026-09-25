# sweep.py
import subprocess

from illum_config import get_illum, ILLUM_CONFIG

SCRIPT = "correlations_single_capture.py"

K_VALUES = [16]  # <-- sweep these; illum combos are resolved from illum_config per K

REP_RATE = 10000000

BASE = [
    "python", SCRIPT,
    "--im_width", "128",
    "--burst_time", "0.05",
    "--bit_depth", "12",
    "--int_time", "2",
    "--split_acquisition", "1",
    "--gate_step_size", "1200",
    "--rep_rate", str(REP_RATE),
    "--plot_correlations", "false",
    "--save_into_file", "true",
    "--timeout", "0",
    "--current", "16",
    "--low_level_amplitude", "-0.5",

]

# Capture types to run. The illum_type for each is looked up from illum_config
# per K, so a type is skipped wherever no row is configured for that K.
CAPTURE_TYPES = ["coarse"]  # ["ham", "coarse", "trapcoarse", "sliding"]

# sliding only: gate width in ns (k sets the shift, the width is independent)
SLIDING_GATE_WIDTH = 50


def runs_for_k(k):
    """(capture_type, illum_type) combos configured for this k, in CAPTURE_TYPES order."""
    combos = []
    for typ in CAPTURE_TYPES:
        # kk is None for schemes whose illumination does not depend on k (sliding)
        for illum_typ in [it for (kk, ct, it) in ILLUM_CONFIG if kk in (k, None) and ct == typ]:
            combos.append((typ, illum_typ))
    return combos


# OUTER LOOP = K, INNER LOOP = capture types configured for that K
for K in K_VALUES:
    for typ, illum_typ in runs_for_k(K):

        illum = get_illum(K, typ, illum_typ)

        cmd = BASE + [
            "--k", str(K),
            "--capture_type", typ,
            "--gate_shrinkage", str(illum["gate_shrinkage"]),
            "--duty", str(illum["duty"]),
            "--illum_type", illum_typ,
            "--high_level_amplitude", str(illum["high_level_amplitude"]),
        ]
        if typ == "sliding":
            cmd += ["--sliding_gate_width", str(SLIDING_GATE_WIDTH)]

        print("==============================================================")
        print(f"K={K}  type={typ}  illum={illum_typ}  amp={illum['high_level_amplitude']}")
        print("==============================================================")

        subprocess.run(cmd, check=True)
