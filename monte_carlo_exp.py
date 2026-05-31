# Imports
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from utils.global_constants import *
from utils.tof_utils import (
    get_code,
    simulate_counts,
    simulate_counts_shared_illum,
    decode_simulation_depths,
    calculate_tof_domain_params,
)

# =============================
# Defaults
# =============================
PHOTON_COUNTS = np.linspace(300, 3000, 10).tolist()
SBRS = np.linspace(0.1, 10, 10).tolist()
TRIALS = 100
N_TBINS = 999
SMOOTH_SIGMA = 1
SHIFT = None
DEPTH_MARGIN = 3.0
REP_RATE = 5 * 1e6
DEPTH_SAMPLE = 0.01
SPLIT_ACQUISITION = True  # if True, scale photon count by acquisition splits per capture type/k
DEFAULT_RUNS = [
    {'type': 'ham',        'k': 3},
    {'type': 'coarse',     'k': 3},
    # {'type': 'trapcoarse', 'k': 3},
    # {'type': 'rect',       'k': 3},
    # {'type': 'traprect',   'k': 3},
]




def scale_photon_count(photon_count, capture_type, k):
    if capture_type == 'ham':
        divisor = k if k <= 3 else 6
    else:
        divisor = k
    return photon_count / divisor


def run_one(idx, cap_type, k, photon_count_base, sbr, x, y,
            trials, n_tbins, tbin_depth_res, depth_margin, depth_sample):
    """Single (photon_count, sbr) evaluation — runs in a worker process.
    All large arrays are recomputed locally to avoid pickling overhead / OOM."""
    import numpy as np
    from utils.tof_utils import (get_code, simulate_counts,
                                  simulate_counts_shared_illum,
                                  decode_simulation_depths)

    depths = np.arange(depth_margin, (n_tbins * tbin_depth_res) - depth_margin, depth_sample)
    waveform_or_illum, demodfs_or_cm, coding_matrix = get_code(cap_type, k, n_tbins)

    photon_count = scale_photon_count(photon_count_base, cap_type, k) if SPLIT_ACQUISITION else photon_count_base

    if cap_type == 'ham':
        _, coded_values = simulate_counts(
            waveform=waveform_or_illum,
            demodfs=demodfs_or_cm,
            depths=depths,
            photon_count=photon_count,
            sbr=sbr,
            tbin_depth_res=tbin_depth_res,
            n_tbins=n_tbins,
            k=k,
        )
    else:
        _, coded_values = simulate_counts_shared_illum(
            illum=waveform_or_illum,
            coding_matrix=demodfs_or_cm,
            depths=depths,
            photon_count=photon_count,
            sbr=sbr,
            tbin_depth_res=tbin_depth_res,
            n_tbins=n_tbins,
            k=k,
        )

    _, rmse, mae = decode_simulation_depths(
        coding_matrix=coding_matrix,
        coded_values=coded_values,
        depths=depths,
        trials=trials,
        tbin_depth_res=tbin_depth_res,
    )

    return idx, x, y, mae * 1000, rmse * 1000


def parse_args():
    p = argparse.ArgumentParser(description="Monte Carlo simulation sweep over photon counts and SBRs")
    p.add_argument("--photon_counts", type=float, nargs="+", default=PHOTON_COUNTS)
    p.add_argument("--sbrs",          type=float, nargs="+", default=SBRS)
    p.add_argument("--trials",        type=int,              default=TRIALS)
    p.add_argument("--n_tbins",       type=int,              default=N_TBINS)
    p.add_argument("--rep_rate",      type=float,            default=REP_RATE)
    p.add_argument("--depth_sample",  type=float,            default=DEPTH_SAMPLE)
    p.add_argument("--depth_margin",  type=float,            default=DEPTH_MARGIN)
    return p.parse_args()


# =============================
# Main
# =============================
if __name__ == "__main__":
    args = parse_args()

    if args.rep_rate is not None:
        (rep_tau, rep_freq, tbin_res,
         t_domain, max_depth, tbin_depth_res,) = calculate_tof_domain_params(args.n_tbins, 1. / args.rep_rate)
    else:
        tbin_depth_res = 1.0
    mae_results  = np.zeros((len(DEFAULT_RUNS), len(args.photon_counts), len(args.sbrs)))
    rmse_results = np.zeros((len(DEFAULT_RUNS), len(args.photon_counts), len(args.sbrs)))

    # build flat task list — only scalars passed to each worker, arrays recomputed inside
    tasks = [
        delayed(run_one)(
            idx, r['type'], r['k'],
            photon_count_base, sbr, x, y,
            args.trials, args.n_tbins, tbin_depth_res,
            args.depth_margin, args.depth_sample,
        )
        for idx, r in enumerate(DEFAULT_RUNS)
        for x, photon_count_base in enumerate(args.photon_counts)
        for y, sbr in enumerate(args.sbrs)
    ]

    print(f"Running {len(tasks)} tasks in parallel...")
    results = Parallel(n_jobs=10, verbose=10)(tasks)

    for idx, x, y, mae, rmse in results:
        mae_results[idx, x, y]  = mae
        rmse_results[idx, x, y] = rmse

    save_dir = "/Users/davidparra/PycharmProjects/py-gated-camera/data/monte_carlo_exp"
    os.makedirs(save_dir, exist_ok=True)

    run_labels = [f"{r['type']}_k{r['k']}" for r in DEFAULT_RUNS]
    photon_min = int(min(args.photon_counts))
    photon_max = int(max(args.photon_counts))
    sbr_min    = f"{min(args.sbrs):.1f}"
    sbr_max    = f"{max(args.sbrs):.1f}"
    runs_str   = "_".join(run_labels)
    filename   = f"ntbins{args.n_tbins}_trials{args.trials}_photons{photon_min}-{photon_max}_sbr{sbr_min}-{sbr_max}_{runs_str}.npz"
    save_path  = os.path.join(save_dir, filename)
    np.savez(
        save_path,
        mae_results=mae_results,
        rmse_results=rmse_results,
        photon_counts=np.array(args.photon_counts),
        sbrs=np.array(args.sbrs),
        run_labels=np.array(run_labels),
        split_acquisition=SPLIT_ACQUISITION,
    )
    print(f"\nSaved to {save_path}")
