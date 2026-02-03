import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import zipfile

from felipe_utils import tof_utils_felipe
from utils.global_constants import *
from utils.file_utils import get_data_folder,  make_correlation_filename
from utils.tof_utils import (
    build_coding_matrix_from_correlations,
    decode_from_correlations,
    get_simulated_coding_matrix,
)

# -----------------------------
# Defaults (PyCharm-friendly)
# -----------------------------
PHOTON_COUNT = 3000
SBR = 1.0
TRIALS = 100
N_TBINS = 1998
SMOOTH_SIGMA = None
SHIFT = None
DEPTH_MARGIN = 200
SIMULATED_CORRELATIONS = False

#Capture types to test
DEFAULT_CAPTURE_TYPES = ["ham", "coarse"]
DEFAULT_MA = list( np.arange(50, 80, 10)) #[50, 60, 75, 100]
DEFAULT_MV = list(np.arange(0.5, 5.1, 1.0) * 1000) # [100, 250, 500, 1000]  # whatever you used in filenames
DEFAULT_DUTY_LIST = [20, 32]
DEFAULT_FREQ_MHZ = 5
DEFAULT_K = 3

PLOT_RESULTS = True


def parse_args():
    p = argparse.ArgumentParser(description="Sweep mA/mV correlation files and decode depths")

    # sweep axes
    p.add_argument("--capture_types", nargs="+", default=DEFAULT_CAPTURE_TYPES)
    p.add_argument("--mA", nargs="+", type=float, default=DEFAULT_MA)
    p.add_argument("--mV", nargs="+", type=float, default=DEFAULT_MV)

    # fixed params used to build filenames
    p.add_argument("--duty_list", nargs="+", type=float, default=DEFAULT_DUTY_LIST)
    p.add_argument("--freq_mhz", type=float, default=DEFAULT_FREQ_MHZ)
    p.add_argument("--k", type=int, default=DEFAULT_K)

    # decode params
    p.add_argument("--photon_count", type=int, default=PHOTON_COUNT)
    p.add_argument("--sbr", type=float, default=SBR)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--n_tbins", type=int, default=N_TBINS)
    p.add_argument("--smooth_sigma", type=float, default=SMOOTH_SIGMA)
    p.add_argument("--shift", type=int, default=SHIFT)
    p.add_argument("--depth_margin", type=int, default=DEPTH_MARGIN)
    p.add_argument("--simulated_correlations", action="store_true", default=SIMULATED_CORRELATIONS)

    # output
    p.add_argument("--out", type=str, default="sweep_decode_results.npz")
    p.add_argument("--plot_results", action="store_true", default=PLOT_RESULTS)

    return p.parse_args()



if __name__ == "__main__":
    args = parse_args()

    ##Check is duty list and capture types is the same size list.
    if len(args.duty_list) != len(args.capture_types):
        raise ValueError(
            f"--duty_list must match --capture_types length. "
            f"Got duty_list={len(args.duty_list)} capture_types={len(args.capture_types)}"
        )

    folder = get_data_folder(READ_PATH_CORRELATIONS_MAC, READ_PATH_CORRELATIONS_WINDOWS)
    depths = np.arange(args.depth_margin, args.n_tbins - args.depth_margin, 1)

    # We will store results per run
    run_rows = []          # list of dicts describing the run
    decoded_list = []      # decoded_depths arrays (trials x len(depths)) typically
    rmse_list = []
    mae_list = []

    total = len(args.capture_types) * len(args.mA) * len(args.mV)
    run_id = 0

    for cap, duty in zip(args.capture_types, args.duty_list):
        for mA in args.mA:
            for mV in args.mV:
                run_id += 1
                filename = make_correlation_filename(cap, args.k, args.freq_mhz, mV, mA, duty)
                path = os.path.join(folder, filename)

                print(f"[{run_id:03d}/{total}] type={cap:6s}  mA={mA:6.1f}  mV={mV:7.1f}  -> {filename}")

                if not os.path.exists(path):
                    print(f"  !! missing file: {path}")
                    continue

                file = np.load(path, allow_pickle=True)
                cfg = file["cfg"].item()

                try:
                    correlations_total = file["correlations"]
                except zipfile.BadZipFile:
                    continue

                name = cfg.get("capture_type", cap)

                # rep_tau handling (your cfg sometimes has rep_tau, sometimes rep_rate)
                rep_tau = cfg.get("rep_tau", None)
                if rep_tau is None and "rep_rate" in cfg:
                    rep_tau = 1.0 / float(cfg["rep_rate"])
                if rep_tau is None:
                    raise KeyError(f"Missing rep_tau/rep_rate in cfg for {filename}")

                # coding matrix
                if args.simulated_correlations:
                    coding_matrix = get_simulated_coding_matrix(name, args.n_tbins, cfg["k"])
                else:
                    coding_matrix = build_coding_matrix_from_correlations(
                        correlations_total,
                        False,
                        args.smooth_sigma,
                        args.shift,
                        args.n_tbins,
                    )

                coding_matrix /= np.max(np.abs(coding_matrix), axis=0, keepdims=True)

                decoded_depths = decode_from_correlations(
                    coding_matrix=coding_matrix,
                    depths=depths,
                    photon_count=args.photon_count,
                    sbr=args.sbr,
                    trials=args.trials,
                )

                rmse = float(np.sqrt(np.mean((decoded_depths - depths) ** 2)))
                mae = float(np.mean(np.abs(decoded_depths - depths)))

                print(f"  -> MAE={mae:.3f}  RMSE={rmse:.3f}")

                run_rows.append(
                    dict(
                        filename=filename,
                        capture_type=cap,
                        k=args.k,
                        freq_mhz=args.freq_mhz,
                        mV=mV,
                        mA=mA,
                        duty=duty,
                        mae=mae,
                        rmse=rmse,
                    )
                )
                decoded_list.append(decoded_depths)
                rmse_list.append(rmse)
                mae_list.append(mae)

    if len(run_rows) == 0:
        raise RuntimeError("No runs were successfully loaded/decoded. Check paths/filenames.")

    # Stack decoded depths
    # decoded_depths shape depends on your decode function; commonly (trials, len(depths))
    decoded_stack = np.stack(decoded_list, axis=0)

    if args.plot_results:
        mA_vals = np.array(args.mA, dtype=float)
        mV_vals = np.array(args.mV, dtype=float)

        # maps value -> index (avoids fragile == searches inside loops)
        mA_idx = {v: i for i, v in enumerate(mA_vals)}
        mV_idx = {v: j for j, v in enumerate(mV_vals)}

        X, Y = np.meshgrid(mV_vals, mA_vals)  # X: mV, Y: mA

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        for cap in args.capture_types:
            Z = np.full((len(mA_vals), len(mV_vals)), np.nan, dtype=float)

            for r in run_rows:
                if r["capture_type"] != cap:
                    continue

                a = float(r["mA"])
                v = float(r["mV"])
                rmse = float(r["rmse"])

                # if you have float weirdness, round them here consistently
                # a = round(a, 6); v = round(v, 6)

                if a in mA_idx and v in mV_idx:
                    if rmse > 50:
                        continue
                    Z[mA_idx[a], mV_idx[v]] = rmse

            surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, label=cap)

        ax.set_title(f"RMSE Monte Carlo Plot")
        ax.set_xlabel("mV")
        ax.set_ylabel("mA")
        ax.set_zlabel("RMSE")
        ax.set_zlim(0, 50)
        ax.legend()
        #fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="RMSE")
        plt.tight_layout()
        plt.show()



    # Save a compact results file
    # np.savez(
    #     args.out,
    #     runs=np.array(run_rows, dtype=object),
    #     depths=depths,
    #     decoded_depths=decoded_stack,
    #     rmse=np.array(rmse_list, dtype=float),
    #     mae=np.array(mae_list, dtype=float),
    # )

    print(f"\n✅ Saved sweep decode results to: {args.out}")
    print(f"   decoded_depths shape = {decoded_stack.shape}  (runs, trials, depths)")
