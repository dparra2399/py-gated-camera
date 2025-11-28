# Libraries
import os

from spad_lib.SPAD512S import SPAD512S
from spad_lib.spad512utils import *
from spad_lib import spad512utils
from spad_lib.file_utils import *
import numpy as np
from felipe_utils import CodingFunctionsFelipe
import argparse
from plot_scripts.plot_utils import plot_gated_images

PORT = 9999  # Check the command Server in the setting tab of the software and change it if necessary
VEX = 7

#Do not edit these parameter
ITERATIONS = 1
OVERLAP = 0
TIMEOUT = 0
PILEUP = 0
GATE_STEPS = 1
GATE_STEP_ARBITRARY = 0
GATE_STEP_SIZE = 0
GATE_DIRECTION = 1
GATE_TRIG = 0

# Editable parameters (defaults; can be overridden via CLI)
TOTAL_TIME = 500# integration time
SPLIT_MEASUREMENTS = False
IM_WIDTH = 512  # image width
BIT_DEPTH = 12
K = 3
N_TBINS = 640
CORRECT_MASTER = False
DECODE_DEPTHS = True
SAVE_INTO_FILE = True
USE_CORRELATIONS = False
USE_FULL_CORRELATIONS = False
SIGMA_SIZE = 1
SHIFT_SIZE = 150
MEDIAN_FILTER_SIZE = 11
GROUND_TRUTH = False
PULSE = False
True
DUTY = 20
VMIN = 5
VMAX = 6

EXP_NUM = 0
SAVE_PATH = '/home/ubi-user/David_P_folder'
# save_path = '/mnt/researchdrive/research_users/David/gated_project_data'

if GROUND_TRUTH:
    SAVE_NAME = f'hamk{K}_gt_exp{EXP_NUM}'
else:
    SAVE_NAME = f'hamk{K}_exp{EXP_NUM}'

SAVE_NAME = SAVE_NAME if PULSE==False else SAVE_NAME + '_pulse'

if __name__=='__main__':
    # --- CLI overrides (hybrid approach): defaults above, CLI can override ---
    parser = argparse.ArgumentParser(description="Gated SPAD capture with Hamiltonian codes")
    parser.add_argument("--total_time", type=int, default=TOTAL_TIME)
    parser.add_argument("--split_measurements", action=argparse.BooleanOptionalAction, default=SPLIT_MEASUREMENTS)
    parser.add_argument("--im_width", type=int, default=IM_WIDTH)
    parser.add_argument("--bit_depth", type=int, default=BIT_DEPTH)
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--n_tbins", type=int, default=N_TBINS)
    parser.add_argument("--correct_master", action=argparse.BooleanOptionalAction, default=CORRECT_MASTER)
    parser.add_argument("--decode_depths", action=argparse.BooleanOptionalAction, default=DECODE_DEPTHS)
    parser.add_argument("--save_into_file", action=argparse.BooleanOptionalAction, default=SAVE_INTO_FILE)
    parser.add_argument("--use_correlations", action=argparse.BooleanOptionalAction, default=USE_CORRELATIONS)

    parser.add_argument("--duty", type=int, default=DUTY)
    parser.add_argument("--vmin", type=float, default=VMIN)
    parser.add_argument("--vmax", type=float, default=VMAX)

    parser.add_argument("--exp_num", type=int, default=EXP_NUM)
    parser.add_argument("--save_path", type=str, default=SAVE_PATH)
    parser.add_argument("--save_name", type=str, default=SAVE_NAME)

    args = parser.parse_args()

    # Apply overrides back to the module-level names (session constants)
    TOTAL_TIME = args.total_time
    SPLIT_MEASUREMENTS = args.split_measurements
    IM_WIDTH = args.im_width
    BIT_DEPTH = args.bit_depth
    K = args.K
    N_TBINS = args.n_tbins
    CORRECT_MASTER = args.correct_master
    DECODE_DEPTHS = args.decode_depths
    SAVE_INTO_FILE = args.save_into_file
    USE_CORRELATIONS = args.use_correlations

    DUTY = args.duty
    VMIN = args.vmin
    VMAX = args.vmax

    EXP_NUM = args.exp_num
    SAVE_PATH = args.save_path
    SAVE_NAME = args.save_name

    SPAD1 = SPAD512S(PORT)
    # Get informations on the system used
    info = SPAD1.get_info()
    print("\nGeneral informations of the camera :")
    print(info)
    temp = SPAD1.get_temps()  # Current temperatures of FPGAs, PCB and Chip
    print("\nCurrent temperatures of FPGAs, PCB and Chip :")
    print(temp)
    freq = SPAD1.get_freq()  # Operating frequencies (Laser and frame)
    print("\nOperating frequencies (Laser and frame) :")
    print(freq)

    # # # Set the voltage to the maximum value
    SPAD1.set_Vex(VEX)

    #Get demodulation functions and split for use with Gated SPAD
    func = getattr(spad512utils, f"GetHamK{K}_GateShifts")
    ham_gate_widths, ham_gate_starts = func(float(freq[-2]))

    coded_vals = np.zeros((IM_WIDTH, IM_WIDTH, K))
    for i in range(K):
        gate_widths_tmp = ham_gate_widths[i]
        gate_starts_tmp = ham_gate_starts[i]
        counts = np.zeros((IM_WIDTH, IM_WIDTH))
        for k in range(len(gate_starts_tmp)):
            gate_width = gate_widths_tmp[k]
            gate_start = gate_starts_tmp[k]

            if SPLIT_MEASUREMENTS:
                intTime = int(TOTAL_TIME // sum(len(sublist) for sublist in gate_starts_tmp))
            else:
                intTime = TOTAL_TIME
            
            print(f'Integration time for hamiltonian row #{i+1}.{k+1}: {intTime}')
            print(f'gate width = {gate_width}, gate offset = {gate_start}')

            current_intTime = intTime
            while current_intTime > 480:
                print(f'starting current time {current_intTime}')
                counts += SPAD1.get_gated_intensity(BIT_DEPTH, 480, ITERATIONS, GATE_STEPS, GATE_STEP_SIZE,
                                                    GATE_STEP_ARBITRARY, gate_width,
                                                    gate_start, GATE_DIRECTION, GATE_TRIG, OVERLAP, 1, PILEUP, IM_WIDTH)[:, :, 0]
                current_intTime -= 480

            counts += SPAD1.get_gated_intensity(BIT_DEPTH, current_intTime, ITERATIONS, GATE_STEPS, GATE_STEP_SIZE,
                                                GATE_STEP_ARBITRARY, gate_width,
                                                gate_start, GATE_DIRECTION, GATE_TRIG, OVERLAP, 1, PILEUP, IM_WIDTH)[:, :, 0]

        coded_vals[:, :, i] = counts

    print(coded_vals.shape)
    unit = "ms"
    factor_unit = 1e-3

    mhz = int(float(freq[-2]) * 1e-6)
    if DUTY == 20 and mhz == 10:
        VOLTAGE = 8.5
    elif DUTY == 20 and mhz == 5:
        VOLTAGE = 6.5
    else:
        VOLTAGE = 10

    if 'pulse' in SAVE_NAME:
        illum_type = 'pulse'
    else:
        illum_type = 'square'

    if DECODE_DEPTHS:
        (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(N_TBINS, 1./ float(freq[-2]))

        if USE_CORRELATIONS:
            corr_path = (
                f"/home/ubi-user/David_P_folder/py-gated-camera/correlation_functions/"
                f"hamk{coded_vals.shape[-1]}_{mhz}mhz_{VOLTAGE}v_{DUTY}w_correlations.npz"
            )
            correlations_total, n_tbins_corr = load_correlations_file(corr_path)
            (
                rep_tau,
                rep_freq,
                tbin_res,
                t_domain,
                max_depth,
                tbin_depth_res,
            ) = calculate_tof_domain_params(n_tbins_corr, 1.0 / float(freq[-2]))
            coding_matrix = build_coding_matrix_from_correlations(correlations_total, IM_WIDTH, n_tbins_corr, freq,
                                                                  USE_FULL_CORRELATIONS, SIGMA_SIZE, SHIFT_SIZE)
            N_TBINS = n_tbins_corr

        else:
            coding_matrix = get_hamiltonain_correlations(
                coded_vals.shape[-1], mhz, VOLTAGE, DUTY, illum_type, n_tbins=N_TBINS
            )
        depth_map, zncc = decode_depth_map(
            coded_vals,
            coding_matrix,
            IM_WIDTH,
            N_TBINS,
            tbin_depth_res,
            USE_CORRELATIONS,
            USE_FULL_CORRELATIONS,
        )

        if CORRECT_MASTER is False:
            depth_map_plot= depth_map[:, :IM_WIDTH//2]
        else:
            depth_map_plot = np.copy(depth_map)

        plot_gated_images(
            coded_vals,
            depth_map_plot,
            None,
            vmin=VMIN,
            vmax=VMAX,
            median_filter_size=MEDIAN_FILTER_SIZE,
        )

    if SAVE_INTO_FILE:
        save_capture_data(
            save_path=SAVE_PATH,
            save_name=SAVE_NAME,
            total_time=TOTAL_TIME,
            im_width=IM_WIDTH,
            bit_depth=BIT_DEPTH,
            n_tbins=N_TBINS,
            iterations=ITERATIONS,
            overlap=OVERLAP,
            timeout=TIMEOUT,
            pileup=PILEUP,
            gate_steps=GATE_STEPS,
            gate_step_arbitrary=GATE_STEP_ARBITRARY,
            gate_step_size=GATE_STEP_SIZE,
            gate_direction=GATE_DIRECTION,
            gate_trig=GATE_TRIG,
            freq=float(freq[-2]),
            voltage=VOLTAGE,
            coded_vals=coded_vals,
            split_measurements=SPLIT_MEASUREMENTS,
            size=DUTY,
            gate_width=None,
            K=K
        )