# Libraries
import os
from spad_lib.SPAD512S import SPAD512S
from spad_lib.spad512utils import *
from spad_lib.file_utils import *
from plot_scripts.plot_utils import *
import numpy as np
import math
import argparse

PORT = 9999  # Check the command Server in the setting tab of the software and change it if necessary
VEX = 7

# Editable parameters (defaults; can be overridden via CLI)
TOTAL_TIME = 100  # integration time
SPLIT_MEASUREMENTS = False
NUM_GATES = 3  # number of time bins
IM_WIDTH = 512  # image width
BIT_DEPTH = 12
N_TBINS = 640
CORRECT_MASTER = False
DECODE_DEPTHS = True
SAVE_INTO_FILE = True
USE_CORRELATIONS = True
USE_FULL_CORRELATIONS = False
SIGMA_SIZE = 1
SHIFT_SIZE = 150
MEDIAN_FILTER_SIZE = 3
GROUND_TRUTH = False

VMIN = 7
VMAX = 7.5

#Non-Editable parameters
ITERATIONS = 1
OVERLAP = 0
TIMEOUT = 0
PILEUP = 0
GATE_STEPS = 1
GATE_STEP_ARBITRARY = 0
GATE_STEP_SIZE = 0
GATE_DIRECTION = 1
GATE_TRIG = 0

EXP_NUM = 2
# SAVE_PATH = '/mnt/researchdrive/research_users/David/gated_project_data'
SAVE_PATH = '/home/ubi-user/David_P_folder'

if GROUND_TRUTH:
    SAVE_NAME = f'coarsek{NUM_GATES}_gt_exp{EXP_NUM}'
else:
    SAVE_NAME = f'coarsek{NUM_GATES}_exp{EXP_NUM}'

if __name__ == '__main__':
    # --- Hardware constants and initialization ---
    SPAD1 = SPAD512S(PORT)

    # Get informations on the system used
    info = SPAD1.get_info()
    print("\nGeneral informations of the camera :")
    print(info)
    temp = SPAD1.get_temps() # Current temperatures of FPGAs, PCB and Chip
    print("\nCurrent temperatures of FPGAs, PCB and Chip :")
    print(temp)
    freq = SPAD1.get_freq() # Operating frequencies (Laser and frame)
    print("\nOperating frequencies (Laser and frame) :")
    print(freq)

    # # # Set the voltage to the maximum value
    SPAD1.set_Vex(VEX)

    # --- CLI overrides (hybrid approach) ---
    parser = argparse.ArgumentParser(description="Split-pulsed gated capture")
    parser.add_argument("--total_time", type=int, default=TOTAL_TIME)
    parser.add_argument("--split_measurements", action=argparse.BooleanOptionalAction, default=SPLIT_MEASUREMENTS)
    parser.add_argument("--num_gates", type=int, default=NUM_GATES)
    parser.add_argument("--im_width", type=int, default=IM_WIDTH)
    parser.add_argument("--bit_depth", type=int, default=BIT_DEPTH)
    parser.add_argument("--n_tbins", type=int, default=N_TBINS)
    parser.add_argument("--correct_master", action=argparse.BooleanOptionalAction, default=CORRECT_MASTER)
    parser.add_argument("--decode_depths", action=argparse.BooleanOptionalAction, default=DECODE_DEPTHS)
    parser.add_argument("--save_into_file", action=argparse.BooleanOptionalAction, default=SAVE_INTO_FILE)
    parser.add_argument("--use_correlations", action=argparse.BooleanOptionalAction, default=USE_CORRELATIONS)
    parser.add_argument("--vmin", type=float, default=VMIN)
    parser.add_argument("--vmax", type=float, default=VMAX)
    parser.add_argument("--exp_num", type=int, default=EXP_NUM)
    parser.add_argument("--save_path", type=str, default=SAVE_PATH)
    parser.add_argument("--save_name", type=str, default=SAVE_NAME)

    args = parser.parse_args()

    TOTAL_TIME = args.total_time
    SPLIT_MEASUREMENTS = args.split_measurements
    NUM_GATES = args.num_gates
    IM_WIDTH = args.im_width
    BIT_DEPTH = args.bit_depth
    N_TBINS = args.n_tbins
    CORRECT_MASTER = args.correct_master
    DECODE_DEPTHS = args.decode_depths
    SAVE_INTO_FILE = args.save_into_file
    USE_CORRELATIONS = args.use_correlations
    VMIN = args.vmin
    VMAX = args.vmax
    EXP_NUM = args.exp_num
    SAVE_PATH = args.save_path
    SAVE_NAME = args.save_name

    #Make list of gate starts which will be the offet param in the SPAD512
    GATE_WIDTH = math.ceil((((1/float(freq[-2]))*1e12) // NUM_GATES) * 1e-3 )
    gate_starts = np.array([(GATE_WIDTH * (gate_step)) for gate_step in range(NUM_GATES)]) * 1e3

    print("\nGate Starts (offsets):")
    print(gate_starts)
    print(f'\nnum gates: {NUM_GATES}')
    print(f'\ngate width: {GATE_WIDTH}')

    #For each gate make a gated acq. using the offset provided above
    coded_vals = np.zeros((IM_WIDTH, IM_WIDTH, NUM_GATES))
    for i in range(NUM_GATES):
        gate_offset = gate_starts[i]

        if SPLIT_MEASUREMENTS:
            intTime = int(TOTAL_TIME // NUM_GATES)
        else:
            intTime = TOTAL_TIME

        print(f'Integration time for gate #{i+1}: {intTime}')

        current_intTime = intTime
        counts = np.zeros((IM_WIDTH, IM_WIDTH))
        while current_intTime > 480:
            print(f'starting current time {current_intTime}')
            counts += SPAD1.get_gated_intensity(BIT_DEPTH, 480, ITERATIONS, GATE_STEPS, GATE_STEP_SIZE,
                                                GATE_STEP_ARBITRARY, GATE_WIDTH,
                                                gate_offset, GATE_DIRECTION, GATE_TRIG, OVERLAP, 1, PILEUP, IM_WIDTH)[:, :, 0]
            current_intTime -= 480

        counts += SPAD1.get_gated_intensity(BIT_DEPTH, current_intTime, ITERATIONS, GATE_STEPS, GATE_STEP_SIZE,
                                            GATE_STEP_ARBITRARY, GATE_WIDTH,
                                            gate_offset, GATE_DIRECTION, GATE_TRIG, OVERLAP, 1, PILEUP, IM_WIDTH)[:, :, 0]

        coded_vals[:, :, i] = counts

    print(coded_vals.shape)
    unit = "ms"
    factor_unit = 1e-3

    mhz = int(float(freq[-2]) * 1e-6)
    if NUM_GATES == 3 and mhz == 10:
        VOLTAGE = 7
        SIZE = 34
    elif NUM_GATES == 3 and mhz == 5:
        VOLTAGE = 5.7
        SIZE = 67
    elif NUM_GATES == 4 and mhz == 10:
        VOLTAGE = 7.6
        SIZE = 25
    elif NUM_GATES == 4 and mhz == 5:
        VOLTAGE = 6
        SIZE = 50
    else:
        VOLTAGE = 10
        SIZE = 12

    if DECODE_DEPTHS:
        (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(N_TBINS,
                                                                                                         1. / float(
                                                                                                             freq[-2]))

        if USE_CORRELATIONS:
            corr_path = (
                f"/home/ubi-user/David_P_folder/py-gated-camera/correlation_functions/"
                f"coarsek{coded_vals.shape[-1]}_{mhz}mhz_{VOLTAGE}v_{SIZE}w_correlations.npz"
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
            irf = get_voltage_function(mhz, VOLTAGE, SIZE, "pulse", N_TBINS)
            coding_matrix = get_coarse_coding_matrix(
                GATE_WIDTH * 1e3,
                coded_vals.shape[-1],
                0,
                GATE_WIDTH * 1e3,
                rep_tau * 1e12,
                N_TBINS,
                irf,
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
            depth_map_plot = depth_map[:, :IM_WIDTH // 2]
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
            size=SIZE,
            gate_width=GATE_WIDTH,
            K=NUM_GATES
        )