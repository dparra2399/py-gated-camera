# Libraries
import os
import glob

from spad_lib.SPAD512S import SPAD512S
from spad_lib.spad512utils import *
from spad_lib.file_utils import *
from plot_scripts.plot_utils import plot_correlation_functions
import numpy as np
import matplotlib.pyplot as plt
import argparse

PORT = 9999  # Check the command Server in the setting tab of the software and change it if necessary
VEX = 7

# Editable parameters (defaults; can be overridden via CLI)
INT_TIME = 4000  # integration time
K = 4  # number of time bins
IM_WIDTH = 512  # image width
BIT_DEPTH = 12
SHIFT = 300  # shift in picoseconds
VOLTAGE = 8.5
DUTY = 20
PLOT_CORRELATIONS = True
SAVE_INTO_FILE = True
SMOOTH_SIGMA = 30
SMOOTH_CORRELATIONS = True

SAVE_PATH = '/home/ubi-user/David_P_folder'

# Non-Editable Parameters
ITERATIONS = 1
OVERLAP = 0
TIMEOUT = 0
PILEUP = 0
GATE_STEP_ARBITRARY = 0
GATE_STEP_SIZE = SHIFT
GATE_DIRECTION = 1
GATE_TRIG = 0


if __name__ == "__main__":
    # --- CLI overrides (hybrid approach) ---
    parser = argparse.ArgumentParser(description="Hamiltonian correlaition function capture")
    parser.add_argument("--int_time", type=int, default=INT_TIME)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--im_width", type=int, default=IM_WIDTH)
    parser.add_argument("--bit_depth", type=int, default=BIT_DEPTH)
    parser.add_argument("--shift", type=int, default=SHIFT)
    parser.add_argument("--voltage", type=float, default=VOLTAGE)
    parser.add_argument("--duty", type=int, default=DUTY)

    args = parser.parse_args()

    INT_TIME = args.int_time
    K = args.k
    IM_WIDTH = args.im_width
    BIT_DEPTH = args.bit_depth
    SHIFT = args.shift
    VOLTAGE = args.voltage
    DUTY = args.duty

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

    TAU = ((1/float(freq[-2])) * 1e12) #Tau in picoseconds
    N_TBINS = int(TAU // SHIFT)
    N_TBINS_EXTENDED = N_TBINS * 3

    MHZ = int(float(freq[-2]) * 1e-6)
    LASER_MHZ = int(MHZ * 2)

    print(f'Laser Repition Rate = {LASER_MHZ}MHZ and Gate Trigger Rate = {MHZ}MHZ')

    if DUTY == 20 and LASER_MHZ == 10:
        VOLTAGE = 8.5
    elif DUTY == 20 and LASER_MHZ == 5:
        VOLTAGE = 6.5
    else:
        VOLTAGE = 10

    SAVE_NAME = f'hamk{K}_{LASER_MHZ}mhz_{VOLTAGE}v_{DUTY}w_correlations'

    print('--------------------Parameters---------------')
    print(f'Number of effective bins: {N_TBINS}')
    print(f'Shift: {SHIFT}')
    print(f'Frequency: {int(float(freq[-2]) * 1e-6)}MHZ')
    print('---------------------------------------------')


    func = getattr(CodingFunctionsFelipe, f"GetHamK{K}")
    (modfs, demodfs) = func(N=N_TBINS, N2=N_TBINS_EXTENDED)
    gated_demodfs_np, gated_demodfs_arr = decompose_ham_codes(demodfs)

    (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(N_TBINS, 1. / float(freq[-2]))

    print(f'Time bin depth resolution {tbin_depth_res * 1000:.3f} mm')

    correlations = np.zeros((IM_WIDTH, IM_WIDTH, K, N_TBINS))

    for i, item in enumerate(gated_demodfs_arr):
            print('-------------------------------------------------------')
            print(f'Starting to measure correlations for demodulation function number {i+1}')
            print('-------------------------------------------------------')
            counts = np.zeros((IM_WIDTH, IM_WIDTH, N_TBINS))
            for k in range(item.shape[-1]):
                gate = item[:, k]
                gate_width, gate_start = get_offset_width_spad512(gate, float(freq[-2]))

                current_intTime = INT_TIME
                while current_intTime > 480:
                    print(f'starting current time {current_intTime}')
                    counts += SPAD1.get_gated_intensity(BIT_DEPTH, 480, ITERATIONS, N_TBINS, GATE_STEP_SIZE,
                                                        GATE_STEP_ARBITRARY, gate_width,
                                                        gate_start, GATE_DIRECTION, GATE_TRIG, OVERLAP, 1, PILEUP, IM_WIDTH)[:, :, 0]
                    current_intTime -= 480

                counts += SPAD1.get_gated_intensity(BIT_DEPTH, current_intTime, ITERATIONS, N_TBINS, GATE_STEP_SIZE,
                                                        GATE_STEP_ARBITRARY, gate_width,
                                                        gate_start, GATE_DIRECTION, GATE_TRIG, OVERLAP, 1, PILEUP, IM_WIDTH)[:, :,  0]

            correlations[:, :, i, :] += counts

            print('-------------------------------------------------------')
            print(f'Finished to measure correlations for demodulation function number {i+1}')
            print('-------------------------------------------------------')

    correlations = np.flip(correlations, axis=-1)

    unit = "ms"
    factor_unit = 1e-3


    if DUTY == 20 and LASER_MHZ == 10:
        VOLTAGE = 8.5
    elif DUTY == 20 and LASER_MHZ == 5:
        VOLTAGE = 6.5
    else:
        VOLTAGE = 10
    # print(mhz)

    if 'pulse' in SAVE_NAME:
        illum_type = 'pulse'
        DUTY = 12
        VOLTAGE = 10
    else:
        illum_type = 'square'
        DUTY = 20


    if PLOT_CORRELATIONS:
        coding_matrix = get_hamiltonain_correlations(K, LASER_MHZ, VOLTAGE, DUTY, illum_type, n_tbins=N_TBINS)

        point_list = [(10, 10), (200, 200), (50, 200)]

        plot_correlation_functions(
            point_list,
            correlations,
            coding_matrix,
            SMOOTH_SIGMA,
            SMOOTH_CORRELATIONS,
        )

    if SAVE_INTO_FILE:
        save_correlation_data(
            save_path=SAVE_PATH,
            save_name=SAVE_NAME,
            total_time=INT_TIME,
            im_width=IM_WIDTH,
            bit_depth=BIT_DEPTH,
            n_tbins=N_TBINS,
            iterations=ITERATIONS,
            overlap=OVERLAP,
            timeout=TIMEOUT,
            pileup=PILEUP,
            gate_steps=N_TBINS,
            gate_step_arbitrary=GATE_STEP_ARBITRARY,
            gate_step_size=GATE_STEP_SIZE,
            gate_direction=GATE_DIRECTION,
            gate_trig=GATE_TRIG,
            freq=float(freq[-2]),
            voltage=VOLTAGE,
            size=DUTY,
            gate_width=None,
            K=K,
            correlations=correlations,)