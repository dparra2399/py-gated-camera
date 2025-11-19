# Libraries
import os
import glob
import math

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
INT_TIME = 1000  # integration time
NUM_GATES = 3  # number of time bins
IM_WIDTH = 512  # image width
BIT_DEPTH = 12
SIZE = 12
SHIFT = 600  # shift in picoseconds
VOLTAGE = 10
PLOT_CORRELATIONS = True
SAVE_INTO_FILE = True
SMOOTH_SIGMA = 10
SMOOTH_CORRELATIONS = False
EXTENDED = True

# Non-Editable Parameters
ITERATIONS = 1
OVERLAP = 0
TIMEOUT = 0
PILEUP = 0
GATE_STEPS = 1
GATE_STEP_ARBITRARY = 0
GATE_STEP_SIZE = 0
GATE_DIRECTION = 1
GATE_TRIG = 0

SAVE_PATH = '/home/ubi-user/David_P_folder'



if __name__ == "__main__":
    # --- CLI overrides (hybrid approach) ---
    parser = argparse.ArgumentParser(description="Hamiltonian correlaition function capture")
    parser.add_argument("--int_time", type=int, default=INT_TIME)
    parser.add_argument("--num_gates", type=int, default=NUM_GATES)
    parser.add_argument("--im_width", type=int, default=IM_WIDTH)
    parser.add_argument("--bit_depth", type=int, default=BIT_DEPTH)
    parser.add_argument("--shift", type=int, default=SHIFT)
    parser.add_argument("--voltage", type=float, default=VOLTAGE)
    parser.add_argument("--size", type=int, default=SIZE)

    args = parser.parse_args()

    INT_TIME = args.int_time
    NUM_GATES = args.num_gates
    IM_WIDTH = args.im_width
    BIT_DEPTH = args.bit_depth
    SHIFT = args.shift
    VOLTAGE = args.voltage
    SIZE = args.size

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
    FREQ = float(freq[-2]) * 2 if EXTENDED else float(freq[-2])
    MHZ = int(FREQ * 1e-6)


    print('--------------------Parameters---------------')
    print(f'Number of effective bins: {N_TBINS}')
    print(f'Shift: {SHIFT}')
    print(f'Frequency: {MHZ}MHZ')
    print('---------------------------------------------')


    GATE_WIDTH = math.ceil((((1/FREQ)*1e12) // NUM_GATES) * 1e-3 )
    gate_starts = np.array([(GATE_WIDTH * (gate_step)) for gate_step in range(NUM_GATES)]) * 1e3

    (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(N_TBINS, 1. / FREQ)

    print(f'Time bin depth resolution {tbin_depth_res * 1000:.3f} mm')

    correlations = np.zeros((IM_WIDTH, IM_WIDTH, NUM_GATES, N_TBINS))

    for i in range(NUM_GATES):
        gate_start_tmp = gate_starts[i]

        print('-------------------------------------------------------')
        print(f'Starting to measure correlations for gate number {i+1}')
        print('-------------------------------------------------------')

        for j in range(N_TBINS):
            gate_start = gate_start_tmp + j * SHIFT
            #gate_start = gate_start % TAU

            if j == 0:
                print(f'\tGate start: {gate_start}')
                print(f'\tGate width: {GATE_WIDTH}')
            
            counts = np.zeros((IM_WIDTH, IM_WIDTH))
            current_intTime = INT_TIME
            while current_intTime > 480:
                # print(f'starting current time {current_intTime}')
                counts += SPAD1.get_gated_intensity(BIT_DEPTH, 480, ITERATIONS, GATE_STEPS, GATE_STEP_SIZE,
                                                    GATE_STEP_ARBITRARY, GATE_WIDTH,
                                                    gate_start, GATE_DIRECTION, GATE_TRIG, OVERLAP, 1, PILEUP,
                                                    IM_WIDTH)[:, :, 0]
                current_intTime -= 480

            counts += SPAD1.get_gated_intensity(BIT_DEPTH, current_intTime, ITERATIONS, GATE_STEPS, GATE_STEP_SIZE,
                                                GATE_STEP_ARBITRARY, GATE_WIDTH,
                                                gate_start, GATE_DIRECTION, GATE_TRIG, OVERLAP, 1, PILEUP,
                                                IM_WIDTH)[:, :, 0]

            if j % 20 == 0:
                print(f'Measuring gate shift number {j}')

            correlations[:, :, i, j] = counts
        print('-------------------------------------------------------')
        print(f'Finished to measure correlations for gate number {i+1}')
        print('-------------------------------------------------------')

    correlations = np.flip(correlations, axis=-1)

    if EXTENDED:
        correlation_circ = correlations[:, :, :, :(N_TBINS +1) // 2:].copy()
        tail = correlations[:, :, :, (N_TBINS +1) // 2:]
        correlation_circ[:, :, :, :N_TBINS - ((N_TBINS +1)//2)] += tail
        correlations = correlation_circ

    if NUM_GATES == 3 and MHZ == 10:
        VOLTAGE = 7
        SIZE = 34
    elif NUM_GATES == 3 and MHZ == 5:
        VOLTAGE = 5.7
        SIZE = 67
    elif NUM_GATES == 4 and MHZ == 10:
        VOLTAGE = 7.6
        SIZE = 25
    elif NUM_GATES == 4 and MHZ == 5:
        VOLTAGE = 6
        SIZE = 50
    else:
        VOLTAGE = 10
        SIZE = 12

    SAVE_NAME = f'coarsek{NUM_GATES}_{MHZ}mhz_{VOLTAGE}v_{SIZE}w_correlations'
    SAVE_NAME = SAVE_NAME + '_extended' if EXTENDED else SAVE_NAME


    if PLOT_CORRELATIONS:
        irf = get_voltage_function(MHZ, VOLTAGE, SIZE, 'pulse', N_TBINS)
        coding_matrix = get_coarse_coding_matrix(GATE_WIDTH * 1e3, NUM_GATES, 0, GATE_WIDTH * 1e3, rep_tau * 1e12,
                                                 N_TBINS, irf)

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
            gate_steps=GATE_STEPS,
            gate_step_arbitrary=GATE_STEP_ARBITRARY,
            gate_step_size=GATE_STEP_SIZE,
            gate_direction=GATE_DIRECTION,
            gate_trig=GATE_TRIG,
            freq=float(freq[-2]),
            voltage=VOLTAGE,
            size=SIZE,
            gate_width=GATE_WIDTH,
            K=NUM_GATES,
            correlations=correlations,)