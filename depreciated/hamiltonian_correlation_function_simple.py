# Libraries

from spad_lib.SPAD512S import SPAD512S
from spad_lib import spad512utils
from utils.file_utils import *
from utils.global_constants import SAVE_PATH_CORRELATIONS
from plot_scripts.plot_utils import plot_correlation_functions
import numpy as np
import argparse
from spad512utils_depreciated import get_hamiltonain_correlations
from utils.tof_utils import calculate_tof_domain_params

PORT = 9999  # Check the command Server in the setting tab of the software and change it if necessary
VEX = 7

# Editable parameters (defaults; can be overridden via CLI)
INT_TIME = 100  # integration time
K = 3  # number of time bins
IM_WIDTH = 512  # image width
BIT_DEPTH = 12
SHIFT = 2500  #50  # shift in picoseconds
VOLTAGE = 8.5
DUTY = 20
PLOT_CORRELATIONS = True
SAVE_INTO_FILE = True
SMOOTH_SIGMA = 30
SMOOTH_CORRELATIONS = False
PULSED = False
EXTENDED = False
MHZ = 5 #*1e6 for repition rate

SAVE_PATH = SAVE_PATH_CORRELATIONS

GATE_SHRINKAGE = 25 #In NS

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


if __name__ == "__main__":
    # --- CLI overrides (hybrid approach) ---
    parser = argparse.ArgumentParser(description="Hamiltonian correlaition function capture")
    parser.add_argument("--int_time", type=int, default=INT_TIME)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--im_width", type=int, default=IM_WIDTH)
    parser.add_argument("--bit_depth", type=int, default=BIT_DEPTH)
    parser.add_argument("--shift", type=int, default=SHIFT)
    parser.add_argument("--voltage", type=float, default=10)
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
    FREQ = float(freq[-2]) * 3 if EXTENDED else float(freq[-2])
    MHZ = int(FREQ * 1e-6)


    print('--------------------Parameters---------------')
    print(f'Number of effective bins: {N_TBINS}')
    print(f'Shift: {SHIFT}')
    print(f'Frequency: {MHZ}MHZ')
    print('---------------------------------------------')


    func = getattr(spad512utils, f"GetHamK{K}_GateShifts")
    ham_gate_widths, ham_gate_starts = func(FREQ)

    #(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = calculate_tof_domain_params(N_TBINS, 1. / FREQ)

    #print(f'Time bin depth resolution {tbin_depth_res * 1000:.3f} mm')

    correlations = np.zeros((IM_WIDTH, IM_WIDTH, K, N_TBINS))

    for i in range(K):

        print('-------------------------------------------------------')
        print(f'Starting to measure correlations for demodulation function number {i+1}')
        print('-------------------------------------------------------')

        gate_widths_tmp = ham_gate_widths[i]
        gate_starts_tmp = ham_gate_starts[i]
        #for j in range(N_TBINS):
        #counts = np.zeros((IM_WIDTH, IM_WIDTH))

        counts = np.zeros((IM_WIDTH, IM_WIDTH, N_TBINS))
        for k in range(len(gate_starts_tmp)):
            gate_width = gate_widths_tmp[k] - GATE_SHRINKAGE
            gate_start_helper = gate_starts_tmp[k]

            #gate_start = gate_start_helper + j * SHIFT
            #gate_start = gate_start % TAU
            gate_start = max(0, gate_starts_tmp[k] + (0 * i))

            #if i == 0:
            print(f'\tGate start: {gate_start}')
            print(f'\tGate width: {gate_width}')

            # current_intTime = INT_TIME
            # while current_intTime > 480:
            #     #print(f'starting current time {current_intTime}')
            #     counts += SPAD1.get_gated_intensity(BIT_DEPTH, 480, ITERATIONS, GATE_STEPS, GATE_STEP_SIZE,
            #                                         GATE_STEP_ARBITRARY, gate_width,
            #                                         gate_start, GATE_DIRECTION, GATE_TRIG, OVERLAP, 1, PILEUP, IM_WIDTH)[:, :, 0]
            #     current_intTime -= 480

            # counts += SPAD1.get_gated_intensity(BIT_DEPTH, current_intTime, ITERATIONS, GATE_STEPS, GATE_STEP_SIZE,
            #                                     GATE_STEP_ARBITRARY, gate_width,
            #                                     gate_start, GATE_DIRECTION, GATE_TRIG, OVERLAP, 1, PILEUP, IM_WIDTH)[:, :,  0]
            
            current_intTime = INT_TIME
            while current_intTime > 480:
                #print(f'starting current time {current_intTime}')
                counts += SPAD1.get_gated_intensity(BIT_DEPTH, 480, ITERATIONS, N_TBINS, SHIFT,
                                                    GATE_STEP_ARBITRARY, gate_width,
                                                    gate_start, GATE_DIRECTION, GATE_TRIG, OVERLAP, 1, PILEUP, IM_WIDTH)
                current_intTime -= 480

            counts += SPAD1.get_gated_intensity(BIT_DEPTH, current_intTime, ITERATIONS, N_TBINS, SHIFT,
                                                GATE_STEP_ARBITRARY, gate_width,
                                                gate_start, GATE_DIRECTION, GATE_TRIG, OVERLAP, 1, PILEUP, IM_WIDTH)

            #if j % 20 == 0:
            #    print(f'Measuring Time Bin: {j}')

        correlations[:, :, i, :] += counts

    print('-------------------------------------------------------')
    print(f'Finished to measure correlations for demodulation function number {i+1}')
    print('-------------------------------------------------------')

    correlations = np.flip(correlations, axis=-1)

    if EXTENDED:
        # chunk size
        base = N_TBINS // 3

        # main chunk (first third)
        correlation_circ = correlations[:, :, :, :base].copy()

        # Fold chunk 1 (second third)
        tail1 = correlations[:, :, :, base:2*base]
        correlation_circ += tail1

        # Fold chunk 2 (third third)
        tail2 = correlations[:, :, :, 2*base:3*base]
        correlation_circ += tail2

        correlations = correlation_circ

    unit = "ms"
    factor_unit = 1e-3


    if DUTY == 20 and MHZ == 10:
        VOLTAGE = 8.5
    elif DUTY == 20 and MHZ == 5:
        VOLTAGE = 6.5
    else:
        VOLTAGE = 10
    # print(mhz)

    if PULSED:
        illum_type = 'pulse'
        #DUTY = 12
        #VOLTAGE = 10
    else:
        illum_type = 'square'
        #DUTY = 20

    SAVE_NAME = f'hamk{K}_{MHZ}mhz_{VOLTAGE}v_{DUTY}w_correlations'

    SAVE_NAME = SAVE_NAME + '_extended' if EXTENDED else SAVE_NAME    
    SAVE_NAME = SAVE_NAME + '_pulsed' if PULSED else SAVE_NAME


    if PLOT_CORRELATIONS:
        coding_matrix = get_hamiltonain_correlations(K, 10, 8.5, 20, 'square', n_tbins=N_TBINS)

        point_list = [(10, 10), (200, 200), (50, 200)]

        plot_correlation_functions(
            point_list,
            correlations,
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
            size=DUTY,
            gate_width=None,
            K=K,
            correlations=correlations,)