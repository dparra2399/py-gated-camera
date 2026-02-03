import numpy as np
import math
from spad_lib.SPAD512S import SPAD512S
from utils.global_constants import VEX, PORT
import sys



def intrinsics_from_fov(W, H, fov_x_deg, fov_y_deg):
    fx = (W/2) / np.tan(np.deg2rad(fov_x_deg/2))
    fy = (H/2) / np.tan(np.deg2rad(fov_y_deg/2))
    cx, cy = W/2, H/2
    return fx, fy, cx, cy

def intrinsics_from_sensor(W, H, f_mm, sensor_w_mm, sensor_h_mm):
    fx = f_mm * W / sensor_w_mm
    fy = f_mm * H / sensor_h_mm
    cx, cy = W/2, H/2
    return fx, fy, cx, cy

def intrinsics_from_pixel_pitch(W, H, f_mm, pitch_um):
    p_mm = pitch_um * 1e-3
    fx = fy = f_mm / p_mm
    cx, cy = W/2, H/2
    return fx, fy, cx, cy


def get_hamk3_gate_shifts(freq, k=3):
    assert k == 3
    tau = float(1 / freq) #repition tau
    demodduty = 1./2.
    shifts = [0, (1. / 3.), (2. / 3.)]
    gate_widths = [[], [], []]
    gate_starts = [[], [], []]
    for i in range(k):
        gate_widths[i].append(math.ceil((demodduty * tau * 1e9)))
        gate_starts[i].append(math.ceil((shifts[i] * tau * 1e12)))
    return gate_widths, gate_starts


def get_hamk4_gate_shifts(freq, k=4):
    assert k == 4
    tau = float(1 / freq) #repition tau
    demodduty1 = np.array([6./12.,6./12.])
    shift1 = 5./12.
    demodduty2 = np.array([6./12.,6./12.])
    shift2 = 2./12.
    demodduty3 = np.array([3./12.,4./12.,3./12.,2./12.])
    shift3 = 0./12.
    demodduty4 = np.array([2./12.,3./12,4./12.,3./12.])
    shift4 = 4./12.
    gate_starts = [[], [], [], []]
    gate_widths = [[], [], [], []]
    demoddutys = [demodduty1, demodduty2, demodduty3, demodduty4]
    shifts = [shift1, shift2, shift3, shift4]
    for i in range(0,k):
        demodduty = demoddutys[i]
        #startindeces = np.floor((np.cumsum(demodduty) - demodduty)*n)
        gate_start = (np.cumsum(demodduty) - demodduty)
        print(np.cumsum(demodduty) - demodduty)
        #endindeces = startindeces + np.floor(demodduty*n) - 1
        for j in range(len(demodduty)):
            if((j%2) == 0):
                shift = math.ceil(shifts[i] * tau * 1e12)
                gate_starts[i].append(math.ceil(gate_start[j] * tau * 1e12) + shift)
                gate_widths[i].append(math.ceil(demodduty[j] * tau * 1e9))
    return gate_widths, gate_starts

def get_coarse_gate_shifts(freq, k):
    gate_width = math.ceil((((1/freq)*1e12) // k) * 1e-3 )
    gate_starts = np.array([[(gate_width * (gate_step))] for gate_step in range(k)]) * 1e3
    gate_widths = np.array([[gate_width] for i in range(k)])
    return gate_widths, gate_starts

def get_gate_shifts(type, freq, k):
    if type == 'coarse':
        name = 'coarse'
    elif type == 'ham':
        name = f'hamk{k}'
    else:
        assert False, 'type must be coarse or ham'
    func = getattr(sys.modules[__name__], f"get_{name}_gate_shifts")
    return func(freq, k)


def print_spad512_information(SPAD1):
    print('--------------------SPAD INFORMATION-------------------')

    info = SPAD1.get_info()
    print("\nGeneral informations of the camera :")
    print(info)
    temp = SPAD1.get_temps()  # Current temperatures of FPGAs, PCB and Chip
    print("\nCurrent temperatures of FPGAs, PCB and Chip :")
    print(temp)
    freq = SPAD1.get_freq()  # Operating frequencies (Laser and frame)
    print("\nOperating frequencies (Laser and frame) :")
    print(freq)

    print('-------------------------------------------------------')


def set_up_spad512(print_info=True):
    SPAD1 = SPAD512S(PORT)
    # # # Set the voltage to the maximum value
    SPAD1.set_Vex(VEX)
    if print_info: print_spad512_information(SPAD1)
    return SPAD1


def burst_capture(
    spad1,
    bit_depth, int_time, burst_time, iterations, gate_steps, gate_step_size,
    gate_step_arbitrary, gate_width, gate_offset,
    gate_direction, gate_trig, overlap, pileup, im_width, timeout
):
    counts = np.zeros((im_width, im_width, gate_steps))
    current_inttime = int_time

    while current_inttime > burst_time:
        counts += spad1.get_gated_intensity(
            bit_depth, burst_time, iterations, gate_steps, gate_step_size,
            gate_step_arbitrary, gate_width, gate_offset,
            gate_direction, gate_trig, overlap, 1, pileup, im_width, timeout
        )
        current_inttime -= burst_time

    counts += spad1.get_gated_intensity(
        bit_depth, current_inttime, iterations, gate_steps, gate_step_size,
        gate_step_arbitrary, gate_width, gate_offset,
        gate_direction, gate_trig, overlap, 1, pileup, im_width, timeout
    )

    return counts



def depth_map_capture(spad1, gate_starts, gate_widths, k, gate_shrinkage,
                        bit_depth, int_time, burst_time, iterations, gate_steps, gate_step_size, #SPAD512 Params
                        gate_step_arbitrary, gate_direction, gate_trig, overlap, pileup, im_width, timeout #SPAD512 Params
                        ):
    coded_vals = np.zeros((im_width, im_width, k))
    for i in range(k):

        print('-------------------------------------------------------')
        print(f'Starting capture for gated function number {i + 1}')
        print('-------------------------------------------------------')

        gate_widths_tmp = gate_widths[i]
        gate_starts_tmp = gate_starts[i]

        counts = np.zeros((im_width, im_width, 1))

        for k in range(len(gate_starts_tmp)):
            gate_width = gate_widths_tmp[k] - gate_shrinkage
            gate_start = gate_starts_tmp[k]


            #gate_start = max(0, gate_starts_tmp[k] + (0 * i))


            print(f'\tGate start: {gate_start}')
            print(f'\tGate width: {gate_width}')

            counts += burst_capture(spad1,
                                   bit_depth=bit_depth, int_time=int_time, burst_time=burst_time,
                                   iterations=iterations, gate_steps=gate_steps, gate_step_size=gate_step_size,
                                   gate_step_arbitrary=gate_step_arbitrary, gate_width=gate_width,
                                   gate_offset=gate_start, gate_direction=gate_direction, gate_trig=gate_trig,
                                   overlap=overlap, pileup=pileup, im_width=im_width, timeout=timeout)

        coded_vals[:, :, i, :] = counts[..., 0]

    print('-------------------------------------------------------')
    print(f'Ending measurements')
    print('-------------------------------------------------------')

    return coded_vals


def correlation_capture(spad1, gate_starts, gate_widths, k, gate_shrinkage,
                        bit_depth, int_time, burst_time, iterations, gate_steps, gate_step_size, #SPAD512 Params
                        gate_step_arbitrary, gate_direction, gate_trig, overlap, pileup, im_width, timeout #SPAD512 Params
                        ):
    correlations = np.zeros((im_width, im_width, k, gate_steps))
    for i in range(k):

        print('-------------------------------------------------------')
        print(f'Starting to measure correlations for gated function number {i + 1}')
        print('-------------------------------------------------------')

        gate_widths_tmp = gate_widths[i]
        gate_starts_tmp = gate_starts[i]

        counts = np.zeros((im_width, im_width, gate_steps))

        for k in range(len(gate_starts_tmp)):
            gate_width = gate_widths_tmp[k] - gate_shrinkage
            gate_start = gate_starts_tmp[k]


            #gate_start = max(0, gate_starts_tmp[k] + (0 * i))


            print(f'\tGate start: {gate_start}')
            print(f'\tGate width: {gate_width}')

            counts += burst_capture(spad1,
                                   bit_depth=bit_depth, int_time=int_time, burst_time=burst_time,
                                   iterations=iterations, gate_steps=gate_steps, gate_step_size=gate_step_size,
                                   gate_step_arbitrary=gate_step_arbitrary, gate_width=gate_width,
                                   gate_offset=gate_start, gate_direction=gate_direction, gate_trig=gate_trig,
                                   overlap=overlap, pileup=pileup, im_width=im_width, timeout=timeout)

        correlations[:, :, i, :] = counts
    #correlations = np.flip(correlations, axis=-1)

    print('-------------------------------------------------------')
    print(f'Ending correlation measurements')
    print('-------------------------------------------------------')

    return correlations

