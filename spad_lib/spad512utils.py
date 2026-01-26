import numpy as np
import math
import SPAD512S
from utils.global_constants import VEX, PORT



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


def GetHamK3_GateShifts(freq):
    K = 3
    tau = float(1 / freq) #Repition tau
    demodDuty = 1./2.
    shifts = [0, (1. / 3.), (2. / 3.)]
    gate_widths = [[], [], []]
    gate_starts = [[], [], []]
    for i in range(K):
        gate_widths[i].append(math.ceil((demodDuty * tau * 1e9)))
        gate_starts[i].append(math.ceil((shifts[i] * tau * 1e12)))
    return gate_widths, gate_starts


def GetHamK4_GateShifts(freq):
    K = 4
    tau = float(1 / freq) #Repition tau
    demodDuty1 = np.array([6./12.,6./12.])
    shift1 = 5./12.
    demodDuty2 = np.array([6./12.,6./12.])
    shift2 = 2./12.
    demodDuty3 = np.array([3./12.,4./12.,3./12.,2./12.])
    shift3 = 0./12.
    demodDuty4 = np.array([2./12.,3./12,4./12.,3./12.])
    shift4 = 4./12.
    gate_starts = [[], [], [], []]
    gate_widths = [[], [], [], []]
    demodDutys = [demodDuty1, demodDuty2, demodDuty3, demodDuty4]
    shifts = [shift1, shift2, shift3, shift4]
    for i in range(0,K):
        demodDuty = demodDutys[i]
        #startIndeces = np.floor((np.cumsum(demodDuty) - demodDuty)*N)
        gate_start = (np.cumsum(demodDuty) - demodDuty)
        print(np.cumsum(demodDuty) - demodDuty)
        #endIndeces = startIndeces + np.floor(demodDuty*N) - 1
        for j in range(len(demodDuty)):
            if((j%2) == 0):
                shift = math.ceil(shifts[i] * tau * 1e12)
                gate_starts[i].append(math.ceil(gate_start[j] * tau * 1e12) + shift)
                gate_widths[i].append(math.ceil(demodDuty[j] * tau * 1e9))
    return gate_widths, gate_starts

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


def set_up_spad512(print_info=False):
    SPAD1 = SPAD512S(PORT)
    # # # Set the voltage to the maximum value
    SPAD1.set_Vex(VEX)
    if print_info: print_spad512_information(SPAD1)


def burst_capture(cfg):
    counts = np.zeros((im_width, im_width, n_tbins))
    current_inttime = int_time
    while current_inttime > 480:
        # print(f'starting current time {current_inttime}')
        counts += spad1.get_gated_intensity(bit_depth, 480, iterations, n_tbins, shift,
                                            gate_step_arbitrary, gate_width,
                                            gate_start, gate_direction, gate_trig, overlap, 1, pileup, im_width)
        current_inttime -= 480

    counts += spad1.get_gated_intensity(bit_depth, current_inttime, iterations, n_tbins, shift,
                                        gate_step_arbitrary, gate_width,
                                        gate_start, gate_direction, gate_trig, overlap, 1, pileup, im_width)

