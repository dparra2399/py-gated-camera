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


def correct_spherical_distortion(depth_map, f_mm=None, pitch_um=None, fx=None, fy=None, cx=None, cy=None):
    """
    Correct spherical (radial) depth distortion from a ToF/LiDAR sensor.

    A ToF sensor measures the radial distance r along each ray. This converts
    it to the planar z-depth (perpendicular distance to the image plane):

        z = r * cos(theta)  =  r / sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)

    Parameters
    ----------
    depth_map : np.ndarray (H, W)
        Raw depth map containing radial distances.
    f_mm : float, optional
        Focal length in mm. Used with pitch_um to derive fx/fy.
    pitch_um : float, optional
        Pixel pitch in micrometres. Used with f_mm to derive fx/fy.
    fx, fy : float, optional
        Focal lengths in pixels. If provided, f_mm and pitch_um are ignored.
    cx, cy : float, optional
        Principal point in pixels. Defaults to image centre if not provided.

    Returns
    -------
    np.ndarray (H, W)
        Corrected planar depth map.
    """
    H, W = depth_map.shape

    if fx is None or fy is None:
        if f_mm is None or pitch_um is None:
            raise ValueError("Provide either (fx, fy) or (f_mm, pitch_um).")
        fx, fy, _cx, _cy = intrinsics_from_pixel_pitch(W, H, f_mm, pitch_um)
        if cx is None: cx = _cx
        if cy is None: cy = _cy
    else:
        if cx is None: cx = W / 2.0
        if cy is None: cy = H / 2.0

    u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                        np.arange(H, dtype=np.float64))

    # cos(theta) = 1 / sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)
    cos_theta = 1.0 / np.sqrt(1.0 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)

    return depth_map * cos_theta


def correct_bistatic_distortion(depth_map, baseline, f_mm=None, pitch_um=None,
                                 fx=None, fy=None, cx=None, cy=None):
    """
    Correct for bistatic (offset transmitter/receiver) geometry AND spherical
    distortion in a single pass.

    A co-located (monostatic) ToF sensor measures the radial distance r from
    the sensor to the point.  When the laser is offset from the detector by a
    baseline vector b (in metres), the measured quantity is the bistatic range:

        r_meas = (|LP| + |PD|) / 2

    Inverting the ellipsoid equation gives the true radial distance t from the
    detector centre to the point:

        t = (4·r² − |b|²) / (2·(2·r − n̂·b))

    where n̂ is the unit ray direction for each pixel and b is the laser
    position relative to the detector.  The planar z-depth is then:

        z = t · cos(θ)

    When baseline=(0,0,0) this reduces to the monostatic spherical correction.

    Parameters
    ----------
    depth_map : np.ndarray (H, W)
        Raw depth map (bistatic range r_meas per pixel, in metres).
    baseline : array-like, shape (3,)
        Laser position relative to detector centre in metres.
        E.g. (0.10, 0, 0) for a 10 cm horizontal offset to the right.
    f_mm : float, optional
        Focal length in mm. Used with pitch_um to derive fx/fy.
    pitch_um : float, optional
        Pixel pitch in micrometres. Used with f_mm to derive fx/fy.
    fx, fy : float, optional
        Focal lengths in pixels. Overrides f_mm / pitch_um if provided.
    cx, cy : float, optional
        Principal point in pixels. Defaults to image centre.

    Returns
    -------
    np.ndarray (H, W)
        Corrected planar depth map (z-depth, in metres).
    """
    H, W = depth_map.shape

    if fx is None or fy is None:
        if f_mm is None or pitch_um is None:
            raise ValueError("Provide either (fx, fy) or (f_mm, pitch_um).")
        fx, fy, _cx, _cy = intrinsics_from_pixel_pitch(W, H, f_mm, pitch_um)
        if cx is None: cx = _cx
        if cy is None: cy = _cy
    else:
        if cx is None: cx = W / 2.0
        if cy is None: cy = H / 2.0

    baseline = np.asarray(baseline, dtype=np.float64)
    b_sq = float(np.dot(baseline, baseline))   # |b|²

    u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                        np.arange(H, dtype=np.float64))

    # unnormalised ray direction: (dx, dy, 1) in camera coordinates
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    ray_norm = np.sqrt(dx ** 2 + dy ** 2 + 1.0)   # ||(dx, dy, 1)||

    # n̂ · b  (projection of baseline onto each unit ray)
    n_dot_b = (dx * baseline[0] + dy * baseline[1] + baseline[2]) / ray_norm

    # cos(θ) = 1 / ray_norm  (same as monostatic spherical correction)
    cos_theta = 1.0 / ray_norm

    r = depth_map.astype(np.float64)

    # true radial distance from detector: t = (4r² - |b|²) / (2*(2r - n̂·b))
    t = (4.0 * r ** 2 - b_sq) / (2.0 * (2.0 * r - n_dot_b))

    return t * cos_theta


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
    gate_starts = [[(gate_width * (gate_step) * 1e3)] for gate_step in range(k)]
    gate_widths = [[gate_width] for i in range(k)]
    return gate_widths, gate_starts

def get_trap_gate_shifts(freq, k):
    gate_widths, gate_starts = get_coarse_gate_shifts(freq, k)
    gate_widths = [[gate_width[0] * 2] for gate_width in gate_widths]
    return gate_widths, gate_starts


def get_gate_shifts(type, freq, k):
    if type == 'coarse' or type == 'rect' or type == "timeslicing":
        name = 'coarse'
    elif type == 'trapcoarse' or type == 'traprect':
        name = 'trap'
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

        coded_vals[:, :, i] = counts[..., 0]

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
    correlations = np.flip(correlations, axis=-1)

    print('-------------------------------------------------------')
    print(f'Ending correlation measurements')
    print('-------------------------------------------------------')

    return correlations

