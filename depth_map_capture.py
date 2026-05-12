#Standard imports
import time
import numpy as np
#Library imports
from utils.global_constants import *
from utils.file_utils import build_parser_from_config, save_capture_and_gt_data, save_capture_data
from utils.parameter_classes import  Config
from spad_lib.spad512utils import set_up_spad512, get_gate_shifts, depth_map_capture, burst_capture
from utils.instrument_utils import SDG5162_GATED_PROJECT, NIDAQ_LDC220
from dataclasses import asdict
##### Editable parameters (defaults; can be overridden via CLI)  #####

# Camera Parameters
IM_WIDTH = 512  # image width
BIT_DEPTH = 12

# Capture parameters
SPLIT_ACQUISITION = True
INT_TIME = 1  #burst integration time
GROUND_TRUTH_INT_TIME = 2_0 #total integration time
BURST_TIME = 1 #burst time so that we dont over flow
MAX_TRIALS = 100
K = 3  # number of time bins

GATE_SHRINKAGE = 5 #In NS
CAPTURE_TYPE = 'ham'

# Illumination Parameters:
HIGH_LEVEL_AMPLITUDE = 0.5 #in Vpp
LOW_LEVEL_AMPLITUDE = -0.5
CURRENT = 16 #in mA
EDGE = 6 * 1e-9 #Edge rate for pulse wave
PHASE = 90
DUTY = 20 # In percentage
REP_RATE = 10 * 1e6 #in HZ
ILLUM_TYPE = 'square'

# Save Parameters
SAVE_INTO_FILE = True
SAVE_PATH = SAVE_PATH_CAPTURE
EXP_PATH = "exp_1"

###### Non-Editable Parameters #####
ITERATIONS = 1
OVERLAP = 0
TIMEOUT = 0
PILEUP = 0
GATE_STEP_ARBITRARY = 0
GATE_STEP_SIZE = 0
GATE_STEPS = 1
GATE_DIRECTION = 1
GATE_TRIG = 0



if __name__ == "__main__":
    parser = build_parser_from_config(Config)
    args = parser.parse_args()

    def apply_defaults(cfg: Config) -> Config:
        # Only fill values that are still None
        defaults = dict(
            im_width=IM_WIDTH,
            bit_depth=BIT_DEPTH,
            int_time=INT_TIME,
            split_acquisition=SPLIT_ACQUISITION,
            ground_truth_int_time=GROUND_TRUTH_INT_TIME,
            burst_time=BURST_TIME,
            max_trials=MAX_TRIALS,
            k=K,
            gate_shrinkage=GATE_SHRINKAGE,
            capture_type=CAPTURE_TYPE,
            high_level_amplitude=HIGH_LEVEL_AMPLITUDE,
            low_level_amplitude=LOW_LEVEL_AMPLITUDE,
            current=CURRENT,
            edge=EDGE,
            phase=PHASE,
            duty=DUTY,
            rep_rate=REP_RATE,
            illum_type=ILLUM_TYPE,
            save_into_file=SAVE_INTO_FILE,
            save_path=SAVE_PATH,
            exp_path=EXP_PATH,
            iterations=ITERATIONS,
            overlap=OVERLAP,
            timeout=TIMEOUT,
            pileup=PILEUP,
            gate_steps=GATE_STEPS,
            gate_step_arbitrary=GATE_STEP_ARBITRARY,
            gate_step_size=GATE_STEP_SIZE,
            gate_direction=GATE_DIRECTION,
            gate_trig=GATE_TRIG,
        )

        for k, v in defaults.items():
            if getattr(cfg, k) is None:
                setattr(cfg, k, v)
        setattr(cfg, 'rep_tau', (1/REP_RATE))
        return cfg

    cfg = Config(**vars(args))
    cfg = apply_defaults(cfg)


    SPAD1 = set_up_spad512()

    sdg = SDG5162_GATED_PROJECT(
        usb_port="USB0::0xF4ED::0xEE3A::SDG050D2150058::INSTR"
    )

    ldc220 = NIDAQ_LDC220(max_amps=40)
    ldc220.set_current(0)


    sdg.set_waveform_and_trigger(cfg.illum_type, cfg.duty, cfg.rep_rate,
                                 cfg.high_level_amplitude, cfg.low_level_amplitude, 0, cfg.edge)

    def phase_shift_helper(phase_shift, rep_rate, illum_type):
        if illum_type == "pulse":
            delay_ns = phase_shift / 360 * (1 / rep_rate)
            sdg.set_delay(delay_ns)
        else:
            sdg.set_phase_shift(phase_shift)

    phase_shift_helper(cfg.phase, cfg.rep_rate, cfg.illum_type)

    sdg.turn_both_channels_on()
    sdg.turn_both_channels_on()
    sdg.turn_both_channels_on()
    sdg.turn_both_channels_on()

    ldc220.set_current(cfg.current)

    gate_widths, gate_starts = get_gate_shifts(cfg.capture_type, cfg.rep_rate, cfg.k)
    total_count = sum(len(sublist) for sublist in gate_widths)

    time.sleep(45)

    trials_calc = int(cfg.ground_truth_int_time // cfg.int_time)
    trials = max(trials_calc, cfg.max_trials)
    needed = {k: v for k, v in asdict(cfg).items() if k in depth_map_capture.__code__.co_varnames}
    needed.pop('int_time')

    trial_runs = []

    current_int_time = 0
    i = 0
    while current_int_time < cfg.ground_truth_int_time:
        int_time = cfg.int_time if i < cfg.max_trials else cfg.burst_time
        int_time = int_time / total_count if cfg.split_acquisition else int_time
        if i == 0 or i == cfg.max_trials: print('int_time:', int_time)
        if cfg.capture_type == "timeslicing":
            ts_needed = {k: v for k, v in asdict(cfg).items() if k in burst_capture.__code__.co_varnames}
            gate_width = gate_widths[0][0]
            ts_needed["gate_step_size"] = gate_width * 1e3
            ts_needed["gate_steps"] = cfg.k
            ts_needed["gate_offset"] = 0
            ts_needed["int_time"] = int_time
            coded_vals = burst_capture(SPAD1, gate_width=gate_width, **ts_needed)
        else:
            coded_vals = depth_map_capture(SPAD1, gate_starts=gate_starts, gate_widths=gate_widths,
                                           int_time=int_time, **needed)

        trial_runs.append(coded_vals)
        current_int_time += int_time
        i += 1
    depth_map_coded_vals = np.stack([x.astype(np.float32) for x in trial_runs])

    ldc220.set_current(0)
    sdg.turn_both_channels_off()

    if cfg.save_into_file:
        save_capture_and_gt_data(save_path=cfg.save_path, cfg_dict=asdict(cfg),coded_vals=depth_map_coded_vals,
                                 gt_coded_vals=None)


