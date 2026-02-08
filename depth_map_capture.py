#Standard imports
import time

#Library imports
from utils.global_constants import *
from utils.file_utils import build_parser_from_config, save_capture_and_gt_data
from utils.parameter_classes import  Config
from spad_lib.spad512utils import set_up_spad512, get_gate_shifts, depth_map_capture
from utils.instrument_utils import SDG5162_GATED_PROJECT, NIDAQ_LDC220
from dataclasses import asdict
##### Editable parameters (defaults; can be overridden via CLI)  #####

# Camera Parameters
IM_WIDTH = 512  # image width
BIT_DEPTH = 12

# Capture parameters
INT_TIME = 1000  # integration time
GROUND_TRUTH_INT_TIME = 200_000
BURST_TIME = 4800 #Maxiumum burst time is 4800 ms
K = 3  # number of time bins

GATE_SHRINKAGE = 1 #In NS
CAPTURE_TYPE = 'coarse'

# Illumination Parameters:
HIGH_LEVEL_AMPLITUDE = 2.4 #in Vpp
LOW_LEVEL_AMPLITUDE = -3.0
CURRENT = 50 #in mA
EDGE = 6 * 1e-9 #Edge rate for pulse wave
PHASE = 20
DUTY = 30 # In percentage
REP_RATE = 5 * 1e6 #in HZ
ILLUM_TYPE = 'gaussian'

# Save Parameters
SAVE_INTO_FILE = True
GROUND_TRUTH = True
SAVE_PATH = SAVE_PATH_CAPTURE
EXP_PATH = 'exp_0'

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
            ground_truth_int_time=GROUND_TRUTH_INT_TIME,
            burst_time=BURST_TIME,
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
            ground_truth=GROUND_TRUTH,
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

    ldc220 = NIDAQ_LDC220(max_amps=100)
    ldc220.set_current(0)

    sdg.set_waveform_and_trigger(cfg.illum_type, cfg.duty, cfg.rep_rate,
                                 cfg.high_level_amplitude, cfg.low_level_amplitude, cfg.phase, cfg.edge)
    sdg.turn_both_channels_on()

    ldc220.set_current(cfg.current)

    gate_widths, gate_starts = get_gate_shifts(cfg.capture_type, cfg.rep_rate, cfg.k)

    time.sleep(20)

    needed = {k: v for k, v in asdict(cfg).items() if k in depth_map_capture.__code__.co_varnames}
    coded_vals = depth_map_capture(SPAD1, gate_starts=gate_starts, gate_widths=gate_widths, **needed)

    gt_coded_vals = None
    if cfg.ground_truth:
        needed['int_time'] = cfg.ground_truth_int_time
        gt_coded_vals = depth_map_capture(SPAD1, gate_starts=gate_starts, gate_widths=gate_widths, **needed)

    ldc220.set_current(0)
    sdg.turn_both_channels_off()

    if cfg.save_into_file:
        save_capture_and_gt_data(save_path=cfg.save_path, cfg_dict=asdict(cfg),
                                     coded_vals=coded_vals, gt_coded_vals=gt_coded_vals)


