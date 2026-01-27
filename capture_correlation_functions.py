#Standard imports
import os
import numpy as np
import time

#Library imports
from utils.global_constants import *
from utils.file_utils import Config, build_parser_from_config, save_correlation_data
from spad_lib.spad512utils import set_up_spad512, get_gate_shifts, correlation_capture
from utils.instrument_utils import SDG5162_GATED_PROJECT, NIDAQ_LDC220
from plot_scripts.plot_utils import plot_correlation_functions
##### Editable parameters (defaults; can be overridden via CLI)  #####

# Camera Parameters
IM_WIDTH = 512  # image width
BIT_DEPTH = 12

# Capture parameters
INT_TIME = 100  # integration time
BURST_TIME = 480
K = 3  # number of time bins
SHIFT = 2500  #50  # shift in picoseconds
GATE_SHRINKAGE = 25 #In NS
CAPTURE_TYPE = 'coarse'

# Illumination Parameters:
AMPLITUDE = 8.5 #in Vpp
CURRENT = 55 #in mA
EDGE = 6 * 1e5 #Edge rate for square wave
DUTY = 32 # In percentage
REP_RATE = 5 * 1e6 #in HZ
ILLUM_TYPE = 'gaussian'

#Plot Parameters
PLOT_CORRELATIONS = True

# Save Parameters
SAVE_INTO_FILE = True
SAVE_PATH = SAVE_PATH_CORRELATIONS


###### Non-Editable Parameters #####
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
    parser = build_parser_from_config(Config)
    args = parser.parse_args()

    def apply_defaults(cfg: Config) -> Config:
        # Only fill values that are still None
        defaults = dict(
            im_width=IM_WIDTH,
            bit_depth=BIT_DEPTH,
            int_time=INT_TIME,
            burst_time=BURST_TIME,
            k=K,
            shift=SHIFT,
            gate_shrinkage=GATE_SHRINKAGE,
            capture_type=CAPTURE_TYPE,
            amplitude=AMPLITUDE,
            current=CURRENT,
            edge=EDGE,
            duty=DUTY,
            rep_rate=REP_RATE,
            illum_type=ILLUM_TYPE,
            plot_correlations=PLOT_CORRELATIONS,
            save_into_file=SAVE_INTO_FILE,
            save_path=SAVE_PATH,
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
        setattr(cfg, 'rep_tau', (1/cfg.rep_rate))
        setattr(cfg, 'n_tbins', int((cfg.rep_tau*1e12) // cfg.shift))
        return cfg

    cfg = Config(**vars(args))
    cfg = apply_defaults(cfg)

    SPAD1 = set_up_spad512()

    sdg = SDG5162_GATED_PROJECT(
        usb_port="USB0::0xF4ED::0xEE3A::SDG050D2150058::INSTR"
    )

    ldc220 = NIDAQ_LDC220()
    ldc220.set_current(0)


    sdg.set_waveform_and_trigger(cfg.illum_type, cfg.duty, cfg.rep_rate, cfg.amplitude, cfg.edge)
    sdg.turn_both_channels_on()

    ldc220.set_current(cfg.current)

    gate_widths, gate_starts = get_gate_shifts(cfg.capture_type, cfg.rep_rate, cfg.k)

    time.sleep(20)
    correlations = correlation_capture(SPAD1, gate_starts=gate_starts, gate_widths=gate_widths, cfg=cfg)

    ldc220.set_current(0)
    sdg.turn_both_channels_off()

    if cfg.plot_correlations:
        point_list = [(10, 10), (200, 200), (50, 200)]
        plot_correlation_functions(
            point_list,
            correlations
        )

    if cfg.save_into_file:
        save_correlation_data(save_path=cfg.save_path, cfg=cfg, correlations=correlations)




