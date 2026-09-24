#Standard imports
import time
import numpy as np

#Library imports
from utils.global_constants import *
from utils.file_utils import build_parser_from_config, save_capture_and_gt_data, capture_phase_shifts
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
INT_TIME = 10  # integration time
GROUND_TRUTH_INT_TIME = 30
BURST_TIME = 10 #Maxiumum burst time is 4800 ms
K = 3  # number of time bins
TRIALS = 1

GATE_SHRINKAGE = 5 #In NS
SLIDING_GATE_WIDTH = None #In NS, only used by sliding (other schemes derive it from k)
CAPTURE_TYPE = 'ham'

# Illumination Parameters:
HIGH_LEVEL_AMPLITUDE =0.5 #in Vpp
LOW_LEVEL_AMPLITUDE = -0.5
CURRENT = 16 #in mA
EDGE = 6 * 1e-9 #Edge rate for pulse wave
DUTY = 30 # In percentage
REP_RATE = 10 * 1e6 #in HZ
ILLUM_TYPE = 'gaussian'

#Single-pixel experiment parameters
PHASE_SHIFTS = ",".join(str(item) for item in [150,90,45])

# Save Parameters
SAVE_INTO_FILE = True
GROUND_TRUTH = True
SAVE_PATH = SAVE_PATH_SINGLE_PIXEL
EXP_PATH = None

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
            k=K,
            trials=TRIALS,
            gate_shrinkage=GATE_SHRINKAGE,
            sliding_gate_width=SLIDING_GATE_WIDTH,
            capture_type=CAPTURE_TYPE,
            high_level_amplitude=HIGH_LEVEL_AMPLITUDE,
            low_level_amplitude=LOW_LEVEL_AMPLITUDE,
            current=CURRENT,
            edge=EDGE,
            duty=DUTY,
            rep_rate=REP_RATE,
            illum_type=ILLUM_TYPE,
            save_into_file=SAVE_INTO_FILE,
            ground_truth=GROUND_TRUTH,
            phase_shifts=PHASE_SHIFTS,
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
    if isinstance(cfg.phase_shifts, str):
        cfg.phase_shifts = capture_phase_shifts(cfg.phase_shifts)


    SPAD1 = set_up_spad512()

    sdg = SDG5162_GATED_PROJECT(
        usb_port="USB0::0xF4ED::0xEE3A::SDG050D2150058::INSTR"
    )

    ldc220 = NIDAQ_LDC220(max_amps=40)
    ldc220.set_current(0)

    sdg.set_waveform_and_trigger(cfg.illum_type, cfg.duty, cfg.rep_rate,
                                 cfg.high_level_amplitude, cfg.low_level_amplitude,0, cfg.edge)
    sdg.turn_both_channels_on()

    ldc220.set_current(cfg.current)
    #exit(0)

    gate_widths, gate_starts = get_gate_shifts(cfg.capture_type, cfg.rep_rate, cfg.k, cfg.sliding_gate_width)

    total_count = sum(len(sublist) for sublist in gate_widths)
    cfg.int_time = cfg.int_time/total_count if cfg.split_acquisition else cfg.int_time

    print(f"int_time: {cfg.int_time}")

    time.sleep(20)

    needed = {k: v for k, v in asdict(cfg).items() if k in depth_map_capture.__code__.co_varnames}
    needed.pop('int_time')

    coded_vals_range = []
    gt_coded_vals_range = [] if cfg.ground_truth else None

    def phase_shift_helper(phase_shift, rep_rate, illum_type):
        if illum_type == "pulse":
            delay_ns = phase_shift / 360 * (1 / rep_rate)
            sdg.set_delay(delay_ns)
        else:
            sdg.set_phase_shift(phase_shift)

    for phase_shift in cfg.phase_shifts:
        phase_shift_helper(phase_shift, cfg.rep_rate, cfg.illum_type)
        #sdg.set_phase_shift(phase_shift)

        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(f'phase_shift : {phase_shift}')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        time.sleep(5)

        trial_runs = []
        for trials in range(cfg.trials):
            if cfg.capture_type == "ham" and cfg.k == 4:
                # One sweep per gate width instead of one capture per gate. The widths and
                # starts come straight from get_gate_shifts, so these are exactly the gates
                # the slow path fires and the tuned gate shrinkage still applies unchanged.
                sweep_needed = {k: v for k, v in asdict(cfg).items() if k in burst_capture.__code__.co_varnames}
                coded_vals = np.zeros((512, cfg.im_width, cfg.k))

                width_groups = {}
                for code in range(cfg.k):
                    for gate_width, gate_start in zip(gate_widths[code], gate_starts[code]):
                        width_groups.setdefault(gate_width, []).append((gate_start, code))

                for gate_width, gates in sorted(width_groups.items()):
                    gates.sort()
                    starts = [start for start, _ in gates]
                    steps = np.diff(starts)
                    assert len(set(steps)) <= 1, f"gate starts {starts} are not uniformly spaced"

                    sweep_needed["gate_steps"] = len(starts)
                    sweep_needed["gate_step_size"] = int(steps[0]) if len(steps) else 0
                    sweep_needed["gate_offset"] = starts[0]

                    if trials == 0:
                        print(f"Doing Fast Capture of {cfg.capture_type} K={cfg.k}")
                        print(f"gate_width: {gate_width - cfg.gate_shrinkage}")
                        print(f"gate_step_size: {sweep_needed['gate_step_size']}")
                        print(f"gate_steps: {len(starts)}, gate_offset: {starts[0]}")
                        print("---------------------------------")
                    print(f"trial number starting: {trials}")

                    counts = burst_capture(SPAD1, gate_width=gate_width - cfg.gate_shrinkage, **sweep_needed)
                    for step, (_, code) in enumerate(gates):
                        coded_vals[:, :, code] += counts[:, :, step]
            elif cfg.capture_type in ("coarse", "trapcoarse", "sliding") or (cfg.capture_type == "ham" and cfg.k < 4):
                needed = {k: v for k, v in asdict(cfg).items() if k in burst_capture.__code__.co_varnames}
                gate_width = gate_widths[0][0] - cfg.gate_shrinkage
                needed["gate_step_size"] = gate_starts[1][0]
                #needed["gate_step_size"] = gate_width * 1e3
                needed["gate_steps"] = cfg.k
                needed["gate_offset"] = 0

                if trials == 0:
                    print(f"Doing Fast Capture of {cfg.capture_type} K={cfg.k}")
                    print(f"gate_width: {gate_width}")
                    print(f"gate_step_size: {gate_starts[1]}")
                    print("---------------------------------")
                print(f"trial number starting: {trials}")
                coded_vals = burst_capture(SPAD1, gate_width=gate_width, **needed)
            else:
                coded_vals = depth_map_capture(SPAD1, gate_starts=gate_starts, gate_widths=gate_widths,
                                  int_time=cfg.int_time, **needed)
            trial_runs.append(coded_vals)
        print("---------------------------------")

        # import matplotlib.pyplot as plt
        # plt.plot(np.sum(np.sum(coded_vals[SINGLE_PIXEL_COORDS['y'][0]:SINGLE_PIXEL_COORDS['y'][1], SINGLE_PIXEL_COORDS['x'][0]:SINGLE_PIXEL_COORDS['x'][1], :], axis=0), axis=0))
        # plt.show()
        # exit()
        coded_vals_range.append(np.stack(trial_runs))
        if cfg.ground_truth:
            gt_coded_vals = depth_map_capture(SPAD1, gate_starts=gate_starts, gate_widths=gate_widths,
                                                         int_time=cfg.ground_truth_int_time, **needed)
            gt_coded_vals_range.append(gt_coded_vals)

    single_pixel_coded_vals = np.swapaxes(
        np.stack([x.astype(np.float32) for x in coded_vals_range]),
        1,
        0
    )
    gt_single_pixel_coded_vals = np.stack(gt_coded_vals_range) if gt_coded_vals_range is not None else None

    ldc220.set_current(0)
    sdg.turn_both_channels_off()

    if cfg.save_into_file:
        save_capture_and_gt_data(save_path=cfg.save_path, cfg_dict=asdict(cfg),
                                     coded_vals=single_pixel_coded_vals, gt_coded_vals=gt_single_pixel_coded_vals)


