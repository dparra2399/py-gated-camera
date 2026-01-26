#Standard imports
import os
import numpy as np
import argparse

#Library imports
from utils.global_constants import *
from utils.file_utils import str2bool

##### Editable parameters (defaults; can be overridden via CLI)  #####

# Camera Parameters
IM_WIDTH = 512  # image width
BIT_DEPTH = 12

# Capture parameters
INT_TIME = 100  # integration time
K = 3  # number of time bins
SHIFT = 2500  #50  # shift in picoseconds
GATE_SHRINKAGE = 25 #In NS
CAPTURE_TYPE = 'ham'

# Illumination Parameters:
VOLTAGE = 8.5 #in Vpp
DUTY = 20 # In percentage
REP_RATE = 5 * 1e6 #in HZ
ILLUM_TYPE = 'square'

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
    # --- CLI overrides (hybrid approach) ---
    parser = argparse.ArgumentParser(description="Correlation function capture")
    parser.add_argument("--int_time", type=int, default=INT_TIME)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--im_width", type=int, default=IM_WIDTH)
    parser.add_argument("--bit_depth", type=int, default=BIT_DEPTH)
    parser.add_argument("--shift", type=int, default=SHIFT)
    parser.add_argument("--voltage", type=float, default=VOLTAGE)
    parser.add_argument("--duty", type=int, default=DUTY)
    parser.add_argument("--rep_rate", type=int, default=REP_RATE)
    parser.add_argument("--save_into_file", type=str2bool, default=SAVE_INTO_FILE)
    parser.add_argument("--save_path", type=str, default=SAVE_PATH)
    parser.add_argument("--plot_correlations",type=str2bool, default=PLOT_CORRELATIONS)
    parser.add_argument("--gate_shrinkage", type=int, default=GATE_SHRINKAGE)
    parser.add_argument("--capture_type", type=str, default=CAPTURE_TYPE)
    parser.add_argument("--illum_type", type=str, default=ILLUM_TYPE)

    args = parser.parse_args()


