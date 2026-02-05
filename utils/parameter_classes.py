from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Camera
    im_width: Optional[int] = None
    bit_depth: Optional[int] = None

    # Capture
    int_time: Optional[int] = None
    burst_time: Optional[int] = None
    k: Optional[int] = None
    gate_step_size: Optional[int] = None
    gate_shrinkage: Optional[int] = None
    capture_type: Optional[str] = None

    # Illumination
    amplitude: Optional[float] = None
    current: Optional[float] = None
    edge: Optional[float] = None
    duty: Optional[int] = None
    rep_rate: Optional[float] = None
    illum_type: Optional[str] = None

    # Plot
    plot_correlations: Optional[bool] = None

    # Save
    save_into_file: Optional[bool] = None
    save_path: Optional[str] = None
    exp_path: Optional[int] = None #Groups into folder


    #Capture only parameters
    ground_truth: Optional[bool] = None #only used for captures
    ground_truth_int_time: Optional[int] = None #only used for dual capture

    # Parameters from previous parameters
    gate_steps: Optional[int] = None
    rep_tau: Optional[float] = None

    # Non-editable / control
    iterations: Optional[int] = None
    overlap: Optional[int] = None
    timeout: Optional[int] = None
    pileup: Optional[int] = None
    gate_step_arbitrary: Optional[int] = None
    gate_direction: Optional[int] = None
    gate_trig: Optional[int] = None


@dataclass
class DecodeConfig:
    # Experiment
    exp_path: Optional[str] = None

    # Domain / decoding
    n_tbins: Optional[int] = None

    # Plot limits
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    median_filter_size: Optional[int] = None

    # Processing toggles
    correct_master: Optional[bool] = None
    plot_depth_maps: Optional[bool] = None
    mask_background_pixels: Optional[bool] = None

    simulated_correlations: Optional[bool] = None
    use_full_correlations: Optional[bool] = None

    # Coding matrix build
    smooth_sigma: Optional[float] = None
    shift: Optional[int] = None

    # Post-processing
    correct_depth_distortion: Optional[bool] = None
    normalize_depth_maps: Optional[bool] = None