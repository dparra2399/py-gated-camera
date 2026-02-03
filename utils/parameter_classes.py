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