import os
import re
import numpy as np
from PIL import Image
from dataclasses import fields, asdict, is_dataclass
from typing import  get_origin, get_args, Union
import argparse

def get_data_folder(data_folder_mac, data_folder_linux) -> str:
    if os.path.exists(data_folder_mac):
        return data_folder_mac
    return data_folder_linux

def load_hot_mask(path: str) -> np.ndarray:
    hot_mask = np.load(path)
    return hot_mask

def filter_npz_files(npz_files, k_list):
    filtered = []
    for f in npz_files:
        base_first = os.path.basename(f).split("_")[0]
        keep = any(
            (str(val) in base_first) or ("_gt_" in base_first and str(val) in base_first)
            for val in k_list
        )
        if keep:
            filtered.append(f)
    return filtered


def filter_capture_files(npz_files):
    filtered = []
    for f in npz_files:
        base_first = os.path.basename(f)
        keep = "gt" not in base_first
        if keep:
            filtered.append(f)
    return filtered


def save_capture_and_gt_data(save_path, cfg_dict, coded_vals, gt_coded_vals):
    save_path = os.path.join(save_path, cfg_dict['exp_path']) \
            if cfg_dict['exp_path'] is not None else make_next_exp_folder(save_path)

    cfg_dict['ground_truth'] = False
    save_capture_data(save_path=save_path, cfg_dict=cfg_dict, coded_vals=coded_vals)
    if gt_coded_vals is not None:
        cfg_dict['ground_truth'] = True
        cfg_dict['int_time'] = cfg_dict['ground_truth_int_time']
        save_capture_data(save_path=save_path, cfg_dict=cfg_dict, coded_vals=gt_coded_vals)


def save_capture_data(save_path, cfg_dict, coded_vals):
    save_name = make_capture_filename(cfg_dict['capture_type'],cfg_dict['k'], cfg_dict['rep_rate'] * 1e-6,
                              cfg_dict['high_level_amplitude'] * 1000, cfg_dict['current'],cfg_dict['duty'],
                              cfg_dict['int_time'], cfg_dict['ground_truth'])

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, save_name)
    np.savez(out_file, coded_vals=coded_vals, cfg=cfg_dict)
    print(f"✅ Saved Capture data to {out_file}.npz")

def save_correlation_data(save_path, cfg_dict, correlations):

    save_name = make_correlation_filename(cfg_dict['capture_type'],cfg_dict['k'], cfg_dict['rep_rate'] * 1e-6,
                              cfg_dict['high_level_amplitude'] * 1000, cfg_dict['current'],cfg_dict['duty'])

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, save_name)
    np.savez(out_file, correlations=correlations, cfg=cfg_dict)
    print(f"✅ Saved correlation data to {out_file}.npz")


def make_filename(capture_type, k, freq_mhz, mV, mA, duty):
    return (
        f"{capture_type}k{k}_"
        f"{freq_mhz:.0f}mhz_"
        f"{mV:.0f}mV_"
        f"{mA:.0f}mA_"
        f"{duty:.0f}duty"
    )

def make_correlation_filename(capture_type, k, freq_mhz, mV, mA, duty):
    return make_filename(capture_type, k, freq_mhz, mV, mA, duty) + '_correlations.npz'

def make_capture_filename(capture_type, k, freq_mhz, mV, mA, duty, int_time, ground_truth):
    ground_truth_tag = '_gt' if ground_truth else f'_{int_time}ms'
    return (make_filename(capture_type, k, freq_mhz, mV, mA, duty) +
            ground_truth_tag +
            '_capture.npz')

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("true", "1", "yes", "y")

def _base_type(annot):
    """
    Optional[int] -> int, Optional[float] -> float, etc.
    """
    origin = get_origin(annot)
    if origin is Union:
        args = [a for a in get_args(annot) if a is not type(None)]
        return args[0] if len(args) == 1 else annot
    return annot

def build_parser_from_config(config_cls, *, bool_parser=str2bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Correlation function capture")

    for f in fields(config_cls):
        t = _base_type(f.type)

        # argparse can't handle type=bool correctly, so use str2bool
        arg_type = bool_parser if t is bool else t

        parser.add_argument(f"--{f.name}", type=arg_type, default=None)

    return parser


def corr_parse_run(s: str):
    cap, k, f, mv, ma, duty = s.split(",")
    return dict(
        capture_type=cap,
        k=int(k),
        freq_mhz=float(f),
        mV=float(mv),
        mA=float(ma),
        duty=float(duty),
    )

def capture_parse_run(s: str):
    cap, k, f, mv, ma, duty, int_time = s.split(",")
    return dict(
        capture_type=cap,
        k=int(k),
        freq_mhz=float(f),
        mV=float(mv),
        mA=float(ma),
        duty=float(duty),
        int_time=int(int_time),
    )


def make_next_exp_folder(base_dir, prefix="exp"):
    """
    Creates:
        exp_0, exp_1, exp_2, ...

    Returns the newly created folder path.
    """
    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    pattern = re.compile(rf"{prefix}_(\d+)$")

    nums = []
    for d in existing:
        m = pattern.match(d)
        if m:
            nums.append(int(m.group(1)))

    next_idx = 0 if len(nums) == 0 else max(nums) + 1

    new_path = os.path.join(base_dir, f"{prefix}_{next_idx}")
    os.makedirs(new_path)

    return new_path
