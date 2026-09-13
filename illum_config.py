"""Illumination settings for the capture sweeps.

Keyed by (k, capture_type, illum_type). Each entry provides the SDG / laser
illumination parameters that used to live in the per-sweep if/else blocks:

  - duty:                 duty cycle (percent)
  - high_level_amplitude: high level (V)
  - low_level_amplitude:  low level  (V)   -- always -0.5 in the current table
  - gate_shrinkage:       gate shrinkage  -- 0 except ham k=3 which is 5

All entries below target ~0.45 mW average optical power.

To add a new combination, add a row here rather than editing the sweep scripts.
"""

ILLUM_CONFIG = {
    # (k, capture_type, illum_type): {duty, high_level_amplitude, low_level_amplitude}
    # ---- K = 3 ----
    (3, "ham",    "square"):   {"duty": 20, "high_level_amplitude": 0.5,  "low_level_amplitude": -0.5, "gate_shrinkage": 5},
    (3, "coarse", "gaussian"): {"duty": 30, "high_level_amplitude": 0.42, "low_level_amplitude": -0.5, "gate_shrinkage": 0},
    # ---- K = 4 ----
    (4, "ham",    "pulse"):    {"duty": 15, "high_level_amplitude": 0.77, "low_level_amplitude": -0.5, "gate_shrinkage": 0},
    (4, "coarse", "gaussian"): {"duty": 23, "high_level_amplitude": 0.54, "low_level_amplitude": -0.5, "gate_shrinkage": 0},
    # ---- K = 8 ----
    (8, "coarse", "gaussian"): {"duty": 12, "high_level_amplitude": 1.2,  "low_level_amplitude": -0.5, "gate_shrinkage": 0},
}

# trapcoarse shares coarse's illumination — mirror every coarse row.
for (_k, _ct, _it), _v in list(ILLUM_CONFIG.items()):
    if _ct == "coarse":
        ILLUM_CONFIG[(_k, "trapcoarse", _it)] = dict(_v)


def get_illum(k, capture_type, illum_type):
    """Return the illumination dict for (k, capture_type, illum_type).

    Raises KeyError with the list of configured keys if the combo is missing,
    so an unconfigured sweep fails loudly instead of using stale values.
    """
    key = (int(k), capture_type, illum_type)
    if key not in ILLUM_CONFIG:
        raise KeyError(
            f"No illumination config for {key}. Configured combos: "
            + ", ".join(str(x) for x in sorted(ILLUM_CONFIG))
        )
    return ILLUM_CONFIG[key]
