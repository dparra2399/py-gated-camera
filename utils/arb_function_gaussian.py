import numpy as np
import os
# =======================================
# Parameters (shape-only)
# =======================================
N = 16384
DUTY_FWHM = 0.30      # FWHM as fraction of period (0..1)
CENTER = 0.5           # pulse center in cycle fraction
VOLTAGE = 2            # keep your 2 so normalization maps to [-1, 1]

# Normalized phase axis: 0..1
x = np.linspace(0.0, 1.0, N, endpoint=False)

# Gaussian in "cycle units"
# If FWHM is in fraction of a cycle, sigma is also in fraction of a cycle.
sigma = DUTY_FWHM / (2 * np.sqrt(2 * np.log(2)))
g = np.exp(-0.5 * ((x - CENTER) / sigma) ** 2)

# Normalize to [-1, 1]
g /= g.max()
g = 2*g - 1


outfile = os.path.join('arb_functions', f"GAUSS{DUTY_FWHM*100:.0f}DUTY.csv")

with open(outfile, "w") as f:
    f.write(f"data length,{N}\n")
    # Put something harmless or omit if your importer requires it
    f.write("frequency,1\n")
    f.write(f"amp,{VOLTAGE}\n")
    f.write("offset,0\n")
    f.write("phase,0\n")
    for _ in range(7):
        f.write(",\n")
    f.write("xpos,value\n")

    # xpos can be normalized 0..1; many importers ignore it anyway
    for xi, gi in zip(x, g):
        f.write(f"{xi:.6E},{gi:.9f}\n")

print(f"Written {outfile}")
