import numpy as np

# =======================================
# Parameters
# =======================================
N = 16384          # data length
T = 200e-9         # 200 ns period (for x axis)
FWHM = 65e-9       # 60 ns FWHM
FREQ = 5e6         # Freq
VOLTAGE = 2        # Voltage or amplitude

# Time axis (xpos column)
t = np.linspace(0, T, N, endpoint=False)

# Gaussian
sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
t0 = T / 2.0
g = np.exp(-0.5 * ((t - t0) / sigma) ** 2)

# Normalize amplitude to 0–1 (you can scale later in AWG)
g /= g.max()
g = 2*g - 1

outfile = f"gauss{int(FWHM*1e9)}ns_{int(FREQ * 1e-6)}mhz_{VOLTAGE}v.csv"

with open(outfile, "w") as f:
    # Header block matching your working file style
    f.write(f"data length,{N}\n")
    f.write(f"frequency,{FREQ}\n")   # mimic your example where freq == data length
    f.write(f"amp,{VOLTAGE}\n")            # arbitrary, AWG amplitude is set later
    f.write("offset,0\n")
    f.write("phase,0\n")
    for _ in range(7):
        f.write(",\n")            # same 7 comma-only lines as your file

    # Column header
    f.write("xpos,value\n")

    # Data lines: xpos,value
    for ti, gi in zip(t, g):
        f.write(f"{ti:.6E},{gi:.9f}\n")

print(f"Written {outfile}")
