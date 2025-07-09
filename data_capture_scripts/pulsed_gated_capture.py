# Libraries
import os
import glob
from SPAD512S import SPAD512S
import numpy as np
import time
import matplotlib.pyplot  as plt
from scipy.stats import linregress
import math


port = 9999 # Check the command Server in the setting tab of the software and change it if necessary
SPAD1 = SPAD512S(port)

# Get informations on the system used
info = SPAD1.get_info()
print("\nGeneral informations of the camera :")
print(info)
temp = SPAD1.get_temps() # Current temperatures of FPGAs, PCB and Chip
print("\nCurrent temperatures of FPGAs, PCB and Chip :")
print(temp)
freq = SPAD1.get_freq() # Operating frequencies (Laser and frame)
print("\nOperating frequencies (Laser and frame) :")
print(freq)

# # # Set the voltage to the maximum value
Vex = 7
SPAD1.set_Vex(Vex)


# Editable parameters
intTime = 10 #integration time
num_gates = 16 #number of time bins
im_width = 512 #image width


#Don't edit
iterations = 1
bitDepth = 8
overlap = 1
timeout = 0
pileup = 0
gate_steps =  num_gates
gate_step_arbitrary = 0
gate_width = math.ceil((((1/float(freq[0]))*1e12) // num_gates) * 1e-3 )
gate_step_size = gate_width * 1e3
gate_offset = 0
gate_direction = 1
gate_trig = 0

print(f'gate steps: {gate_steps}')
print(f'gate width: {gate_width}')
print(f'gate step size: {gate_step_size}')

counts = SPAD1.get_gated_intensity(bitDepth, intTime, iterations, gate_steps, gate_step_size, gate_step_arbitrary, gate_width, 
                                    gate_offset, gate_direction, gate_trig, overlap, 0, pileup, im_width)

print(counts.shape)
unit = "ms"
factor_unit = 1e-3
        
    # Show the image after calibration
plt.figure()
plt.imshow(np.mean(counts, axis=2), cmap='gray') # Here we take the mean of counts over the number of iterations
plt.colorbar()
plt.title(f"{bitDepth}-bit image with {intTime}{unit} integration time.")
plt.show()

