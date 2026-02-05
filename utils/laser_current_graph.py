import pyvisa
from instrument_utils import *
import numpy as np
import time
import matplotlib.pyplot as plt

wavelength = 850 #nm
max_current = 60
start_current = 40
num_measurements = 5
step = 1

pm = Instrument(
    usb_port="USB0::0x1313::0x8078::P0012224::INSTR"
)

ni_daq = NIDAQ_LDC220(90)

pm.write("CONF:POW")
pm.write(f"SENS:CORR:WAV {wavelength}")   # set wavelength (nm)
pm.write("SENS:POW:RANG:AUTO ON")

current_arr = np.arange(start_current, max_current, step)
power_current = np.zeros((2, current_arr.shape[0]))
for idx, current in enumerate(current_arr):
    #print(current)
    average_current = 0
    average_power = 0
    for i in range(num_measurements):
        ni_daq.set_current(current)
        time.sleep(5)
        read_current = ni_daq.read_current_zero()

        print(f'read current: {read_current}')

        # Read power
        power_W = float(pm.query('MEAS:POW?'))
        power_mW = power_W * 1e3
        print(f"Power: {power_mW} mW")

        average_current += read_current
        average_power += power_mW

    average_current /= num_measurements
    average_power /= num_measurements

    power_current[0, idx] = average_current
    power_current[1, idx] = average_power

ni_daq.set_current(0) #Shut off laser after!

fig, ax = plt.subplots()

ax.plot(power_current[0, :], power_current[1, :])
ax.set_xlabel('Current [A]')
ax.set_ylabel('Power [mW]')
ax.set_title('Power Measurement')
plt.savefig('Power-Current_L830P.png')
plt.show()