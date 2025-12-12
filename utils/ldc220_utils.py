import time
import nidaqmx
from nidaqmx.constants import TerminalConfiguration
import pyvisa


Imax = 2000  # mA
zero_set = 1 #mA

def set_voltage(volts):
    assert volts >= 0 and (volts / 10.0) * Imax < 75, f'Voltage must be between 0 and {(75 / Imax) * 10}'
    with nidaqmx.Task() as task:
        task.ao_channels.add_ao_voltage_chan("Dev1/ao0", min_val=-10.0, max_val=10.0)
        task.write(volts)

def set_current(mA):
    assert mA >= 0 and mA <= 75, f'Current must be between 0 and 75'
    mA -= zero_set
    volts = (mA / Imax) * 10.0
    set_voltage(volts)


def read_volts():
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=-10.0, max_val=10.0,
                                             terminal_config=TerminalConfiguration.RSE
                                             )
        volts = task.read()
    return volts

def read_current():
    volts = read_volts()
    mA = (volts / 10.0) * Imax
    return mA

def read_current_zero():
    volts_tmp = read_volts()
    set_voltage(0)
    volts_zero = read_volts()
    print(volts_tmp)
    set_voltage(volts_tmp - volts_zero)
    return round((abs(volts_tmp - volts_zero) / 10.0) * Imax + zero_set)

#31 for pulse, #26 for square
rm = pyvisa.ResourceManager()
sdg = rm.open_resource('USB0::0xF4ED::0xEE3A::SDG050D2150058::INSTR')

sdg.write('*IDN?')
sdg.write('C1:OUTP ON')
#sdg.write('C1:BSWV WVTP,SINE')

#sdg.write('OUTP1 OFF')
#print(sdg.read())