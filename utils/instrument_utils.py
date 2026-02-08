import nidaqmx
from nidaqmx.constants import TerminalConfiguration
import pyvisa



class Instrument:
    def __init__(
        self,
        com_port: str | None = None,
        ip_address: str | None = None,
        tcp_port: int | None = None,
        usb_port: str | None = None,
        **kwargs,
    ) -> None:

        if com_port is not None and ip_address is None and tcp_port is None:
            # Serial (COM)
            port_number = "".join(s for s in com_port if s.isdigit())
            resource_name = f"ASRL{port_number}::INSTR"

        elif ip_address is not None and com_port is None:
            # TCP/IP
            if tcp_port is None:
                resource_name = f"TCPIP0::{ip_address}::inst0::INSTR"
            else:
                resource_name = f"TCPIP0::{ip_address}::{tcp_port}::SOCKET"

        elif usb_port is not None:
            # USB (explicit VISA resource string)
            resource_name = usb_port

        else:
            raise ValueError(
                "Invalid arguments: provide either "
                "com_port, ip_address (+ optional tcp_port), or usb_port."
            )

        self._instrument = pyvisa.ResourceManager().open_resource(
            resource_name,
            write_termination="\n",
            read_termination="\n",
            **kwargs,
        )

        self._check_connection()

    def _check_connection(self):
        idn = self.query("*IDN?")
        if idn:
            self._idn = idn.strip()
            print(f"Connected to {self._idn}.")
        else:
            self.disconnect()
            raise RuntimeError("Instrument could not be identified.")

    def write(self, command: str) -> None:
        self._instrument.write(command)

    def query(self, command: str) -> str:
        return self._instrument.query(command)

    def disconnect(self) -> None:
        self._instrument.close()


class NIDAQ_LDC220:
    def __init__(
        self,
        Imax: int = 2000,
        zero_set: int = 1,
        max_amps: int = 75
    ):
        self.Imax = Imax
        self.zero_set = zero_set
        self.max_amps = max_amps

    def set_voltage(self, volts):
        volts = max(0, volts)
        assert volts >= 0 and (volts / 10.0) * self.Imax < self.max_amps, f'Voltage must be between 0 and {(self.max_amps / self.Imax) * 10}'
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan("Dev1/ao0", min_val=-10.0, max_val=10.0)
            task.write(volts)

    def set_current(self, mA):
        assert mA >= 0 and mA <= self.max_amps, f'Current must be between 0 and 75'
        mA -= self.zero_set
        volts = (mA / self.Imax) * 10.0
        self.set_voltage(volts)


    def read_volts(self):
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=-10.0, max_val=10.0,
                                                 terminal_config=TerminalConfiguration.RSE
                                                 )
            volts = task.read()
        return volts

    def read_current(self):
        volts = self.read_volts()
        mA = (volts / 10.0) * self.Imax
        return mA

    def read_current_zero(self):
        volts_tmp = self.read_volts()
        self.set_voltage(0)
        volts_zero = self.read_volts()
        self.set_voltage(volts_tmp - volts_zero)
        return round((abs(volts_tmp - volts_zero) / 10.0) * self.Imax + self.zero_set, 5)

class SDG5162_GATED_PROJECT:
    def __init__(self,
                com_port: str | None = None,
                ip_address: str | None = None,
                tcp_port: int | None = None,
                usb_port: str | None = None,
                 ):
        self.sdg = Instrument(
            usb_port=usb_port,
            com_port=com_port,
            ip_address=ip_address,
            tcp_port=tcp_port,
        )

    def find_gauss_index(self, duty):
        qry =  self.sdg.query('STL?').split(', ')
        string_to_find = f'GAUSS{duty:.0f}DUTY'
        try:
            qry_index = qry.index(string_to_find)
        except ValueError:
            raise ValueError("Could not find '" + string_to_find + "' in instrument.")
        index = qry[qry_index - 1][1:]
        return index

    def set_gaussian(self, duty, rep_rate, high_level, low_level, phase, edge):
        #print(self.sdg.query("STL?"))
        idx = self.find_gauss_index(duty)
        self.sdg.write(f"C1:ARWV INDEX,{idx}")
        self.sdg.write(f"C1:BSWV HLEV,{high_level}")
        self.sdg.write(f"C1:BSWV LLEV,{low_level}")
        self.sdg.write(f"C1:BSWV PHSE,{phase}")
        self.sdg.write(f"C1:BSWV FRQ,{rep_rate}")
        self.sdg.write("C1:OUTP PLRT,INVT")

    def set_trigger(self, rep_rate):
        self.sdg.write(f"C2:BSWV WVTP,PULSE")
        self.sdg.write(f"C2:BSWV AMP,.7")
        self.sdg.write(f"C2:BSWV WIDTH,12e-9")
        self.sdg.write("C2:BSWV RISE,6E-9")
        self.sdg.write("C2:BSWV FALL,6E-9")
        self.sdg.write(f"C2:BSWV FRQ,{rep_rate}")

    def set_square(self, duty, rep_rate, high_level, low_level, phase, edge):
        self.sdg.write(f"C1:BSWV WVTP,SQUARE")
        self.sdg.write(f"C1:BSWV HLEV,{high_level}")
        self.sdg.write(f"C1:BSWV LLEV,{low_level}")
        self.sdg.write(f"C1:BSWV DUTY,{duty}")
        self.sdg.write(f"C1:BSWV PHSE,{phase}")
        self.sdg.write(f"C1:BSWV FRQ,{rep_rate}")
        self.sdg.write("C1:OUTP PLRT,INVT")

    def set_pulse(self, duty, rep_rate, high_level, low_level, phase, edge):
        self.sdg.write(f"C1:BSWV WVTP,PULSE")
        self.sdg.write(f"C1:BSWV HLEV,{high_level}")
        self.sdg.write(f"C1:BSWV LLEV,{low_level}")
        self.sdg.write(f"C1:BSWV DUTY,{duty}")
        self.sdg.write(f"C1:BSWV RISE,{edge}")
        self.sdg.write(f"C1:BSWV FALL,{edge}")
        self.sdg.write(f"C1:BSWV FRQ,{rep_rate}")
        self.sdg.write("C1:OUTP PLRT,INVT")


    def turn_channel_on(self, channel):
        self.sdg.write(f"C{channel}:OUTP ON")

    def turn_channel_off(self, channel):
        self.sdg.write(f"C{channel}:OUTP OFF")

    def turn_both_channels_on(self):
        self.turn_channel_on(0)
        self.turn_channel_on(1)

    def turn_both_channels_off(self):
        self.turn_channel_off(0)
        self.turn_channel_off(1)
        self.turn_channel_off(0)
        self.turn_channel_off(1)

    def read_parameters(self, channel):
        resp =  self.sdg.query(f'C{channel}:BSWV?')
        parts = resp.split(' ', 1)[1].split(',')  # remove "C1:BSWV "
        d = {}
        for k, v in zip(parts[::2], parts[1::2]):
            try:
                d[k] = float(''.join(c for c in v if c in '0123456789+-.eE'))
            except:
                d[k] = v
        return d

    def set_waveform(self, type,  duty, rep_rate,high_level_amplitude, low_level_amplitude,  phase, edge):
        try:
            func = getattr(self, f"set_{type}")
        except AttributeError:
            raise ValueError(f"Unsupported illumination: {type}")
        func(duty, rep_rate, high_level_amplitude, low_level_amplitude, phase, edge)

    def set_waveform_and_trigger(self, type, duty, rep_rate, high_level_amplitude, low_level_amplitude, phase, edge):
        self.set_waveform(type, duty, rep_rate, high_level_amplitude, low_level_amplitude, phase, edge)
        self.set_trigger(rep_rate)

sdg = SDG5162_GATED_PROJECT(
    usb_port="USB0::0xF4ED::0xEE3A::SDG050D2150058::INSTR"
)

sdg.set_square(duty=20, rep_rate=5*1e6, high_level=4, low_level=-4, phase=20, edge=None)