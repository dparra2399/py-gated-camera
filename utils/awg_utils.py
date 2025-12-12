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


sdg = Instrument(
    usb_port="USB0::0xF4ED::0xEE3A::SDG050D2150058::INSTR"
)


print(sdg.query('MMEM:CAT? "C:/"'))