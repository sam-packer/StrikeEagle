"""Network utilities for auto-detecting the local LAN IP."""

import socket


def get_lan_ip() -> str:
    """Return the LAN IP of this machine.

    Opens a UDP socket to a public address (without sending data) to let the
    OS choose the correct interface, then reads the local address.  Falls back
    to gethostbyname if that fails.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())
