"""Launch the Panel server as a subprocess and communicate with it."""

import json
import platform
import subprocess
import sys
import time
import webbrowser
from urllib.request import urlopen

DEFAULT_PORT = 5006


def launch_server(port: int = DEFAULT_PORT) -> subprocess.Popen:
    """Start the Panel server as a subprocess.

    Returns the Popen handle — call .terminate() to stop it.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "explore.server"],
    )
    time.sleep(2)  # give the server time to start
    return proc


def _open_background(url: str) -> None:
    """Open a URL in the default browser without stealing focus."""
    if platform.system() == "Darwin":
        subprocess.Popen(["open", "-g", url])
    elif platform.system() == "Windows":
        subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
    else:
        webbrowser.open(url)


def notify(serial: str, port: int = DEFAULT_PORT, focus: bool = False) -> str:
    """Open a new browser tab for the given scan. Prefer Client.plot()."""
    url = f"http://localhost:{port}/explore?serial={serial}"
    if focus:
        webbrowser.open(url)
    else:
        _open_background(url)
    return url


class Client:
    """Client for interacting with a scan in the Panel server."""

    def __init__(self, serial: str, port: int = DEFAULT_PORT):
        self.serial = serial
        self.port = port

    def plot(self, focus: bool = False) -> str:
        """Open a browser tab for this scan. Returns the URL."""
        return notify(self.serial, port=self.port, focus=focus)

    def get_picks(self) -> list[dict]:
        """Query the server for picked points on this scan."""
        url = f"http://localhost:{self.port}/picks?serial={self.serial}"
        with urlopen(url) as resp:
            return json.loads(resp.read())

    def __repr__(self) -> str:
        return f"Client(serial='{self.serial}', port={self.port})"
