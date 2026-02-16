"""Launch the Panel server as a subprocess and notify it of new scans."""

import platform
import subprocess
import sys
import time
import webbrowser

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
        # start "" /min opens minimized — not perfect but avoids full focus steal
        subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
    else:
        webbrowser.open(url)


def notify(serial: str, port: int = DEFAULT_PORT, focus: bool = False) -> str:
    """Open a new browser tab for the given scan.

    The tab triggers server-side rendering on load.
    Returns the URL for this scan.

    Args:
        serial: The scan serial number.
        port: Server port.
        focus: If True, steal focus to the browser. Default False.
    """
    url = f"http://localhost:{port}/explore?serial={serial}"
    if focus:
        webbrowser.open(url)
    else:
        _open_background(url)
    return url
