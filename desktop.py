"""Desktop launcher — run the local PDF Extractor as a one-click app.

Double-click **Start PDF Extractor.bat** (or run
``.venv\\Scripts\\python.exe desktop.py`` from this folder). It starts the local
server in the background and opens the app in a new **Chrome browser window**
(your normal browser — full address bar, zoom, bookmarks). The small launcher
window keeps the server running; close it (or press Ctrl+C) to stop the app.

Everything still runs locally — the browser is just pointed at the same server
you get at http://localhost:8000. No extra Python dependency is needed (only the
core requirements + a browser, which you already have).
"""

import os
import socket
import subprocess
import threading
import time
import urllib.request

import uvicorn

APP = "app:app"
TITLE = "Local PDF Extractor"

# Set if the background server thread dies (e.g. the port got taken in the
# bind race) so _wait_until_up can fail fast instead of polling for the full
# timeout.
_server_failed = threading.Event()


def _free_port(preferred: int = 8000) -> int:
    """Return an open localhost port — preferred if free, else an ephemeral one."""
    for candidate in (preferred, 0):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", candidate))
            return s.getsockname()[1]
        except OSError:
            continue
        finally:
            s.close()
    return preferred


def _serve(port: int) -> None:
    """Run uvicorn in this (background) thread.

    ``install_signal_handlers`` is disabled because uvicorn can only install
    them from the main thread.
    """
    config = uvicorn.Config(APP, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    try:
        server.run()
    except Exception as exc:  # e.g. the port was taken in the bind race
        print(f"Server thread failed: {exc}")
        _server_failed.set()


def _wait_until_up(url: str, timeout: float = 180.0) -> bool:
    """Poll the server until it answers (it only answers after model warmup)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_failed.is_set():
            return False  # the server thread already died — don't wait the full timeout
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.4)
    return False


def _find_chrome() -> str | None:
    """Path to Chrome (preferred) or Edge, or None for the default browser."""
    pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        os.path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(pf86, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(local, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(pf86, "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(pf, "Microsoft", "Edge", "Application", "msedge.exe"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _open_in_browser(url: str) -> None:
    """Open url in a new full Chrome/Edge window, else the default browser."""
    chrome = _find_chrome()
    if chrome:
        try:
            subprocess.Popen([chrome, "--new-window", url])
            return
        except Exception:
            pass
    import webbrowser
    webbrowser.open(url)


def main() -> None:
    port = _free_port(8000)
    url = f"http://127.0.0.1:{port}/"

    threading.Thread(target=_serve, args=(port,), daemon=True).start()
    print(f"Starting {TITLE} on {url} (loading models — first run is slower)...")

    if not _wait_until_up(url):
        print("The server did not start in time. Check the messages above.")
        input("Press Enter to close...")
        return

    _open_in_browser(url)

    bar = "=" * 60
    print("\n" + bar)
    print(f"  {TITLE} is running — it opened in your browser.")
    print("  KEEP THIS WINDOW OPEN while you use the app.")
    print("  Close this window (or press Ctrl+C) to stop the app.")
    print(bar + "\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
