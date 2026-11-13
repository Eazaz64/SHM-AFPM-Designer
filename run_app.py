"""Launcher used by PyInstaller and for `python run_app.py`."""
import sys
from afpm_shm.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
