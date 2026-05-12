#!/usr/bin/env python3
"""Compatibility entrypoint for running the local source tree directly."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PACKAGE = SRC / "qwen_proxy"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

__path__ = [str(PACKAGE)]
__version__ = "0.1.0"


def main():
    from qwen_proxy.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
