"""Servers are spawned as scripts (python mcp_servers/x.py); put the repo root on sys.path
so they can import the app package."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
