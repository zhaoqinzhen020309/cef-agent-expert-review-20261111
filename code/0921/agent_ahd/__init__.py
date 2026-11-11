"""Import the source tree when this repository is checked out on its own.

The historical distribution lived inside a parent ``agent_ahd`` directory.
Keeping the source in place also preserves existing experiment paths.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent)]
