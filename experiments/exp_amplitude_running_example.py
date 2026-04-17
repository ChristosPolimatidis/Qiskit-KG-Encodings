from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.main import main


if __name__ == "__main__":
    main(
        [
            "--encoding",
            "amplitude",
            "--input",
            "data/running_example.ttl",
            "--weights",
            "2,1,3,1,1,2",
        ]
    )
