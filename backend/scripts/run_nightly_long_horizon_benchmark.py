from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.engine import DaVinciDecisionEngine


def main() -> None:
    engine = DaVinciDecisionEngine()
    benchmark = engine.benchmark_long_horizon_nightly_suite()
    print(json.dumps(benchmark, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
