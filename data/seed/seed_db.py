"""Create data/ticketpilot.db with deterministic mock data (`make seed`)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ticketpilot.config import get_settings
from ticketpilot.seed import seed_database


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=settings.business_db_path)
    args = parser.parse_args()
    scenarios = Path(__file__).with_name("scenarios.yaml")
    seed_database(args.db, settings.REFERENCE_DATE, scenarios_path=scenarios)
    print(f"Seeded {args.db} (reference date {settings.REFERENCE_DATE}); scenarios → {scenarios}")


if __name__ == "__main__":
    main()
