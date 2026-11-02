UV ?= uv
RUN := $(UV) run
SPLIT ?= dev

.PHONY: setup lint format test cov seed index tickets split review-labels retrieval-eval \
        eval-dev eval-test sweep ablations calibration-export calibration-score time-manual \
        gate readme-results api ui up down

setup:
	$(UV) sync --all-extras --frozen
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example — add ANTHROPIC_API_KEY")
	-$(RUN) pre-commit install

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy src

format:
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

test:
	$(RUN) pytest --cov --cov-report=term:skip-covered --cov-fail-under=85

cov:
	$(RUN) pytest --cov --cov-report=term-missing

seed:
	$(RUN) python data/seed/seed_db.py

index:
	$(RUN) python scripts/index_kb.py

tickets:
	$(RUN) python scripts/build_tickets.py

split:
	$(RUN) python scripts/split_eval.py

review-labels:
	$(RUN) python scripts/review_labels.py

retrieval-eval:
	$(RUN) python -m evals.retrieval_eval --split $(SPLIT)

eval-dev:
	$(RUN) python -m evals.harness --split dev

eval-test:
	$(RUN) python -m evals.harness --split test --i-am-reporting

# Latest real (non-dry-run) report directory for the given split
LATEST_RUN = $(shell ls -d evals/reports/*-$(SPLIT) 2>/dev/null | sort | tail -1)

sweep:
	$(RUN) python -m evals.sweep --run $(LATEST_RUN)

ablations:
	$(RUN) python -m evals.ablations --split dev

calibration-export:
	$(RUN) python -m evals.calibration export --run $(LATEST_RUN) --n 50

calibration-score:
	$(RUN) python -m evals.calibration score

time-manual:
	$(RUN) python scripts/time_manual.py

gate:
	$(RUN) python -m evals.gate --current $(LATEST_RUN)/metrics.json

readme-results:
	$(RUN) python scripts/update_readme_results.py

api:
	$(RUN) uvicorn ticketpilot.api.main:app --host 0.0.0.0 --port 8000

ui:
	$(RUN) streamlit run src/ticketpilot/review_ui/app.py --server.port 8501

up:
	docker compose up --build -d

down:
	docker compose down
