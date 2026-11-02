UV ?= uv
RUN := $(UV) run
SPLIT ?= dev

.PHONY: setup lint format test cov seed index tickets split review-labels retrieval-eval \
        eval-dev eval-test sweep api ui up down smoke

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
	$(RUN) pytest

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

api:
	$(RUN) uvicorn ticketpilot.api.main:app --host 0.0.0.0 --port 8000

ui:
	$(RUN) streamlit run src/ticketpilot/review_ui/app.py --server.port 8501

up:
	docker compose up --build -d

down:
	docker compose down
