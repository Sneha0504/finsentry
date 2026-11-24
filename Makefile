# FinSentry - one-command reproduce.
#
# IMPORTANT: this project lives under a path containing a SPACE
# ("Coding Projects"). Every recipe quotes its paths so the targets work
# regardless. Do not unquote $(VENV_PY) / $(ROOT).
#
# Usage:
#   make setup       # create .venv, install pinned deps + the package (editable)
#   make reproduce   # the ONE command: clean -> full pipeline -> committed figures
#   make figures     # regenerate the committed figures into assets/
#   make streamlit   # launch the local Streamlit app
#   make test        # dbt data tests only
#   make clean       # remove generated data/reports/dbt artifacts (keeps .venv)

# Absolute repo root. $(CURDIR) holds the path as a SINGLE value (unlike
# $(MAKEFILE_LIST), which make word-splits and so mangles the space in
# "Coding Projects"). Run make from the repo root (the documented usage).
ROOT := $(CURDIR)
VENV := $(ROOT)/.venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
DBT := $(VENV)/bin/dbt
DBT_PROJ := $(ROOT)/dbt/finsentry

# Interpreter used to CREATE the venv. Must be Python 3.11 or 3.12 (numpy 1.26.4
# has no cp313 wheels; pandas 2.2.3 needs >=3.9). Override if `python3` is older:
#   make setup PYTHON=python3.12
PYTHON ?= python3

# Run dbt from the repo root with the project's own profiles (DuckDB path is
# relative to cwd); export an absolute DuckDB path so it is cwd-independent.
export FINSENTRY_DUCKDB := $(ROOT)/data/warehouse.duckdb

.DEFAULT_GOAL := help
.PHONY: help setup reproduce figures streamlit test clean

help:
	@echo "FinSentry targets:"
	@echo "  make setup       create .venv and install pinned deps + package"
	@echo "  make reproduce   full pipeline end-to-end, then commit figures to assets/"
	@echo "  make figures     regenerate committed figures into assets/"
	@echo "  make streamlit   launch the local Streamlit app"
	@echo "  make test        run dbt data tests"
	@echo "  make clean       remove generated artifacts (keeps .venv)"

# --- Environment -----------------------------------------------------------
setup:
	@echo ">> using interpreter: $(PYTHON)"
	@$(PYTHON) -c 'import sys; v=sys.version_info; assert (3,11)<=(v.major,v.minor)<=(3,12), \
		f"FinSentry needs Python 3.11 or 3.12 (numpy==1.26.4 / pandas==2.2.3); got {v.major}.{v.minor}. \
Re-run: make setup PYTHON=python3.12"' \
		|| { echo "!! wrong Python version (see above). Aborting."; exit 1; }
	@echo ">> creating venv at \"$(VENV)\""
	$(PYTHON) -m venv "$(VENV)"
	"$(VENV_PIP)" install --upgrade pip
	"$(VENV_PIP)" install -r "$(ROOT)/requirements.txt"
	"$(VENV_PIP)" install -e "$(ROOT)"
	@echo ">> setup complete. Next: make reproduce"

# --- The one-command reproduce ---------------------------------------------
# clean removes prior generated state so this is a from-clean run; the pipeline
# runs all 8 stages (generate -> load -> dbt run -> dbt test -> dq_monitoring ->
# weekly_review -> verify_numbers -> eval) and must exit 0; then the four real
# figures are promoted into the committed assets/ directory.
reproduce: clean
	@echo ">> running the full FinSentry pipeline (real run; fixed seed)"
	cd "$(ROOT)" && "$(VENV_PY)" -m finsentry.pipeline_runner
	@$(MAKE) --no-print-directory figures
	@echo ">> reproduce complete: data/warehouse.duckdb, reports/*, assets/*.png"

# --- Figures ---------------------------------------------------------------
# Copies the four headline PNGs produced by the pipeline into the tracked
# assets/ dir. Requires a prior run (reproduce runs the pipeline first); if the
# warehouse/figures are missing, the script prints how to build them.
figures:
	@echo ">> promoting real figures into assets/"
	cd "$(ROOT)" && "$(VENV_PY)" "$(ROOT)/scripts/copy_figures.py"

# --- Streamlit app ---------------------------------------------------------
streamlit:
	@echo ">> launching Streamlit (Ctrl-C to stop). http://localhost:8501"
	cd "$(ROOT)" && "$(VENV)/bin/streamlit" run "$(ROOT)/streamlit/app.py"

# --- dbt tests only --------------------------------------------------------
test:
	cd "$(ROOT)" && "$(DBT)" test --no-use-colors \
		--project-dir "$(DBT_PROJ)" --profiles-dir "$(DBT_PROJ)"

# --- Clean (keeps .venv and committed assets/) -----------------------------
clean:
	@echo ">> cleaning generated artifacts (keeping .venv and assets/)"
	rm -f  "$(ROOT)/data/warehouse.duckdb" "$(ROOT)/data/warehouse.duckdb.wal"
	rm -f  "$(ROOT)/data/raw/"*.csv
	rm -rf "$(ROOT)/reports/analyst"
	rm -f  "$(ROOT)/reports/"*.png "$(ROOT)/reports/"*.json "$(ROOT)/reports/"*.md \
	       "$(ROOT)/reports/"*.csv "$(ROOT)/reports/"*.jsonl
	rm -rf "$(DBT_PROJ)/target" "$(DBT_PROJ)/dbt_packages" "$(DBT_PROJ)/logs"
