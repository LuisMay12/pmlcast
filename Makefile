PYTHON ?= /opt/homebrew/opt/python@3.12/bin/python3.12
VENV := .venv
BIN := $(VENV)/bin
PY := $(BIN)/python
NAME ?= stage1
START ?= 2022-01-01
CATALOG ?= data/catalog/nodes.parquet

.PHONY: venv compile install fmt lint test catalog collect-stage1 \
	collect-stage2-mda collect-stage2-mtr quality dataset baselines train \
	experiments daily score serve docker-build docker-run

venv:
	uv venv --python $(PYTHON) $(VENV)

compile:
	uv pip compile --universal requirements.in -o requirements.txt
	uv pip compile --universal requirements-dev.in -o requirements-dev.txt

install:
	uv pip install --python $(PY) -r requirements.txt -r requirements-dev.txt

fmt:
	$(BIN)/black pmlcast tests

lint:
	$(BIN)/black --check pmlcast tests
	$(BIN)/flake8 pmlcast tests

test:
	$(BIN)/pytest

catalog:
	$(PY) -m pmlcast.catalog --catalog "$(CATALOG)" --stage 1
	$(PY) -m pmlcast.catalog --catalog "$(CATALOG)" --stage 2

collect-stage1:
	$(PY) -m pmlcast.cenace --stage 1 --proceso MDA

collect-stage2-mda:
	$(PY) -m pmlcast.cenace --stage 2 --proceso MDA

collect-stage2-mtr:
	$(PY) -m pmlcast.cenace --stage 2 --proceso MTR --start 2022-01-01

quality:
	$(PY) -m pmlcast.quality

dataset:
	$(PY) -m pmlcast.dataset --name $(NAME) \
		--nodes-file data/snapshot/nodes_$(NAME).csv --start $(START)

baselines:
	$(PY) -m pmlcast.baselines --dataset $(NAME)

train:
	$(PY) -m pmlcast.train --dataset $(NAME) --model history

experiments:
	$(PY) -m pmlcast.experiments --dataset $(NAME)

daily:
	$(PY) -m pmlcast.daily

score:
	$(PY) -m pmlcast.daily --score-only

serve:
	$(PY) -m pmlcast.api

docker-build:
	docker build -t pmlcast:$(shell $(PY) -c "import pmlcast; print(pmlcast.__version__)") -t pmlcast:latest .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/data:/data pmlcast:latest
