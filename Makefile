PYTHON ?= /opt/homebrew/opt/python@3.12/bin/python3.12
VENV := .venv
BIN := $(VENV)/bin
PY := $(BIN)/python
NAME ?= stage1
START ?= 2022-01-01
CATALOG ?= data/catalog/nodes.parquet
NODES ?= data/snapshot/nodes_$(NAME).csv

.PHONY: venv compile install fmt lint test catalog collect-stage1 \
	collect-stage2-mda collect-stage2-mtr quality dataset baselines train \
	experiments daily score serve docker-build docker-run catalog-snapshot \
	dataset-final baselines-final train-final onnx demo-data deploy-image \
	deploy-run

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
		--nodes-file $(NODES) --start $(START)

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

# --- Reproduce the shipped model -----------------------------------------
# The generic targets above keep the project's development defaults (Huber,
# a 7-day window). These pin the exact settings of the published model,
# read back from its MLflow run: 99 SIN nodes, 14-day window, data from
# 2019 to 2026-09-19, LSTM 64 units, dropout 0.2, MAE loss, patience 6.

catalog-snapshot:
	mkdir -p data/catalog
	cp data/snapshot/catalog_v20260218.parquet data/catalog/nodes.parquet

dataset-final:
	$(PY) -m pmlcast.dataset --name final \
		--nodes-file data/snapshot/nodes_stage2.csv \
		--start 2019-01-01 --end 2026-09-19 --input-days 14

baselines-final:
	$(PY) -m pmlcast.baselines --dataset final

train-final:
	$(PY) -m pmlcast.train --dataset final --model history --units 64 \
		--dropout 0.2 --loss mae --patience 6 --seed 0

onnx:
	PYTHONPATH=. $(PY) scripts/export_onnx.py

# --- Public demo -------------------------------------------------------------

demo-data:
	PYTHONPATH=. $(PY) scripts/build_demo_data.py

deploy-image:
	docker build -f deploy/Dockerfile -t pmlcast-demo .

deploy-run:
	docker run --rm -p 7860:7860 pmlcast-demo
