.PHONY: install test eval eval-live record

install:
	test -d .venv || python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

test:
	.venv/bin/pytest -q

eval:
	.venv/bin/python eval.py --mode replay

eval-live:
	.venv/bin/python eval.py --mode live

record:
	.venv/bin/python eval.py --mode record
