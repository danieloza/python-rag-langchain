# Senior RAG - Developer Automation

PYTHON=.\.venv\Scripts\python.exe
UVICORN=.\.venv\Scripts\uvicorn.exe

install:
	$(PYTHON) -m pip install -r requirements.txt

api:
	$(UVICORN) api.main:app --reload --port 8002

evaluate:
	$(PYTHON) src/evaluate.py

lint:
	.\.venv\Scriptsuff.exe check .
	.\.venv\Scripts\mypy.exe .
