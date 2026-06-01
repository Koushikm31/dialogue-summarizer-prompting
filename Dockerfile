FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
COPY src/ src/
COPY deployment/ deployment/

RUN pip install --no-cache-dir -r requirements.txt

CMD exec uvicorn deployment.app:app --host 0.0.0.0 --port ${PORT:-8080}