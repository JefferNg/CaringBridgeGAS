FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py index.html ./

EXPOSE 8080

CMD exec uvicorn server:app --host 0.0.0.0 --port ${PORT}
