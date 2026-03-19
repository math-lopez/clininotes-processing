FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY . .

CMD python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT