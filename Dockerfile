FROM python:3.10-slim

WORKDIR /app

# system deps (minimum requirement)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# FastAPI
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]