FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ make libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY deployment/huggingface/app.py .

EXPOSE 7860                     # ← HF Spaces requires this port

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]