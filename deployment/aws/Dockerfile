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

COPY deployment/aws/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY saved_models/ ./saved_models/
COPY deployment/aws/app.py .

ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sweetguma/bert-sentiment-model')"


EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
