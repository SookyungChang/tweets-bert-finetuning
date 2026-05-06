FROM public.ecr.aws/lambda/python:3.10

# System dependencies for matplotlib, torch, and compiling packages
RUN yum install -y gcc g++ make libgomp git && \
    yum clean all

# Install CPU-only torch first (saves ~1GB vs CUDA build — Lambda has no GPU)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio\
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY deployment/aws/app.py .


RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='sweetguma/bert-sentiment-model')"

# Lambda entrypoint
CMD ["app.handler"]

