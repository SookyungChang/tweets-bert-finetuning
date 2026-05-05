FROM public.ecr.aws/lambda/python:3.10

# System dependencies for matplotlib, torch, and compiling packages
RUN yum install -y gcc g++ make libgomp git && \
    yum clean all

# Install CPU-only torch first (saves ~1GB vs CUDA build — Lambda has no GPU)
RUN pip install --no-cache-dir \
    torch==2.9.1+cpu \
    torchvision==0.24.1+cpu \
    torchaudio==2.9.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY app.py .

# Pre-bake the BERT model into the image
# so Lambda doesn't download it on every cold start
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sweetguma/bert-sentiment-model')"

# Lambda entrypoint
CMD ["app.handler"]

