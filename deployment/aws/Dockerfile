FROM public.ecr.aws/lambda/python:3.10

# System dependencies for matplotlib, torch, and compiling packages
RUN yum install -y gcc g++ make libgomp git && \
    yum clean all

# Install CPU-only torch first (saves ~1GB vs CUDA build — Lambda has no GPU)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio\
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
# Note: Ensure requirements.txt is inside deployment/aws/
COPY deployment/aws/requirements.txt .
RUN pip install --no-cache-dir -v -r requirements.txt

# 4. Set Environment Variables for Hugging Face
# Lambda's filesystem is read-only except for /tmp. 
# We must tell HF to download the model into the image during build.
ENV TRANSFORMERS_CACHE=/var/task/model_cache
ENV HF_HOME=/var/task/model_cache

# Copy source code
COPY src/ ./src/
COPY deployment/aws/app.py .


RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='sweetguma/bert-sentiment-model')"

# Lambda entrypoint
CMD ["app.handler"]

