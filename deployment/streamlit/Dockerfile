FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ make libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
ENV PIP_ROOT_USER_ACTION=ignore

# CPU-only torch
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
COPY deployment/streamlit/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY saved_models/ ./saved_models/
COPY deployment/streamlit/streamlit_app.py .

# HuggingFace cache
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface

# Streamlit config
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true        
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false  

EXPOSE 7860

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \