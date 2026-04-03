FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl prometheus-node-exporter && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight backend_api deps + Heavy backend_local PyTorch CPU deps
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi uvicorn huggingface_hub prometheus-client python-multipart transformers accelerate soundfile librosa

# Copy all backend scripts
COPY . /app/

EXPOSE 9001 9002 9100

# Default CMD (can be overridden by compose for different services)
CMD ["sh", "-c", "prometheus-node-exporter & uvicorn backend_api:app --host 0.0.0.0 --port 9001"]
