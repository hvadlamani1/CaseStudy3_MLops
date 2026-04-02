FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies required by backend_api.py
RUN pip install --no-cache-dir fastapi uvicorn huggingface_hub prometheus-client python-multipart

# Copy the specific backend API script
COPY backend_api.py /app/

# Expose the API port based on prometheus.yml config
EXPOSE 9001

# Run uvicorn on port 9001
CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "9001"]
