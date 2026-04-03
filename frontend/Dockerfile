FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl prometheus-node-exporter && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies required by frontend_ui.py
RUN pip install --no-cache-dir gradio requests prometheus-client

# Copy the frontend UI script
COPY frontend_ui.py /app/

# Expose the Gradio App port (7001), Prometheus metrics port (8001), and Node Exporter (9100)
EXPOSE 7001 8001 9100

# Run the node exporter in background and then frontend
CMD ["sh", "-c", "prometheus-node-exporter & python frontend_ui.py"]
