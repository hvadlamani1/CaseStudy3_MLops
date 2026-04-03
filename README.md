# Project Monitoring Setup

This project successfully integrates Prometheus for monitoring both the Backend and Frontend components, strictly adhering to the assignment requirements.

## 1. Exposed Python-level Metrics
Both products utilize the official `prometheus_client` library to expose internal operational data instead of relying solely on external or Docker-level metrics.

## 2. Meaningful Custom Metrics (4-10 per product)

### **Frontend Metrics (`frontend_ui.py`)**
1. `frontend_requests_total` (Counter): The total number of incoming transcription requests initiated through the UI.
2. `frontend_request_duration_seconds` (Histogram): Captures the complete round-trip latency for the frontend to submit an audio file and retrieve the transcription/translation.
3. `frontend_backend_errors_total` (Counter): Tracks upstream API connection breakdowns and Server-side (500) errors.
4. `frontend_exceptions_total` (Counter): Catches arbitrary Python runtime or Gradio UI exceptions.

### **Backend Metrics (`backend_api.py` & `backend_local.py`)**
1. `backend_requests_total` (Counter): Tracks the sheer volume of transcription inference tasks.
2. `backend_request_duration_seconds` (Histogram): Total end-to-end processing time for an audio payload.
3. `backend_transcription_duration_seconds` (Histogram): The isolated compute latency of the Whisper Automatic Speech Recognition model.
4. `backend_translation_duration_seconds` (Histogram): The isolated compute latency of the LLM generating conversational translation text.

## 3. Metrics Port Availability
- **Frontend**: The `start_http_server(8001)` function spins up a background thread, reliably exposing the metrics on port `8001`.
- **Backend**: FastAPI natively exposes metrics on `/metrics` directly on the primary host port (`9001` for API, `9002` for Local) utilizing the `make_asgi_app()` method.

## 4. Setup Process for Python Prometheus Client
To ensure proper integration and setup for local testing or Docker execution, the Prometheus implementation followed these steps:

1. **Dependency Installation**: `prometheus-client` was explicitly added to all build layers alongside FastAPI and Gradio.
2. **Metric Definition**: Counters and Histograms were declared as global objects at the head of each script to prevent duplicate registry collision.
3. **Execution Tracking**: Using context managers (e.g. `with TRANSCRIPTION_LATENCY.time():`) and `time.time()` math allowed precise observation of inference latency without slowing down execution.
4. **Scraping Configuration**: `run/prometheus.yml` formally links the deployed Prometheus service to scrape these Python endpoints dynamically inside the Docker `group1_net` bridge network:
    ```yaml
      - targets: ['group1_frontend:8001']
      - targets: ['group1_backend:9001']
      - targets: ['group1_backend_local:9002']
    ```

## 5. Docker Container Monitoring (Node Exporter)
To observe hardware resource consumption (CPU, Memory, Disk) across the dockerized system:
1. **Apt Installation**: The Debian package `prometheus-node-exporter` is natively installed natively on `frontend/Dockerfile` and `backend/Dockerfile` during the build phase (`apt-get install -y prometheus-node-exporter`).
2. **Background Daemonizing**: Standard Docker images typically only execute a single process. I overrode the `CMD` argument in both Dockerfiles/`compose.yml` heavily using `sh -c "prometheus-node-exporter & ..."`! This successfully spawns the Node Exporter as a background daemon listening on its default port (`9100`) while preserving the primary Python web app in the foreground.
3. **Scraping Availability**: Target paths automatically resolve the background daemon on `frontend:9100`, `backend:9100`, and `backend_local:9100` via Docker's internal DNS inside the configured Prometheus scrape jobs.
