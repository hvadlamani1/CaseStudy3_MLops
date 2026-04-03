import os
import time
import tempfile
import torch
import numpy as np
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app, Counter, Histogram

# --- Prometheus Metrics ---
BACKEND_LOCAL_REQUESTS = Counter("backend_local_requests_total", "Total requests to local backend")
BACKEND_LOCAL_LATENCY = Histogram("backend_local_request_duration_seconds", "Total latency of local request")
TRANSCRIPTION_LATENCY = Histogram("backend_local_transcription_duration_seconds", "Time taken for speech recognition")
TRANSLATION_LATENCY = Histogram("backend_local_translation_duration_seconds", "Time taken for text translation")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="ATC Local Inference API",
    description="Backend API executing PyTorch models locally.",
    version="1.0.0"
)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/health")
async def health():
    return {"status": "ok"}

# --- Local ML Globals ---
model = None
processor = None
atc_translator = None

def detect_device():
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32
    else:
        return "cpu", torch.float32

device, torch_dtype = detect_device()

def load_resources():
    global model, processor, atc_translator, device, torch_dtype
    if model is None:
        print("Loading Whisper model...")
        loaded_model = WhisperForConditionalGeneration.from_pretrained(
            "tclin/whisper-large-v3-turbo-atcosim-finetune",
            torch_dtype=torch_dtype
        )
        model = loaded_model.to(device)
        processor = WhisperProcessor.from_pretrained("tclin/whisper-large-v3-turbo-atcosim-finetune")

    if atc_translator is None:
        print("Loading Translator model...")
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto", 
            device_map="auto"
        )
        atc_translator = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

def atc_english_translation(atc_prompt):
    load_resources()
    messages = [
        {"role": "system", "content": "You are an aviation expert. Translate the following technical ATC radio transmission into simple, conversational plain English. Do not give definitions, just simply translate to conversational english! Be concise."},
        {"role": "user", "content": atc_prompt}
    ]
    prompt = atc_translator.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    outputs = atc_translator(prompt, do_sample=False, max_new_tokens=256, return_full_text=False)
    return outputs[0]['generated_text'].strip()

# --- API Endpoint ---
@app.post("/process_audio")
async def process_audio(audio_file: UploadFile = File(...)):
    BACKEND_LOCAL_REQUESTS.inc()
    request_start_time = time.time()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb") as temp_audio:
        content = await audio_file.read()
        temp_audio.write(content)
        temp_audio_path = temp_audio.name

    try:
        load_resources()
        
        t0 = time.time()
        with TRANSCRIPTION_LATENCY.time():
            speech, sample_rate = sf.read(temp_audio_path)
            speech = speech.astype(np.float32)
            if len(speech.shape) > 1:
                speech = np.mean(speech, axis=1)
            if sample_rate != 16000:
                speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)

            input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device=device, dtype=torch_dtype)

            generated_ids = model.generate(input_features, max_new_tokens=128, repetition_penalty=1.1)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        t1 = time.time()
        
        with TRANSLATION_LATENCY.time():
            translation = atc_english_translation(transcription)
        t2 = time.time()
        
        os.remove(temp_audio_path)

        BACKEND_LOCAL_LATENCY.observe(time.time() - request_start_time)
        return JSONResponse(content={
            "transcription": transcription,
            "translation": translation,
            "transcription_time_sec": round(float(t1 - t0), 2),
            "translation_time_sec": round(float(t2 - t1), 2)
        })

    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        BACKEND_LOCAL_LATENCY.observe(time.time() - request_start_time)
        raise HTTPException(status_code=500, detail=str(e))
