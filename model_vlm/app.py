from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from scripts import read_text_from_txt
from tempfile import NamedTemporaryFile
from vlm.registry import get_model, list_models
import torch
import os
import time

app = FastAPI()

prompt_path = "sys_prompts/moondream_prompt.txt"

@app.get("/models")
async def models_endpoint():
    return {"models": list_models()}


@app.get("/devices")
async def devices_endpoint():
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    devices = []
    if cuda_available:
        for i in range(device_count):
            devices.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
            })
    return {
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "devices": devices,
    }


@app.post("/caption")
async def caption_endpoint(
    file: UploadFile = File(...),
    model: str = Form("moondream"),
    question: str = Form(None),
):
    
    if question is None:
        question = read_text_from_txt(prompt_path)
    
    with NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    adapter = get_model(model)
    t0 = time.time()
    result = adapter.infer(temp_path, question)
    latency_ms = int((time.time() - t0) * 1000)

    os.remove(temp_path)

    return JSONResponse({
        "caption": result.get("text"),
        "device": result.get("device"),
        "latency_ms": latency_ms,
        "model": model,
    })


@app.post("/warmup")
async def warmup_endpoint(model: str = Form("moondream")):
    adapter = get_model(model)
    # No-op warmup for ollama; for local models, ensure loaded
    return {"status": "ok", "model": model}
