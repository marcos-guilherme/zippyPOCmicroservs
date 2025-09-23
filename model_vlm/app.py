from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
from moondream_call import caption_img
from ollama_call import llava_call


app = FastAPI()


@app.post("/caption")
async def caption_endpoint(
    file: UploadFile = File(...),
    model: str = Form("moondream"),
    question: str = Form("What is the main subject of this image?"),
):
    with NamedTemporaryFile(delete=False, suffix=".png") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    if model == "moondream":
        answer = caption_img(temp_path)

    if model == "ollama_llava":
        prompt = f"Describe the image in detail."
        answer = llava_call(temp_path, prompt)

    return JSONResponse({"caption": answer})
