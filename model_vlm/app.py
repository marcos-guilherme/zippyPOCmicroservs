from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from scripts import read_text_from_txt
from tempfile import NamedTemporaryFile
from moondream_call import query_img
from ollama_call import llava_call
import os

app = FastAPI()

prompt_path = "sys_prompts/moondream_prompt.txt"

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

    answer = ""

    if model == "moondream":
        answer = query_img(temp_path, question)

    elif model == "ollama_llava":
        answer = llava_call(temp_path, question)

    os.remove(temp_path)

    return JSONResponse({"caption": answer})
