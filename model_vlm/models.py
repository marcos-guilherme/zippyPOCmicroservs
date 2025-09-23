from pydantic import BaseModel
from fastapi import File, UploadFile, Form


# Uso Posterior
class ImageInput(BaseModel):
    file: UploadFile = (File(...),)
    question: str = Form("Describe this image")
