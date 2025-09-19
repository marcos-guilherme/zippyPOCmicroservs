from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import httpx
import os
import shutil
import uuid

app = FastAPI()

VISUAL_SERVICE_URL = os.getenv("VISUAL_SERVICE_URL", "http://servico_visual:7000")

@app.post("/analisar_upload")
async def analisar_upload_imagem(
    acao: str = Form(...),  
    file: UploadFile = File(...) # 'file' é o arquivo de imagem
):
    """
    Recebe um arquivo de imagem e uma ação, e redireciona para o serviço visual.
    """
    try:
        file_extension = file.filename.split('.')[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = f"/tmp/{unique_filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)


        async with httpx.AsyncClient() as client:
            async with file.file as f:
                response = await client.post(
                    f"{VISUAL_SERVICE_URL}/processar_imagem",
                    headers={"Content-Type": file.content_type},
                    params={"acao": acao}, # Envia a ação como parâmetro na URL
                    content=f
                )
            response.raise_for_status()

        return response.json()

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Erro no serviço visual: {exc.response.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno do servidor: {str(e)}"
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)