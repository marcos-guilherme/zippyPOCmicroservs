from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import httpx
import os
import shutil
import uuid

app = FastAPI()

# O nome do serviço no docker-compose é o "hostname"
VISUAL_SERVICE_URL = os.getenv("VISUAL_SERVICE_URL", "http://servico_visual:7000")

@app.post("/analisar_upload")
async def analisar_upload_imagem(
    acao: str = Form(...),  # 'acao' agora vem como um campo de formulário
    file: UploadFile = File(...) # 'file' é o arquivo de imagem
):
    """
    Recebe um arquivo de imagem e uma ação, e redireciona para o serviço visual.
    """
    try:
        # Crie um nome de arquivo único para evitar colisões
        file_extension = file.filename.split('.')[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = f"/tmp/{unique_filename}"

        # Salve o arquivo temporariamente no contêiner
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Crie a URL de acesso. Nesse caso, estamos usando o mesmo endpoint
        # para simular o acesso à imagem. Em produção, seria uma URL de S3.
        # Por simplicidade, vamos usar o caminho do arquivo temporário.
        # Em um cenário real, o cliente visual não teria acesso a essa pasta!
        # Por isso, o padrão de uma URL externa (S3, CDN) é muito mais seguro.
        # Vamos simular o envio do arquivo via bytes. Isso é mais seguro.

        async with httpx.AsyncClient() as client:
            # Envie o arquivo diretamente para o serviço visual, como um stream
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
        # Limpe o arquivo temporário, se ele existir
        if os.path.exists(file_path):
            os.remove(file_path)