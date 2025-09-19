from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
import httpx
import os

app = FastAPI()

# URLs dos serviços de modelo
URL_CLASSIFICACAO = os.getenv("URL_CLASSIFICACAO", "http://modelo_classificacao:5001")

@app.post("/processar_imagem")
async def processar_imagem(
    acao: str, # Ação agora vem como um parâmetro da URL
    file: UploadFile = File(...) # O arquivo é recebido aqui
):
    """
    Recebe um arquivo de imagem e a roteia para o serviço de modelo apropriado.
    """
    try:
        # Pega os bytes do arquivo para enviar ao modelo
        file_bytes = await file.read()

        async with httpx.AsyncClient() as client:
            if acao == "classificar":
                response = await client.post(
                    f"{URL_CLASSIFICACAO}/classificar",
                    files={"file": (file.filename, file_bytes, file.content_type)}
                )
            else:
                raise HTTPException(status_code=400, detail="Ação desconhecida")

            response.raise_for_status()
            return response.json()

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Erro no serviço de modelo: {exc.response.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno do servidor: {str(e)}"
        )