
***

## API de Modelos Vision-Language

Esta API expõe modelos Vision-Language para captioning e classificação de imagens, com endpoints para consulta de modelos, dispositivos, geração de captions e warmup.

### Como iniciar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Use o docker-compose para subir os serviços:
   ```bash
   docker-compose up --build
   ```
   - O app será exposto em `localhost:8005`
   - Ollama e ngrok já estão configurados no compose.

***

## Endpoints

### Listar modelos disponíveis

- **GET `/models`**
- Retorna os nomes dos modelos registrados:

```json
{
  "models": ["moondream", "ollama_llava", "blip", "unsloth_vision", "pixtral"]
}
```

***

### Listar dispositivos CUDA disponíveis

- **GET `/devices`**
- Retorna informações sobre GPUs CUDA disponíveis:

```json
{
  "cuda_available": true,
  "cuda_device_count": 2,
  "devices": [
    {"index": 0, "name": "NVIDIA RTX 3080"},
    {"index": 1, "name": "NVIDIA RTX 4090"}
  ]
}
```

***

### Gerar caption/classificar imagem

- **POST `/caption`**
- Envia uma imagem para geração de legenda ou classificação com prompt personalizado ou padrão.

**Parâmetros:**
- `file` (form-data, obrigatório): arquivo de imagem.
- `model` (form-data, opcional): nome do modelo. Default: `"moondream"`.
- `question` (form-data, opcional): prompt a ser usado. Default: prompt de classificação parental.

**Resposta:**
```json
{
  "caption": "safe",
  "device": "cuda",
  "latency_ms": 832,
  "model": "moondream"
}
```

#### Exemplo de uso (cURL):

```bash
curl -X POST "http://localhost:8005/caption" \
  -F "file=@minha_imagem.png" \
  -F "model=blip" \
  -F "question=Descreva a imagem."
```

***

### Realizar warmup de modelo

- **POST `/warmup`**
- Inicializa/prepara o modelo especificado (carregamento, cache, etc).

**Parâmetros:**
- `model` (form-data, opcional): nome do modelo. Default: `"moondream"`. (No momento use somente `"ollama_llava"`)

**Resposta:**
```json
{
  "status": "ok",
  "model": "moondream"
}
```

***

## Observações

- Os modelos disponíveis podem ser consultados em `/models`.
- O prompt padrão realiza classificação parental ("safe"/"unsafe"), mas prompts livres podem ser usados para caption ou outras tarefas.
- Exemplos completos, arquitetura Docker e dependências estão nos arquivos deste repositório.

***

## Modelos suportados

| Nome interno        | Finalidade / Origem              |   
|--------------------|----------------------------------|   
| moondream          | Safe/unsafe + Caption, HuggingFace |   
| ollama_llava       | Caption via Ollama                |   
| blip               | Caption, HuggingFace              |   
| unsloth_vision     | Caption avançado, HuggingFace     |   
| pixtral            | Caption, HuggingFace              |   

***

Qualquer dúvida sobre uso ou integração, consulte `/models` para nomes exatos ou revise exemplos acima.[2][3][1]

[1](https://notagateway.com.br/blog/como-documentar-api-passo-a-passo-para-otimizar-integracoes/)
[2](https://document360.com/pt-br/blog/documentacao-de-api/)
[3](https://twlivre.org/tutoriais/api-docs/introducao-apis/)
