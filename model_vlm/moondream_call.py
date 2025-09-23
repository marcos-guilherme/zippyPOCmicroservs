from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# def query_img(image_path, question):
#     image = Image.open(image_path).convert("RGB")
#     answer = model_moondream.query(image, question)["answer"]
#     return answer

def query_img(image_path, prompt: str):

    model_moondream = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map={"": "cuda"}
    )


    image = Image.open(image_path).convert("RGB")
    answer = model_moondream.query(image, prompt)["answer"]
    return answer