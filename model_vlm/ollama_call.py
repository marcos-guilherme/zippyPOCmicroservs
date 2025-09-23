import ollama
import base64


def llava_call(image_path: str, prompt: str):
    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        response = ollama.chat(
            model="llava-7b",
            messages=[{"role": "user", "content": prompt, "image": base64_image}],
        )

        return response["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"
