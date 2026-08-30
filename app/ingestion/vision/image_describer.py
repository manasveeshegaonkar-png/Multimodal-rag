import os
import base64
import requests

IMAGE_DIR = "data/processed/images"
OUTPUT_FILE = "data/processed/image_descriptions.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:3b"


def describe_image(image_path):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    payload = {
        "model": MODEL,
        "prompt": (
            "Analyze this image from a technical book. "
            "Describe all important information in the image, including "
            "diagrams, labels, tables, graphs, equations, and their meaning. "
            "Be detailed but concise. This description will be used for "
            "retrieval in a RAG system."
        ),
        "images": [image_base64],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["response"]


def main():
    descriptions = []

    for filename in os.listdir(IMAGE_DIR):

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            image_path = os.path.join(IMAGE_DIR, filename)

            print(f"Analyzing: {filename}")

            description = describe_image(image_path)

            descriptions.append({
                "image": filename,
                "description": description
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        import json
        json.dump(descriptions, f, ensure_ascii=False, indent=2)

    print("\nImage descriptions completed!")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()