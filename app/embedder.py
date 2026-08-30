import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# -----------------------------
# File paths
# -----------------------------
CHUNKS_FILE = "data/processed/chunks.json"
IMAGE_DESCRIPTIONS_FILE = "data/processed/image_descriptions.json"

EMBEDDINGS_FILE = "data/processed/embeddings.npy"
MAPPING_FILE = "data/processed/chunk_mapping.json"


# -----------------------------
# Load text chunks
# -----------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} text chunks")


# -----------------------------
# Load image descriptions
# -----------------------------
with open(IMAGE_DESCRIPTIONS_FILE, "r", encoding="utf-8") as f:
    image_descriptions = json.load(f)

print(f"Loaded {len(image_descriptions)} image descriptions")


# -----------------------------
# Prepare combined documents
# -----------------------------
documents = []
mapping = []


# -----------------------------
# Add text chunks
# -----------------------------
for i, chunk in enumerate(chunks):

    text = chunk.get("text", "")

    if text.strip():

        documents.append(text)

        mapping.append({
            "chunk_id": chunk.get("chunk_id", i),
            "page": chunk.get("page"),
            "type": "text",
            "text": text
        })


# -----------------------------
# Add image descriptions
# -----------------------------
for image in image_descriptions:

    description = image.get("description", "")
    image_name = image.get("image", "")

    if description.strip():

        # Extract page number from filename
        # Example:
        # page_101_image_1.jpeg -> 101
        match = re.search(r"page_(\d+)", image_name)

        page = int(match.group(1)) if match else None

        documents.append(description)

        mapping.append({
            "page": page,
            "type": "image",
            "image": image_name,
            "description": description,
            "text": description
        })


print(f"Total documents to embed: {len(documents)}")


# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Create embeddings
# -----------------------------
embeddings = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
).astype("float32")


# -----------------------------
# Save embeddings
# -----------------------------
np.save(EMBEDDINGS_FILE, embeddings)


# -----------------------------
# Save mapping
# -----------------------------
with open(MAPPING_FILE, "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)
print("DEBUG mapping length:", len(mapping))
print("DEBUG image mappings:", sum(1 for x in mapping if x.get("type") == "image"))
print("DEBUG last mapping:", mapping[-1])

# -----------------------------
# Final information
# -----------------------------
print()
print("====================================")
print("MULTIMODAL EMBEDDING COMPLETED")
print("====================================")
print(f"Text chunks: {len(chunks)}")
print(f"Image descriptions: {len(image_descriptions)}")
print(f"Total embeddings: {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Saved embeddings: {EMBEDDINGS_FILE}")
print(f"Saved mapping: {MAPPING_FILE}")