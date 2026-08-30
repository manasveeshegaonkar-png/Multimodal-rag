import numpy as np
import faiss

# -----------------------------
# File paths
# -----------------------------
EMBEDDINGS_FILE = "data/processed/embeddings.npy"
FAISS_FILE = "data/processed/faiss.index"


# -----------------------------
# Load embeddings
# -----------------------------
embeddings = np.load(EMBEDDINGS_FILE).astype("float32")


# -----------------------------
# Create FAISS index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# -----------------------------
# Save FAISS index
# -----------------------------
faiss.write_index(index, FAISS_FILE)


# -----------------------------
# IMPORTANT
# -----------------------------
# Do NOT recreate chunk_mapping.json here.
#
# embedder.py already created the correct
# multimodal mapping containing both:
#   - text chunks
#   - image descriptions
#
# -----------------------------

print(f"Indexed {index.ntotal} embeddings")
print(f"Saved FAISS index to {FAISS_FILE}")
print("Existing multimodal chunk mapping preserved.")