import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.ingestion.vision.indexing.retrieval.generation.generator import generate_answer


# -----------------------------
# Load FAISS index
# -----------------------------
index = faiss.read_index("data/processed/faiss.index")

embeddings = np.load(
    "data/processed/embeddings.npy"
).astype("float32")

with open(
    "data/processed/chunk_mapping.json",
    "r",
    encoding="utf-8"
) as f:
    chunks = json.load(f)

# -----------------------------
# Load multimodal mapping
# -----------------------------
with open(
    "data/processed/chunk_mapping.json",
    "r",
    encoding="utf-8"
) as f:
    chunks = json.load(f)


# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, top_k=5):

    import re

    print("\nQUERY:", query)

    # --------------------------------
    # Detect explicitly requested page
    # --------------------------------
    page_match = re.search(
        r"\bpage\s*(\d+)\b",
        query.lower()
    )

    requested_page = (
        int(page_match.group(1))
        if page_match
        else None
    )

    if requested_page is not None:
        print("Requested page:", requested_page)

    # --------------------------------
    # Create query embedding
    # --------------------------------
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    print("Embedding shape:", query_embedding.shape)

    # --------------------------------
    # PAGE-SPECIFIC RETRIEVAL
    # --------------------------------
    if requested_page is not None:

        page_indices = [
            i for i, chunk in enumerate(chunks)
            if chunk.get("page") == requested_page
        ]

        print(
            "Chunks found on page:",
            len(page_indices)
        )

        if page_indices:

            # Because embeddings are normalized,
            # dot product = cosine similarity.
            page_embeddings = embeddings[page_indices]

            similarities = np.dot(
                page_embeddings,
                query_embedding[0]
            )

            ranked = np.argsort(
                similarities
            )[::-1][:top_k]

            selected = [
                (
                    page_indices[i],
                    float(similarities[i])
                )
                for i in ranked
            ]

        else:
            selected = []

    # --------------------------------
    # NORMAL SEMANTIC RETRIEVAL
    # --------------------------------
    else:

        distances, indices = index.search(
            query_embedding,
            top_k
        )

        selected = [
            (int(idx), float(distance))
            for idx, distance in zip(
                indices[0],
                distances[0]
            )
            if idx != -1
        ]

    # --------------------------------
    # Build results
    # --------------------------------
    results = []

    for idx, score in selected:

        chunk = chunks[idx]

        text = chunk.get(
            "text",
            chunk.get("description", "")
        )

        print("\nCHUNK INDEX:", idx)
        print("PAGE:", chunk.get("page"))
        print("TYPE:", chunk.get("type"))

        print("\nRETRIEVED CHUNK:")
        print(text[:500])

        results.append({
            "chunk_id": chunk.get("chunk_id"),
            "page": chunk.get("page"),
            "type": chunk.get("type"),
            "image": chunk.get("image"),
            "text": text,
            "score": score
        })

    print("\nTotal results:", len(results))

    return results
# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    query = input("Enter your question: ")

    results = retrieve(query, top_k=5)

    context = "\n\n".join(
        f"[Page {result['page']} | "
        f"Type: {result['type']}]\n"
        f"{result['text']}"
        for result in results
        if result["text"]
    )

    answer = generate_answer(
        query,
        context
    )

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:\n")

    seen = set()

    for result in results:

        source = (
            result["page"],
            result["type"]
        )

        if source not in seen:

            if result["type"] == "image":
                print(
                    f"- Page {result['page']} "
                    f"(image: {result['image']})"
                )
            else:
                print(
                    f"- Page {result['page']} "
                    f"(text)"
                )

            seen.add(source)