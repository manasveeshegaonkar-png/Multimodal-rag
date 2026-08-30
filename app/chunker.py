import fitz
import json

pdf_path = "data/raw/sample.pdf"
output_path = "data/processed/chunks.json"

document = fitz.open(pdf_path)

chunks = []

chunk_id = 0

for page_number, page in enumerate(document):

    text = page.get_text().strip()

    if not text:
        continue

    paragraphs = text.split("\n\n")

    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) > 1500:

            chunks.append({
                "chunk_id": chunk_id,
                "page": page_number + 1,
                "type": "text",
                "text": current_chunk.strip()
            })

            chunk_id += 1

            current_chunk = paragraph

        else:

            current_chunk += "\n\n" + paragraph

    if current_chunk.strip():

        chunks.append({
            "chunk_id": chunk_id,
            "page": page_number + 1,
            "type": "text",
            "text": current_chunk.strip()
        })

        chunk_id += 1

document.close()

with open(output_path, "w", encoding="utf-8") as file:

    json.dump(
        chunks,
        file,
        indent=2,
        ensure_ascii=False
    )

print("Chunking completed!")
print("Total chunks:", len(chunks))
print("Saved to:", output_path)