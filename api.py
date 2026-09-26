import os
import re
import uuid
import shutil
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pymupdf

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "data" / "raw" / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print(
        "\nWARNING: GEMINI_API_KEY is not configured."
        "\nThe application will start, but AI answers will not work"
        " until the API key is added.\n"
    )

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

MAX_UPLOAD_SIZE_MB = int(
    os.getenv("MAX_UPLOAD_SIZE_MB", "50")
)

MAX_UPLOAD_SIZE_BYTES = (
    MAX_UPLOAD_SIZE_MB * 1024 * 1024
)

CHUNK_SIZE = 350
CHUNK_OVERLAP = 70
TOP_K = 6


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="PDF Insights",
    description=(
        "Multimodal Retrieval-Augmented Generation system "
        "for asking questions about uploaded PDF documents."
    ),
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = None


def get_embedding_model():
    """
    Load the sentence-transformer model only when required.

    This avoids downloading/loading the model when the API
    is merely started.
    """

    global embedding_model

    if embedding_model is None:
        print(
            "\nLoading embedding model:",
            EMBEDDING_MODEL_NAME
        )

        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        print("Embedding model loaded.")

    return embedding_model


# ============================================================
# IN-MEMORY DOCUMENT STORE
# ============================================================

documents: dict[str, dict[str, Any]] = {}


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    query: str
    document_id: str


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_filename(filename: str) -> str:
    """
    Remove unsafe path characters from an uploaded filename.
    """

    filename = Path(filename).name

    filename = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        filename
    )

    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return filename


def create_document_id() -> str:
    """
    Generate a collision-resistant document ID.
    """

    return uuid.uuid4().hex


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Split page text into overlapping word-based chunks.
    """

    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        overlap = chunk_size // 5

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_faiss_index(
    chunks: list[dict[str, Any]]
):
    """
    Create normalized sentence embeddings and a FAISS
    inner-product index.

    Because vectors are normalized, inner product behaves
    as cosine similarity.
    """

    if not chunks:
        return None

    model = get_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


def retrieve_chunks(
    document: dict[str, Any],
    question: str,
    top_k: int = TOP_K
) -> list[dict[str, Any]]:
    """
    Perform semantic retrieval against the document's
    FAISS index.
    """

    index = document.get("index")
    chunks = document.get("chunks", [])

    if index is None or not chunks:
        return []

    model = get_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    k = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position < 0:
            continue

        if index_position >= len(chunks):
            continue

        chunk = chunks[index_position].copy()

        chunk["score"] = float(score)

        results.append(chunk)

    return results


def render_page(
    pdf_path: str,
    page_number: int
) -> bytes:
    """
    Render a specific PDF page as PNG bytes.
    """

    document = pymupdf.open(pdf_path)

    try:

        if page_number < 1:
            raise ValueError(
                "Page number must be at least 1."
            )

        if page_number > len(document):
            raise ValueError(
                f"Page {page_number} does not exist."
            )

        page = document[
            page_number - 1
        ]

        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(
                1.5,
                1.5
            ),
            alpha=False
        )

        return pixmap.tobytes("png")

    finally:
        document.close()


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]]
) -> str:
    """
    Build the grounded prompt supplied alongside the
    original PDF.
    """

    context_blocks = []

    for result in retrieved_chunks:

        context_blocks.append(
            "\n".join(
                [
                    (
                        f"[RETRIEVED SOURCE | "
                        f"PAGE {result['page']} | "
                        f"CHUNK {result['chunk_id']}]"
                    ),
                    result["text"]
                ]
            )
        )

    retrieved_context = "\n\n".join(
        context_blocks
    )

    prompt = f"""
You are the AI assistant inside a multimodal PDF
question-answering application.

The user uploaded a PDF.

You have access to:
1. The original PDF, including its text, tables,
   figures, diagrams and page visuals.
2. Semantically retrieved text chunks from that PDF.

Your job is to answer the user's question using ONLY
information supported by the uploaded PDF.

USER QUESTION:
{question}

SEMANTICALLY RETRIEVED CONTEXT:
{retrieved_context}

IMPORTANT RULES:

- Answer only from the uploaded PDF.
- Do not use outside knowledge to fill missing information.
- If the PDF does not contain enough information, clearly say:
  "I could not find enough information in the provided PDF."
- Do not invent facts, numbers, citations or page references.
- You may use the original PDF to inspect tables, figures,
  diagrams and other visual information when necessary.
- When explaining an answer, mention the relevant page
  numbers naturally, for example "According to page 12..."
- Keep the answer clear and useful.
- Prefer concise explanations unless the question requires
  detail.
- If the user asks for a comparison, use a structured
  comparison.
- If the user asks for a definition, explain it simply.
- If the user asks about a table, chart, diagram or figure,
  inspect the corresponding visual information in the PDF.
"""

    return prompt


def generate_answer(
    pdf_path: str,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    gemini_file: Any
) -> str:
    """
    Generate a grounded multimodal answer using Gemini.

    The original PDF is provided to Gemini so that the model
    can understand visual content in addition to the retrieved
    semantic text.
    """

    if gemini_client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    prompt = build_rag_prompt(
        question,
        retrieved_chunks
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            gemini_file,
            prompt
        ]
    )

    answer = getattr(
        response,
        "text",
        None
    )

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return answer.strip()


def upload_pdf_to_gemini(
    file_path: str
):
    """
    Upload the original PDF to Gemini Files API.

    This allows Gemini to inspect the complete PDF,
    including visual information.
    """

    if gemini_client is None:
        return None

    print(
        "\nUploading PDF to Gemini Files API..."
    )

    uploaded_file = gemini_client.files.upload(
        file=file_path
    )

    print(
        "Gemini file uploaded:",
        getattr(
            uploaded_file,
            "name",
            "unknown"
        )
    )

    return uploaded_file


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    """
    Serve the actual application UI.
    """

    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():

        return {
            "message": "PDF Insights API is running.",
            "status": "ok",
            "frontend": "frontend/index.html not found"
        }

    return FileResponse(
        index_file
    )


@app.get("/health")
async def health():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
        "gemini_configured": (
            gemini_client is not None
        ),
        "embedding_model": (
            EMBEDDING_MODEL_NAME
        ),
        "documents_loaded": len(
            documents
        )
    }


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):
    """
    Upload and process a PDF.

    Processing pipeline:

    PDF
      ↓
    Save
      ↓
    PyMuPDF text extraction
      ↓
    Page-aware chunks
      ↓
    Sentence Transformer embeddings
      ↓
    FAISS index
      ↓
    Gemini multimodal PDF upload
      ↓
    Document ready for questions
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename was provided."
        )

    if not file.filename.lower().endswith(
        ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty."
            )

        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"PDF is too large. Maximum size is "
                    f"{MAX_UPLOAD_SIZE_MB} MB."
                )
            )

        document_id = create_document_id()

        original_filename = safe_filename(
            file.filename
        )

        stored_filename = (
            f"{document_id}_{original_filename}"
        )

        file_path = (
            UPLOAD_DIR /
            stored_filename
        )

        with open(
            file_path,
            "wb"
        ) as output_file:

            output_file.write(
                contents
            )

        print(
            "\n" + "=" * 70
        )

        print(
            "PROCESSING PDF:",
            original_filename
        )

        # ----------------------------------------------------
        # OPEN PDF
        # ----------------------------------------------------

        document = pymupdf.open(
            str(file_path)
        )

        page_count = len(
            document
        )

        chunks = []

        pages_with_text = 0
        pages_with_images = 0

        # ----------------------------------------------------
        # PAGE-BY-PAGE PROCESSING
        # ----------------------------------------------------

        for page_number, page in enumerate(
            document,
            start=1
        ):

            page_text = (
                page.get_text(
                    "text"
                ).strip()
            )

            image_count = len(
                page.get_images(
                    full=True
                )
            )

            if image_count > 0:
                pages_with_images += 1

            if page_text:

                pages_with_text += 1

                page_chunks = split_text(
                    page_text
                )

                for chunk_number, chunk_text in enumerate(
                    page_chunks,
                    start=1
                ):

                    chunks.append(
                        {
                            "chunk_id": (
                                f"p{page_number}"
                                f"_c{chunk_number}"
                            ),
                            "page": page_number,
                            "text": chunk_text,
                            "type": "text"
                        }
                    )

        document.close()

        print(
            "Pages:",
            page_count
        )

        print(
            "Pages containing text:",
            pages_with_text
        )

        print(
            "Pages containing images:",
            pages_with_images
        )

        print(
            "Text chunks:",
            len(chunks)
        )

        # ----------------------------------------------------
        # FAISS INDEX
        # ----------------------------------------------------

        index = build_faiss_index(
            chunks
        )

        # ----------------------------------------------------
        # GEMINI FILE
        # ----------------------------------------------------

        gemini_file = None

        if gemini_client is not None:

            try:

                gemini_file = (
                    upload_pdf_to_gemini(
                        str(file_path)
                    )
                )

            except Exception as gemini_error:

                print(
                    "\nGemini upload warning:",
                    repr(gemini_error)
                )

                gemini_file = None

        # ----------------------------------------------------
        # STORE DOCUMENT
        # ----------------------------------------------------

        documents[document_id] = {
            "document_id": document_id,
            "filename": original_filename,
            "path": str(file_path),
            "pages": page_count,
            "chunks": chunks,
            "index": index,
            "gemini_file": gemini_file
        }

        print(
            "Document ID:",
            document_id
        )

        print(
            "=" * 70
        )

        return {
            "message": (
                "PDF uploaded and processed successfully."
            ),
            "document_id": document_id,
            "filename": original_filename,
            "pages": page_count,
            "chunks": len(chunks),
            "pages_with_text": pages_with_text,
            "pages_with_images": pages_with_images,
            "semantic_search": (
                index is not None
            ),
            "multimodal_ai": (
                gemini_file is not None
            )
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            "\nPDF PROCESSING ERROR:",
            repr(error)
        )

        if "file_path" in locals():

            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass

        raise HTTPException(
            status_code=500,
            detail=(
                "PDF processing failed: "
                f"{str(error)}"
            )
        )


@app.post("/ask")
async def ask_pdf(
    request: QuestionRequest
):
    """
    Answer a question using:

    Question
       ↓
    Sentence-transformer embedding
       ↓
    FAISS semantic retrieval
       ↓
    Relevant page-aware chunks
       ↓
    Original PDF to Gemini
       ↓
    Grounded multimodal answer
    """

    question = (
        request.query or ""
    ).strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    document = documents.get(
        request.document_id
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Document not found. "
                "Please upload the PDF again."
            )
        )

    if gemini_client is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "GEMINI_API_KEY is not configured. "
                "Add it to the environment before asking questions."
            )
        )

    try:

        print(
            "\n" + "=" * 70
        )

        print(
            "QUESTION:",
            question
        )

        # ----------------------------------------------------
        # SEMANTIC RETRIEVAL
        # ----------------------------------------------------

        results = retrieve_chunks(
            document,
            question,
            TOP_K
        )

        print(
            "Retrieved chunks:",
            len(results)
        )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        gemini_file = document.get(
            "gemini_file"
        )

        if gemini_file is None:

            gemini_file = upload_pdf_to_gemini(
                document["path"]
            )

            document["gemini_file"] = (
                gemini_file
            )

        answer = generate_answer(
            document["path"],
            question,
            results,
            gemini_file
        )

        # ----------------------------------------------------
        # SOURCE CREATION
        # ----------------------------------------------------

        sources = []

        seen_pages = set()

        for result in results:

            page = result["page"]

            if page in seen_pages:
                continue

            seen_pages.add(
                page
            )

            sources.append(
                {
                    "page": page,
                    "type": "text",
                    "chunk_id": result[
                        "chunk_id"
                    ],
                    "score": round(
                        result["score"],
                        4
                    ),
                    "filename": document[
                        "filename"
                    ]
                }
            )

        print(
            "Sources:",
            [
                source["page"]
                for source in sources
            ]
        )

        print(
            "=" * 70
        )

        return {
            "answer": answer,
            "sources": sources,
            "document_id": request.document_id,
            "filename": document[
                "filename"
            ]
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            "\nQUESTION ANSWERING ERROR:",
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not generate an answer: "
                f"{str(error)}"
            )
        )


@app.get(
    "/source/{document_id}/{page}"
)
async def get_pdf_page(
    document_id: str,
    page: int
):
    """
    Render a page from the exact PDF that the user uploaded.
    """

    document = documents.get(
        document_id
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    if page < 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Page number must be 1 or greater."
            )
        )

    if page > document["pages"]:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Page {page} does not exist. "
                f"The PDF contains "
                f"{document['pages']} pages."
            )
        )

    try:

        image_bytes = render_page(
            document["path"],
            page
        )

        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={
                "Cache-Control": "no-cache"
            }
        )

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:

        print(
            "\nSOURCE VIEWER ERROR:",
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not render source page: "
                f"{str(error)}"
            )
        )


# ============================================================
# START LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )