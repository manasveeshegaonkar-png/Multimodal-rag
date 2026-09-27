import os
import re
import uuid
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
from fastembed import TextEmbedding
from google import genai


load_dotenv()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FRONTEND_DIR = BASE_DIR / "frontend"

UPLOAD_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ENVIRONMENT / CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:
    print(
        "\nWARNING: GEMINI_API_KEY is not configured."
        "\nThe application will start, but AI answers will not work"
        "\nuntil the API key is added.\n"
    )


GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


MAX_UPLOAD_SIZE_MB = int(
    os.getenv(
        "MAX_UPLOAD_SIZE_MB",
        "50"
    )
)


MAX_UPLOAD_SIZE_BYTES = (
    MAX_UPLOAD_SIZE_MB
    * 1024
    * 1024
)


# ============================================================
# RAG CONFIGURATION
# ============================================================

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
# FASTEMBED MODEL
# ============================================================

embedding_model = None


def get_embedding_model():
    """
    Load the lightweight FastEmbed model lazily.

    The model is loaded only when an embedding is actually needed.
    This helps reduce memory usage on low-memory deployments.
    """

    global embedding_model

    if embedding_model is None:

        print(
            "\nLoading lightweight embedding model:"
        )

        print(
            EMBEDDING_MODEL_NAME
        )

        embedding_model = TextEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            lazy_load=True
        )

        print(
            "Lightweight embedding model initialized."
        )

    return embedding_model


# ============================================================
# DOCUMENT EMBEDDINGS
# ============================================================

def embed_documents(
    texts: list[str]
) -> np.ndarray:

    if not texts:
        return np.empty(
            (0, 384),
            dtype="float32"
        )

    model = get_embedding_model()

    vectors = list(
        model.embed(
            texts,
            batch_size=8
        )
    )

    embeddings = np.asarray(
        vectors,
        dtype="float32"
    )

    if embeddings.size == 0:
        return np.empty(
            (0, 384),
            dtype="float32"
        )

    faiss.normalize_L2(
        embeddings
    )

    return embeddings


# ============================================================
# QUERY EMBEDDING
# ============================================================

def embed_query(
    question: str
) -> np.ndarray:

    model = get_embedding_model()

    vectors = list(
        model.query_embed(
            [question]
        )
    )

    query_embedding = np.asarray(
        vectors,
        dtype="float32"
    )

    if query_embedding.size == 0:
        return np.empty(
            (0, 384),
            dtype="float32"
        )

    faiss.normalize_L2(
        query_embedding
    )

    return query_embedding


# ============================================================
# IN-MEMORY DOCUMENT STORAGE
# ============================================================

documents: dict[
    str,
    dict[str, Any]
] = {}


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(
    BaseModel
):

    query: str

    document_id: str


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_filename(
    filename: str
) -> str:

    filename = Path(
        filename
    ).name

    filename = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        filename
    )

    if not filename.lower().endswith(
        ".pdf"
    ):
        filename += ".pdf"

    return filename


# ============================================================
# DOCUMENT ID
# ============================================================

def create_document_id() -> str:

    return uuid.uuid4().hex


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:

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

        chunk = (
            " ".join(
                words[start:end]
            )
            .strip()
        )

        if chunk:
            chunks.append(
                chunk
            )

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(
    chunks: list[dict[str, Any]]
):

    if not chunks:
        return None

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embed_documents(
        texts
    )

    if embeddings.size == 0:
        return None

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    del embeddings

    return index


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(
    document: dict[str, Any],
    question: str,
    top_k: int = TOP_K
) -> list[dict[str, Any]]:

    index = document.get(
        "index"
    )

    chunks = document.get(
        "chunks",
        []
    )

    if index is None or not chunks:
        return []

    query_embedding = embed_query(
        question
    )

    if query_embedding.size == 0:
        return []

    k = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    del query_embedding

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position < 0:
            continue

        if index_position >= len(chunks):
            continue

        chunk = chunks[
            index_position
        ].copy()

        chunk["score"] = float(
            score
        )

        results.append(
            chunk
        )

    return results


# ============================================================
# RENDER PDF PAGE
# ============================================================

def render_page(
    pdf_path: str,
    page_number: int
) -> bytes:

    document = pymupdf.open(
        pdf_path
    )

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

        return pixmap.tobytes(
            "png"
        )

    finally:

        document.close()


# ============================================================
# RAG PROMPT
# ============================================================

def build_rag_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]]
) -> str:

    context_blocks = []

    for result in retrieved_chunks:

        context_blocks.append(
            "\n".join(
                [
                    (
                        f"[SOURCE PAGE {result['page']} | "
                        f"CHUNK {result['chunk_id']} | "
                        f"RELEVANCE {result['score']:.4f}]"
                    ),
                    result["text"]
                ]
            )
        )

    retrieved_context = (
        "\n\n".join(
            context_blocks
        )
    )

    if not retrieved_context:

        retrieved_context = (
            "[No sufficiently relevant text chunks were retrieved.]"
        )


    prompt = f"""
You are the AI assistant inside an application called PDF Insights.

Your job is to answer the user's question using the uploaded PDF.

The original PDF is available to you. It may contain:

- normal text
- tables
- charts
- figures
- diagrams
- images
- formulas
- examples

You also receive semantically retrieved text from the PDF.

USER QUESTION:
{question}


RETRIEVED INFORMATION:
{retrieved_context}


============================================================
1. MAIN GOAL
============================================================

Give the user a clear, natural and easy-to-understand answer.

Your answer should feel like a helpful ChatGPT response from a
knowledgeable person explaining something to a student.

Do NOT sound like:

- a textbook
- a research paper
- an academic report
- technical documentation
- a PDF summary
- a search-result extraction
- a collection of copied facts


============================================================
2. ANSWER THE QUESTION FIRST
============================================================

Start directly with the answer.

Do NOT unnecessarily start with phrases such as:

"According to the provided context..."

"Based on the retrieved information..."

"The document states..."

"The PDF discusses..."

"The text says..."

Only mention the PDF when doing so is genuinely useful.


============================================================
3. USE SIMPLE, NATURAL ENGLISH
============================================================

Use language that a college student can easily understand.

For example, prefer:

"An input vector is simply a collection of values that represents
one data sample and is given to a machine learning model as input."

Avoid overly academic wording such as:

"An input vector constitutes an n-dimensional representation of the
features supplied to a machine-learning function."

Explain the idea instead of trying to reproduce the writing style
of the PDF.


============================================================
4. VERY IMPORTANT: PLAIN TEXT FORMATTING
============================================================

For normal questions, answer using plain readable text.

DO NOT use:

- LaTeX
- mathematical notation
- special mathematical symbols
- unnecessary Unicode mathematical symbols
- symbolic variable notation
- equation formatting
- "$...$"
- "\\(...\\)"
- "\\[...\\]"
- expressions such as x₁, x₂, xₙ
- expressions such as X = (...)
- unnecessary arrows or symbolic representations

unless the user explicitly asks for a formula, equation,
mathematical notation, or the exact representation from the PDF.


For example, DO NOT write:

"An input vector is represented as X = (x₁, x₂, ..., xₙ)."

Instead write:

"An input vector is a collection of values representing the
features of one data sample."


The PDF may contain mathematical notation.

You should UNDERSTAND that notation when necessary, but do NOT
automatically reproduce it in your answer.

Understanding the PDF's notation does NOT mean copying that notation
into the response.


============================================================
5. WHEN SYMBOLS ARE ACTUALLY NECESSARY
============================================================

Only use mathematical notation when:

1. The user explicitly asks for it, OR
2. It is essential for answering the question and cannot reasonably
   be explained in words.

If notation is necessary, first explain the concept in normal words.

For example:

"An input vector is a group of values representing the features of
one sample. In mathematical notation, the PDF represents it as
X = (...)."

Only do this when the notation is relevant to the user's question.


============================================================
6. DO NOT COPY PDF FORMATTING
============================================================

The PDF may contain:

- equations
- numbered definitions
- academic terminology
- long lists
- headings
- symbolic notation
- formal descriptions

Do not reproduce these simply because they appear in the PDF.

Understand the information and explain it naturally.


============================================================
7. KEEP SIMPLE QUESTIONS SIMPLE
============================================================

If the user asks a simple question, give a simple answer.

For example, if the user asks:

"What is an input vector?"

A good answer would normally be:

"An input vector is simply a collection of values that represents
one data sample and is given to a machine learning model as input.

Each value represents a feature of that sample. For example, a
student could be represented using information such as age, subject,
gender, and year of study.

In simple terms, you can think of an input vector as the information
about one example that the model uses to make a prediction."

Do not turn this into a long academic explanation unless the user
asks for more detail.


============================================================
8. SHORT PARAGRAPHS
============================================================

Prefer short paragraphs.

Use bullets only when they genuinely improve readability.

Do NOT automatically create sections such as:

"Key Details from the Text"

"Synonyms"

"Components / Features"

"Types of Values"

"Important Points"

"Summary"

unless they are genuinely useful or the user asks for them.


============================================================
9. DO NOT REPEAT YOURSELF
============================================================

Explain each idea once.

Do not give the same definition in several different forms.

Do not repeat the user's question unnecessarily.

Do not add a conclusion that simply repeats the answer.


============================================================
10. TECHNICAL TERMS
============================================================

Use technical terms when they are important.

But explain them naturally when necessary.

For example:

"An embedding is a numerical representation of text that allows the
system to compare the meaning of different pieces of text."

Do not define every technical word automatically.


============================================================
11. EXAMPLES
============================================================

Use examples when they make the concept easier to understand.

Keep examples simple.

If the example is your own illustrative example rather than something
directly stated in the PDF, make sure it is clear that it is only an
example.

Do not present your own example as something the PDF said.


============================================================
12. EQUATIONS
============================================================

Do not reproduce equations merely because the PDF contains them.

Only include an equation if:

- the user asks for it, or
- the equation is essential to understanding the requested answer.

For a normal conceptual question, explain the idea using words.


============================================================
13. PDF GROUNDING
============================================================

The uploaded PDF is the source of truth.

Use information supported by the uploaded PDF.

Do NOT invent:

- facts
- numbers
- formulas
- definitions
- conclusions
- citations
- page numbers

Do NOT use outside knowledge to fill important gaps.

If the answer cannot be supported by the uploaded PDF, say:

"I couldn't find enough information about this in the provided PDF."


============================================================
14. RETRIEVED CHUNKS
============================================================

The retrieved chunks are search results.

They are NOT automatically the final answer.

Some retrieved chunks may be only partially relevant.

Use only the parts that actually help answer the question.

The original PDF is also available for checking the surrounding
context.

If the retrieved chunks are insufficient but the answer may be
available elsewhere in the PDF, use the original PDF before deciding
that the information is unavailable.


============================================================
15. PAGE REFERENCES
============================================================

Mention page numbers naturally when useful.

For example:

"The PDF explains this on page 16."

or:

"This example appears on page 16."

Do not invent page numbers.

Only mention a page number when it can be supported by the PDF.


============================================================
16. VISUAL CONTENT
============================================================

The PDF may contain charts, tables, figures, diagrams and images.

If the user's question is specifically about one of these, use the
original PDF to understand the relevant content.

Do not claim to have interpreted a visual if the available PDF
information does not support that interpretation.


============================================================
17. DIFFERENT TYPES OF QUESTIONS
============================================================

For a definition question:

Give the definition first and then a short explanation or example
if useful.

For a "how" question:

Explain the process in clear steps.

For a "why" question:

Explain the reason directly.

For a comparison:

Explain the differences clearly.

Use a table only when it genuinely makes the comparison easier.

For a detailed question:

Give enough detail to properly answer it, but keep the language
natural and readable.


============================================================
18. LENGTH
============================================================

Match the answer length to the question.

Simple question:
Usually 1–3 short paragraphs.

Moderately detailed question:
A few short paragraphs or a small number of bullets.

Complex question:
Provide enough detail to properly answer it.

Do not make the answer longer simply because more information exists
in the PDF.


============================================================
19. FINAL QUALITY CHECK
============================================================

Before responding, check that the answer is:

1. Correct
2. Supported by the PDF
3. Direct
4. Easy to understand
5. Natural
6. Not repetitive
7. Not unnecessarily academic
8. Appropriately concise
9. Written in plain English
10. Free from unnecessary mathematical notation
11. Free from unnecessary PDF-style formatting
12. Not simply copying the wording or structure of the PDF

Now answer the user's question.
"""

    return prompt


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    pdf_path: str,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    gemini_file: Any
) -> str:

    if gemini_client is None:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    prompt = build_rag_prompt(
        question,
        retrieved_chunks
    )

    response = (
        gemini_client
        .models
        .generate_content(
            model=GEMINI_MODEL,
            contents=[
                gemini_file,
                prompt
            ]
        )
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


# ============================================================
# UPLOAD PDF TO GEMINI
# ============================================================

def upload_pdf_to_gemini(
    file_path: str
):

    if gemini_client is None:
        return None

    print(
        "\nUploading PDF to Gemini Files API..."
    )

    uploaded_file = (
        gemini_client
        .files
        .upload(
            file=file_path
        )
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
# ROOT
# ============================================================

@app.get("/")
async def root():

    index_file = (
        FRONTEND_DIR
        / "index.html"
    )

    if not index_file.exists():

        return {
            "message": "PDF Insights API is running.",
            "status": "ok",
            "frontend": (
                "frontend/index.html not found"
            )
        }

    return FileResponse(
        index_file
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "gemini_configured": (
            gemini_client is not None
        ),
        "gemini_model": GEMINI_MODEL,
        "embedding_model": (
            EMBEDDING_MODEL_NAME
        ),
        "embedding_backend": (
            "FastEmbed / ONNX Runtime"
        ),
        "documents_loaded": len(
            documents
        )
    }


# ============================================================
# UPLOAD ENDPOINT
# ============================================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

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

    file_path = None

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
            f"{document_id}_"
            f"{original_filename}"
        )

        file_path = (
            UPLOAD_DIR
            / stored_filename
        )

        with open(
            file_path,
            "wb"
        ) as output_file:

            output_file.write(
                contents
            )

        del contents


        print(
            "\n" + "=" * 70
        )

        print(
            "PROCESSING PDF:",
            original_filename
        )


        document = pymupdf.open(
            str(file_path)
        )

        page_count = len(
            document
        )

        chunks = []

        pages_with_text = 0

        pages_with_images = 0


        for page_number, page in enumerate(
            document,
            start=1
        ):

            page_text = (
                page
                .get_text(
                    "text"
                )
                .strip()
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

                for (
                    chunk_number,
                    chunk_text
                ) in enumerate(
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


        index = build_faiss_index(
            chunks
        )


        print(
            "FAISS index created:",
            index is not None
        )


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
                    repr(
                        gemini_error
                    )
                )

                gemini_file = None


        documents[
            document_id
        ] = {

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
            "Gemini multimodal file available:",
            gemini_file is not None
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


        if file_path is not None:

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


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post("/ask")
async def ask_pdf(
    request: QuestionRequest
):

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


        results = retrieve_chunks(
            document,
            question,
            TOP_K
        )


        print(
            "Retrieved chunks:",
            len(results)
        )


        print(
            "Retrieved pages:",
            [
                result["page"]
                for result in results
            ]
        )


        gemini_file = document.get(
            "gemini_file"
        )


        if gemini_file is None:

            print(
                "Gemini PDF reference missing."
            )

            print(
                "Uploading PDF again..."
            )

            gemini_file = (
                upload_pdf_to_gemini(
                    document["path"]
                )
            )

            document[
                "gemini_file"
            ] = gemini_file


        if gemini_file is None:

            raise RuntimeError(
                "The PDF could not be uploaded to Gemini."
            )


        answer = generate_answer(
            document["path"],
            question,
            results,
            gemini_file
        )


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
                    "chunk_id": (
                        result[
                            "chunk_id"
                        ]
                    ),
                    "score": round(
                        result[
                            "score"
                        ],
                        4
                    ),
                    "filename": (
                        document[
                            "filename"
                        ]
                    )
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

            "document_id": (
                request.document_id
            ),

            "filename": (
                document[
                    "filename"
                ]
            )

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


# ============================================================
# SOURCE PAGE
# ============================================================

@app.get(
    "/source/{document_id}/{page}"
)
async def get_pdf_page(
    document_id: str,
    page: int
):

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


    if page > document[
        "pages"
    ]:

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
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )