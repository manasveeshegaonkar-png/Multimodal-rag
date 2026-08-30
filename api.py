from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import pymupdf
import os

from app.ingestion.vision.indexing.retrieval.retriever import retrieve
from app.ingestion.vision.indexing.retrieval.generation.generator import generate_answer


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="PDF Insights API",
    description="Multimodal RAG API for answering questions from PDF documents.",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODEL
# =========================================================

class QuestionRequest(BaseModel):
    query: str


# =========================================================
# PDF PATH
# =========================================================

PDF_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "data",
        "raw",
        "sample.pdf"
    )
)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "PDF Insights API is running",
        "status": "ok"
    }


# =========================================================
# ASK PDF
# =========================================================

@app.post("/ask")
def ask_pdf(request: QuestionRequest):

    question = request.query.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        print("\n" + "=" * 60)
        print("QUESTION:", question)
        print("=" * 60)

        # =================================================
        # 1. RETRIEVE RELEVANT CHUNKS
        # =================================================

        results = retrieve(
            question,
            top_k=8
        )

        if not results:

            return {
                "answer": (
                    "I could not find relevant information "
                    "in the provided PDF."
                ),
                "sources": []
            }

        print(
            f"\nRetrieved {len(results)} relevant chunks."
        )

        # =================================================
        # 2. BUILD CONTEXT
        # =================================================

        context_parts = []

        for result in results:

            page = result.get(
                "page",
                "Unknown"
            )

            source_type = result.get(
                "type",
                "text"
            )

            text = result.get(
                "text",
                ""
            )

            if text:

                context_parts.append(
                    f"""
[PAGE {page} | SOURCE TYPE: {source_type}]

{text}
"""
                )

        context = "\n".join(
            context_parts
        )

        print(
            "\nContext length:",
            len(context),
            "characters"
        )

        # =================================================
        # 3. GENERATE ANSWER
        # =================================================

        answer = generate_answer(
            question,
            context
        )

        # =================================================
        # 4. BUILD SOURCES
        # =================================================

        sources = []

        seen_sources = set()

        for result in results:

            page = result.get(
                "page"
            )

            source_type = result.get(
                "type",
                "text"
            )

            image = result.get(
                "image"
            )

            chunk_id = result.get(
                "chunk_id"
            )

            score = result.get(
                "score"
            )

            source_key = (
                page,
                source_type,
                image
            )

            if source_key in seen_sources:

                continue

            seen_sources.add(
                source_key
            )

            sources.append({

                "page": page,

                "type": source_type,

                "image": image,

                "chunk_id": chunk_id,

                "score": score

            })

        # =================================================
        # 5. RESPONSE
        # =================================================

        response = {

            "answer": answer,

            "sources": sources

        }

        print(
            "\nSources returned:",
            len(sources)
        )

        for source in sources:

            print(
                f"  Page {source['page']} "
                f"| {source['type']}"
            )

        print("=" * 60)

        return response

    except Exception as e:

        print(
            "\nPDF INSIGHTS ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# VIEW ORIGINAL PDF PAGE
# =========================================================

@app.get("/source/{page}")
def get_pdf_page(page: int):

    try:

        # ---------------------------------------------
        # Check PDF exists
        # ---------------------------------------------

        if not os.path.exists(PDF_PATH):

            raise HTTPException(
                status_code=404,
                detail=f"PDF not found: {PDF_PATH}"
            )

        # ---------------------------------------------
        # Validate page number
        # ---------------------------------------------

        if page < 1:

            raise HTTPException(
                status_code=400,
                detail="Page number must be 1 or greater."
            )

        # ---------------------------------------------
        # Open PDF
        # ---------------------------------------------

        document = pymupdf.open(
            PDF_PATH
        )

        total_pages = len(document)

        # ---------------------------------------------
        # Check page range
        # ---------------------------------------------

        if page > total_pages:

            document.close()

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Page {page} does not exist. "
                    f"PDF contains {total_pages} pages."
                )
            )

        # ---------------------------------------------
        # PDF pages are zero-indexed internally
        # ---------------------------------------------

        pdf_page = document[page - 1]

        # ---------------------------------------------
        # Render page as image
        # ---------------------------------------------

        pixmap = pdf_page.get_pixmap(
            matrix=pymupdf.Matrix(1.5, 1.5),
            alpha=False
        )

        image_bytes = pixmap.tobytes(
            "png"
        )

        document.close()

        print(
            f"Source page rendered successfully: {page}"
        )

        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={
                "Cache-Control": "no-cache"
            }
        )

    except HTTPException:

        raise

    except Exception as e:

        print(
            "\nSOURCE VIEWER ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )