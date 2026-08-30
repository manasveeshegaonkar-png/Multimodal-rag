from fastapi import FastAPI

# Create the FastAPI application
app = FastAPI(title="Multimodal RAG System")


@app.get("/")
def root():
    return {
        "message": "Multimodal RAG System is running",
        "status": "ok"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }