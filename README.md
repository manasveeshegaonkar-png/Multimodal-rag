📚 Multimodal RAG PDF Insights

A Multimodal Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content using semantic search and Gemini multimodal AI.

The system retrieves relevant information from the uploaded document using vector similarity search and then uses the original PDF with Gemini to generate a clear, grounded answer. It also provides page-level source references so users can trace the answer back to the document.

⸻

🚀 Live Application

🔗 Live Demo: https://multimodal-rag-api-oose.onrender.com

⸻

✨ Features

* 📄 Upload PDF documents
* 🔎 Semantic search using vector embeddings
* ⚡ Fast similarity search with FAISS
* 🤖 Gemini-powered question answering
* 🖼️ Multimodal understanding of PDF content
* 📊 Support for text, tables, figures and diagrams
* 📑 Page-aware document processing
* 🔗 Source references for retrieved information
* 👀 View the exact source page used for an answer
* 🧠 Context-aware RAG responses
* 🌐 Deployable as a web application
* 🔐 API key configuration through environment variables

⸻

🧠 How It Works

The application follows a Retrieval-Augmented Generation pipeline:

                ┌─────────────────┐
                │    PDF Upload   │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │   PyMuPDF       │
                │ Text Extraction │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Semantic        │
                │ Chunking        │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ FastEmbed       │
                │ Embeddings      │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ FAISS Vector    │
                │ Index           │
                └────────┬────────┘
                         │
                    User Question
                         ↓
                ┌─────────────────┐
                │ Query Embedding │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ FAISS Semantic  │
                │ Retrieval       │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Relevant Chunks │
                │ + Source Pages │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Gemini           │
                │ Multimodal AI   │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Grounded Answer │
                │ + Sources       │
                └─────────────────┘

⸻

🔍 RAG Pipeline

1. PDF Upload

The user uploads a PDF through the web interface.

The application:

* validates the file
* checks the file size
* creates a unique document ID
* stores the PDF
* processes it page by page

⸻

2. Text Extraction

PyMuPDF is used to extract text from each PDF page.

Each extracted piece of text retains its associated page number.

For example:

Page 11
   ↓
Text
   ↓
Chunks
   ↓
p11_c1
p11_c2
p11_c3

This page-aware structure allows the application to later show the user where the retrieved information came from.

⸻

3. Semantic Chunking

The extracted text is divided into smaller overlapping chunks.

The current configuration uses:

Chunk size:     350 words
Chunk overlap:   70 words

The overlap helps preserve context between neighboring chunks.

Each chunk stores metadata such as:

chunk_id
page
text
type

⸻

4. Embeddings

Each text chunk is converted into a numerical vector using FastEmbed.

The embeddings represent the semantic meaning of the text.

This allows the system to search for information based on meaning rather than exact keyword matching.

FastEmbed runs through ONNX Runtime, providing a lightweight embedding pipeline.

⸻

5. FAISS Vector Search

The generated embeddings are stored in a FAISS vector index.

The application uses:

IndexFlatIP

The vectors are normalized before indexing, allowing inner-product similarity to behave as cosine similarity.

When the user asks a question:

Question
   ↓
Query embedding
   ↓
FAISS similarity search
   ↓
Top relevant chunks

The application currently retrieves up to:

TOP_K = 6

relevant chunks.

⸻

🔗 How FAISS Maps Back to Metadata

FAISS stores vectors and returns their index positions.

It does not directly store the complete chunk metadata.

The application maintains the relationship between the FAISS position and the corresponding chunk list.

For example:

FAISS position 0 → chunks[0]
FAISS position 1 → chunks[1]
FAISS position 2 → chunks[2]

Each chunk contains its metadata:

chunk_id
page
text
type

Therefore, when FAISS returns an index position, the application can retrieve the corresponding chunk and its page information.

This allows the final response to provide source information such as:

Page 11
Page 10
Page 13

⸻

🤖 Multimodal Gemini Generation

After semantic retrieval, the application sends two important pieces of information to Gemini:

1. The original uploaded PDF
2. The relevant retrieved text context

This gives Gemini access to the complete document while also providing focused semantic context from FAISS.

This is particularly useful for questions involving:

* text
* tables
* figures
* diagrams
* visual content

The system therefore combines:

Semantic Retrieval
        +
Original PDF
        ↓
Gemini Multimodal Generation

⸻

🎯 Grounded Responses

The generation prompt instructs Gemini to:

* answer using the uploaded PDF
* avoid unsupported external information
* avoid inventing facts
* avoid inventing page numbers
* clearly state when sufficient information cannot be found
* provide relevant page references
* explain concepts clearly
* structure comparisons when required

This helps reduce hallucination and keeps answers grounded in the uploaded document.

⸻

📑 Source References

For every generated answer, the application identifies the relevant retrieved pages.

The response can contain source information such as:

Sources
Page 11
Page 10
Page 13

The user can then select View to see the corresponding page from the uploaded PDF.

This provides traceability between:

Answer
  ↓
Retrieved chunk
  ↓
Page number
  ↓
Original PDF page

⸻

🛠️ Tech Stack

Backend

* Python
* FastAPI
* Uvicorn
* Pydantic

AI / Machine Learning

* Google Gemini API
* FastEmbed
* ONNX Runtime
* FAISS

Document Processing

* PyMuPDF

Frontend

* HTML
* CSS
* JavaScript

Other

* Python-dotenv
* NumPy
* Git / GitHub

⸻

📁 Project Structure

multimodal-rag/
│
├── api.py
├── requirements.txt
├── .env
├── .gitignore
│
├── frontend/
│   └── index.html
│
├── data/
│   └── raw/
│       └── uploads/
│
└── README.md

.env should never be committed to GitHub.

⸻

⚙️ Installation

1. Clone the Repository

git clone <your-github-repository-url>
cd multimodal-rag

⸻

2. Create a Virtual Environment

Windows

python -m venv venv

Activate it:

venv\Scripts\activate

⸻

3. Install Dependencies

pip install -r requirements.txt

⸻

4. Configure Environment Variables

Create a .env file in the project root:

GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MAX_UPLOAD_SIZE_MB=50

Replace:

your_gemini_api_key

with your actual Gemini API key.

Never upload your .env file to GitHub.

⸻

▶️ Running the Application Locally

Start the FastAPI server:

python api.py

The application runs on:

http://127.0.0.1:8000

You can also access the FastAPI documentation at:

http://127.0.0.1:8000/docs

⸻

🔌 API Endpoints

GET /

Serves the frontend application.

⸻

GET /health

Checks whether the API is running and displays information about:

* Gemini configuration
* embedding model
* embedding backend
* loaded documents

⸻

POST /upload

Uploads and processes a PDF.

The endpoint performs:

PDF validation
      ↓
File storage
      ↓
Text extraction
      ↓
Chunking
      ↓
Embeddings
      ↓
FAISS indexing
      ↓
Gemini PDF upload

⸻

POST /ask

Accepts a question and document ID.

Example request:

{
  "query": "What is semantic chunking?",
  "document_id": "your-document-id"
}

The endpoint performs semantic retrieval and generates the final Gemini response.

⸻

GET /source/{document_id}/{page}

Renders a specific page of the uploaded PDF as an image.

Example:

/source/abc123/11

This allows the frontend to display the exact source page.

⸻

🧪 Example Usage

Step 1

Upload a PDF.

Step 2

The application processes the document.

For example:

28 pages
38 semantic chunks
Multimodal AI ready

Step 3

Ask:

What is semantic chunking?

Step 4

The system:

Question
   ↓
Query embedding
   ↓
FAISS retrieval
   ↓
Relevant chunks
   ↓
Gemini + original PDF
   ↓
Generated answer

Step 5

The application displays:

* Answer
* Supporting pages
* Source information
* View buttons for the original pages

⸻

💡 Why This Project?

Traditional PDF search often relies heavily on exact keywords.

This project uses semantic retrieval, meaning the system can retrieve information based on the meaning of the question.

For example, a user may ask:

What does the document say about breaking text into meaningful sections?

even if the document uses the terminology:

semantic chunking

The embedding-based retrieval system can identify the semantic relationship between the question and the relevant document content.

⸻

📌 Key Learning Outcomes

Through this project, I worked with:

* Retrieval-Augmented Generation
* Vector embeddings
* Semantic search
* FAISS
* Document chunking
* Metadata mapping
* PDF processing
* Multimodal LLMs
* Gemini API
* FastAPI
* ONNX-based inference
* API deployment
* Source-grounded AI responses

The project also helped me understand how different components of a modern AI application work together:

Document Processing
        ↓
Embeddings
        ↓
Vector Database / Search
        ↓
Retrieval
        ↓
LLM
        ↓
Grounded Response

⸻

🚀 Future Improvements

Possible future improvements include:

* Persistent vector indexes
* Multi-document search
* Conversation history
* Authentication
* Better table extraction
* Image-specific retrieval
* Streaming responses
* More advanced reranking
* Persistent document storage
* Document management dashboard

⸻

👩‍💻 Author

Manasvee Shegaonkar

Computer Science & Engineering — AI & ML

Interested in:

* Artificial Intelligence
* Machine Learning
* Generative AI
* RAG Systems
* AI/ML Engineering

⸻

⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub!
