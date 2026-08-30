import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


def generate_answer(query, context):

    prompt = f"""
You are PDF Insights, an AI assistant that answers questions from a technical book.

Your job is to provide a detailed, accurate, well-structured answer based ONLY on the
provided retrieved context.

IMPORTANT RULES:

1. Use ONLY the provided context.
2. Do NOT invent information that is not present in the context.
3. Give a detailed explanation rather than a one-sentence answer.
4. For technical questions, explain:
   - the definition
   - the main concept
   - how it works
   - important components or steps
   - examples when the context contains them
5. Use paragraphs, bullet points, and numbered lists when they improve readability.
6. If the question asks "what", explain what it is.
7. If it asks "how", explain the process step-by-step.
8. If it asks "why", explain the reasoning.
9. If the context contains equations, terminology, examples, tables, or important details,
   include the relevant information.
10. Do not mention that you are using a context or retrieval system.
11. If the answer genuinely cannot be found in the provided context, say:
   "I could not find the answer in the provided book."

Retrieved context:
-------------------------
{context}
-------------------------

User question:
{query}

Detailed answer:
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 800
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"].strip()