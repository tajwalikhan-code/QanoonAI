import os
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------
# Load GROQ API Key
# -----------------------------
import os
from groq import Groq

# Read API key from HuggingFace Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Create Groq client
client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# Load Embedding Model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load Legal Document
# -----------------------------
def load_document(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
# Chunking Function
# -----------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# -----------------------------
# Prepare Data
# -----------------------------
text = load_document("data/pakistan_law.txt")
chunks = chunk_text(text)

embeddings = embedding_model.encode(chunks)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# -----------------------------
# RAG Chat Function
# -----------------------------
def legal_chatbot(query):

    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding).astype("float32"), 3)
    retrieved_chunks = [chunks[i] for i in I[0]]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a Pakistani Legal Information Assistant.
Answer ONLY using the context below.
If the answer is not found, say:
"I do not have enough information in the provided legal documents."
Context:
{context}
Question:
{query}
Answer:
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    sources_text = "\n\nSources:\n"
    for chunk in retrieved_chunks:
        sources_text += f"- {chunk[:250]}...\n"

    return answer + sources_text

# -----------------------------
# Gradio Interface
# -----------------------------
demo = gr.Interface(
    fn=legal_chatbot,
    inputs=gr.Textbox(label="Ask your legal question"),
    outputs=gr.Textbox(label="Answer"),
    title="⚖️ QanoonAI - Pakistani Legal Information Assistant",
    description="This chatbot provides legal information based on Pakistani law documents. For educational purposes only."
)

demo.launch()
