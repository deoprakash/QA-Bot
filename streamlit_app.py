import streamlit as st
import pymongo
from sentence_transformers import SentenceTransformer
import PyPDF2
import io
from typing import List, Dict
import hashlib
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

# --------------------------------------------------------------------
# 🌟 Streamlit App Configuration
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Quick QBot - PDF Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------
# 🧠 Initialize Session State Variables
# --------------------------------------------------------------------
if "embeddings_created" not in st.session_state:
    st.session_state.embeddings_created = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Upload"

# --------------------------------------------------------------------
# ⚙️ Initialize MongoDB & Embedding Model
# --------------------------------------------------------------------
@st.cache_resource
def init_connections():
    try:
        mongoDB_api = st.secrets.get("MONGO_DB_URI")
        if not mongoDB_api:
            st.error("❌ MONGO_DB_URI not found in Streamlit secrets.")
            return None, None, None, None

        client = pymongo.MongoClient(mongoDB_api, serverSelectionTimeoutMS=10000)
        db = client.pdf_qa_system
        collection = db.documents

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return client, db, collection, model
    except Exception as e:
        st.error(f"⚠️ Failed to initialize connections: {e}")
        return None, None, None, None

# --------------------------------------------------------------------
# 📖 PDF Text Extraction
# --------------------------------------------------------------------
def extract_text_from_pdf(pdf_file) -> List[Dict]:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        chunks = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            sentences = text.split(". ")
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk + sentence) < 500:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "page": page_num + 1,
                            "source": pdf_file.name
                        })
                    current_chunk = sentence + ". "
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "page": page_num + 1,
                    "source": pdf_file.name
                })
        return chunks
    except Exception as e:
        st.error(f"❌ Error extracting text from PDF: {e}")
        return []

# --------------------------------------------------------------------
# 🧩 Create and Store Embeddings
# --------------------------------------------------------------------
def create_embeddings(chunks: List[Dict], model, collection) -> bool:
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        docs_to_insert = []

        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk["text"]).tolist()
            doc_hash = hashlib.md5(chunk["text"].encode()).hexdigest()
            document = {
                "text": chunk["text"],
                "page": chunk["page"],
                "source": chunk["source"],
                "embedding": embedding,
                "doc_hash": doc_hash,
                "created_at": datetime.now(),
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            }
            docs_to_insert.append(document)
            progress_bar.progress((i + 1) / len(chunks))
            status_text.text(f"Processing chunk {i + 1}/{len(chunks)}")

        inserted_count = 0
        for doc in docs_to_insert:
            if not collection.find_one({"doc_hash": doc["doc_hash"]}):
                collection.insert_one(doc)
                inserted_count += 1

        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ {len(chunks)} chunks processed. {inserted_count} new documents added.")
        st.session_state.pdf_processed = True
        st.session_state.active_tab = "Ask"
        return True
    except Exception as e:
        st.error(f"⚠️ Error creating embeddings: {e}")
        return False

# --------------------------------------------------------------------
# 🔍 Vector Search (with Fallback)
# --------------------------------------------------------------------
def manual_vector_search(query_text, model, collection, k=5):
    try:
        query_embedding = np.array(model.encode(query_text))
        docs = list(collection.find({}, {"text": 1, "page": 1, "source": 1, "embedding": 1, "_id": 0}))
        scored = []
        for doc in docs:
            if "embedding" in doc and doc["embedding"]:
                score = float(cosine_similarity([query_embedding], [np.array(doc["embedding"])])[0][0])
                doc["score"] = score
                scored.append(doc)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]
    except Exception as e:
        st.error(f"⚠️ Manual vector search error: {e}")
        return []

def vector_search(query_text, model, collection, k=5):
    try:
        if collection.count_documents({}) == 0:
            st.warning("⚠️ No documents found. Upload PDFs first.")
            return []

        try:
            query_embedding = model.encode(query_text).tolist()
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_search_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,
                        "limit": k
                    }
                },
                {"$project": {"text": 1, "page": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}, "_id": 0}},
            ]
            results = list(collection.aggregate(pipeline))
            if results:
                return results
        except Exception:
            pass
        return manual_vector_search(query_text, model, collection, k)
    except Exception as e:
        st.error(f"⚠️ Vector search error: {e}")
        return []

# --------------------------------------------------------------------
# 🤖 Query Groq API
# --------------------------------------------------------------------
def query_groq(prompt: str) -> str:
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            return "❌ GROQ_API_KEY not found in Streamlit secrets."

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        res = requests.post(url, headers=headers, json=data, timeout=30)
        res.raise_for_status()
        return res.json().get("choices", [{}])[0].get("message", {}).get("content", "No response.")
    except Exception as e:
        return f"⚠️ Error querying Groq API: {e}"

# --------------------------------------------------------------------
# 🧠 Answer Question
# --------------------------------------------------------------------
def answer_question(question, model, collection):
    search_results = vector_search(question, model, collection, k=5)
    if not search_results:
        return "No relevant documents found.", []

    context_parts, total_length = [], 0
    for doc in search_results:
        text = doc.get("text", "")
        if total_length + len(text) > 7000:
            break
        context_parts.append(f"Page {doc.get('page')} - {doc.get('source')}:\n{text}\n---")
        total_length += len(text)

    context = "\n".join(context_parts)
    prompt = f"""
Based on the following document excerpts, answer the question.
If the answer is not clearly in the text, say so.

Context:
{context}

Question: {question}

Answer (concise and factual):
"""
    return query_groq(prompt), search_results

# --------------------------------------------------------------------
# 🏁 Main App Layout
# --------------------------------------------------------------------
def main():
    client, db, collection, model = init_connections()
    if client is None or db is None or collection is None or model is None:
        st.stop()

    st.title("📚 Quick QBot — PDF Q&A System")
    st.sidebar.title("🔧 System Status")

    try:
        doc_count = collection.count_documents({})
    except Exception:
        doc_count = 0
    st.sidebar.metric("📄 Documents in DB", doc_count)

    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "❓ Ask Questions", "📊 Database Info"])

    # ---------------- Upload Tab ----------------
    with tab1:
        st.header("📤 Upload PDFs")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        if uploaded_files and st.button("🚀 Process PDFs"):
            all_chunks = []
            for f in uploaded_files:
                all_chunks.extend(extract_text_from_pdf(f))
            if all_chunks:
                create_embeddings(all_chunks, model, collection)

    # ---------------- Ask Tab ----------------
    with tab2:
        st.header("💬 Ask Questions")
        if doc_count == 0:
            st.warning("⚠️ No documents found.")
        else:
            question = st.text_input("Enter your question:")
            if st.button("🔍 Ask") and question:
                answer, sources = answer_question(question, model, collection)
                st.session_state.chat_history.append({"question": question, "answer": answer})
                st.session_state.active_tab = "Ask"

            for chat in reversed(st.session_state.chat_history):
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")

    # ---------------- Database Tab ----------------
    with tab3:
        st.header("📊 Database Info")
        st.write(f"Total documents: {doc_count}")
        if st.button("🗑️ Clear All Documents"):
            collection.delete_many({})
            st.success("✅ Database cleared.")

# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
