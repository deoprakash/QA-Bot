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

# Page configuration
st.set_page_config(
    page_title="QuickQBot - PDF Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'embeddings_created' not in st.session_state:
    st.session_state.embeddings_created = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

@st.cache_resource
def init_connections():
    """Initialize MongoDB connection and embedding model."""
    try:
        mongoDB_api = st.secrets.get("MONGO_DB_URI")
        if not mongoDB_api:
            st.error("MONGO_DB_URI not found in Streamlit secrets.")
            return None, None, None, None

        client = pymongo.MongoClient(mongoDB_api)
        db = client.pdf_qa_system
        collection = db.documents

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return client, db, collection, model
    except Exception as e:
        st.error(f"Failed to initialize connections: {e}")
        return None, None, None, None


def extract_text_from_pdf(pdf_file) -> List[Dict]:
    """Extract text from PDF and chunk it."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        chunks = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            sentences = text.split('. ')
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk + sentence) < 500:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append({'text': current_chunk.strip(), 'page': page_num + 1, 'source': pdf_file.name})
                    current_chunk = sentence + ". "
            if current_chunk:
                chunks.append({'text': current_chunk.strip(), 'page': page_num + 1, 'source': pdf_file.name})
        return chunks
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return []


def create_embeddings(chunks: List[Dict], model, collection) -> bool:
    """Create embeddings and store in MongoDB."""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        documents_to_insert = []

        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk['text']).tolist()
            doc_hash = hashlib.md5(chunk['text'].encode()).hexdigest()
            document = {
                'text': chunk['text'],
                'page': chunk['page'],
                'source': chunk['source'],
                'embedding': embedding,
                'doc_hash': doc_hash,
                'created_at': datetime.now(),
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
            }
            documents_to_insert.append(document)
            progress_bar.progress((i + 1) / len(chunks))
            status_text.text(f"Processing chunk {i + 1}/{len(chunks)}")

        inserted_count = 0
        for doc in documents_to_insert:
            try:
                if not collection.find_one({'doc_hash': doc['doc_hash']}):
                    collection.insert_one(doc)
                    inserted_count += 1
            except Exception as e:
                st.warning(f"Failed to insert document: {e}")

        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Successfully processed {len(chunks)} chunks. {inserted_count} new documents inserted.")
        st.info("🔄 Refreshing page... Your documents are ready for questions!")
        return True
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return False


def manual_vector_search(query_text: str, model, collection, k: int = 5) -> List[Dict]:
    """Manual vector search using cosine similarity."""
    try:
        query_embedding = np.array(model.encode(query_text))
        documents = list(collection.find({}, {"text": 1, "page": 1, "source": 1, "embedding": 1, "_id": 0}))
        similarities = []
        for doc in documents:
            if 'embedding' in doc and doc['embedding']:
                doc_embedding = np.array(doc['embedding'])
                score = float(cosine_similarity([query_embedding], [doc_embedding])[0][0])
                doc['score'] = score
                similarities.append(doc)
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:k]
    except Exception as e:
        st.error(f"Manual vector search error: {e}")
        return []


def vector_search(query_text: str, model, collection, k: int = 5) -> List[Dict]:
    """Vector search with fallback to manual similarity."""
    try:
        if collection.count_documents({}) == 0:
            st.warning("No documents found in database. Please upload PDFs first.")
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
                {"$project": {"text": 1, "page": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}, "_id": 0}}
            ]
            results = list(collection.aggregate(pipeline))
            if results:
                return results
        except Exception:
            pass

        return manual_vector_search(query_text, model, collection, k)
    except Exception as e:
        st.error(f"Vector search error: {e}")
        return []


def query_groq(prompt: str) -> str:
    """Query Groq API safely."""
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found in Streamlit secrets.")
            return "Error: GROQ_API_KEY not configured."

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "No answer from API.")
    except Exception as e:
        st.error(f"Error querying Groq API: {e}")
        return f"Error querying Groq API: {e}. Returning placeholder answer."

def answer_question(question: str, model, collection) -> tuple:
    """Answer question using RAG, with context control."""
    search_results = vector_search(question, model, collection, k=5)
    if not search_results:
        return "No relevant documents found. Upload PDFs first.", []

    context_parts, total_length, used_sources = [], 0, []
    max_context_length = 7000

    for doc in search_results:
        text = doc.get('text', '')
        if total_length + len(text) > max_context_length:
            break
        page, source, score = doc.get('page', 'Unknown'), doc.get('source', 'Unknown'), doc.get('score', 0)
        context_parts.append(f"Document (Page {page})\nSource: {source}\nContent: {text}\n---")
        total_length += len(text)
        used_sources.append(doc)

    if not context_parts:
        return "Could not fit any documents in context window.", []

    context = "\n".join(context_parts)
    prompt = f"""
Based on the following document excerpts, answer the question. If the answer is not clearly in the documents, say so.

Context:
{context}

Question: {question}

Answer (be concise and cite the source when possible):
"""
    answer = query_groq(prompt)
    return answer, used_sources


def main():
    client, db, collection, model = init_connections()
    
    # ✅ Explicit None checks (do not use all([...]))
    if client is None or db is None or collection is None or model is None:
        st.stop()

    st.title("📚 QuickQBot - PDF Q&A System")
    st.sidebar.title("🔧 System Status")

    doc_count = collection.count_documents({}) if collection is not None else 0
    st.sidebar.metric("📄 Documents in DB", doc_count)

    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "❓ Ask Questions", "📊 Database Info"])

    with tab1:
        st.header("📤 Upload PDFs")
        
        # Show processing completion message if just completed
        if st.session_state.processing_complete:
            st.success("✅ Processing complete! Your documents are ready for questions.")
            st.session_state.processing_complete = False
        
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        if uploaded_files and st.button("🚀 Process PDFs"):
            with st.spinner("Processing PDFs..."):
                all_chunks = []
                for file in uploaded_files:
                    chunks = extract_text_from_pdf(file)
                    all_chunks.extend(chunks)
                if all_chunks and collection is not None:
                    success = create_embeddings(all_chunks, model, collection)
                    if success:
                        st.session_state.processing_complete = True
                        import time
                        time.sleep(1)  # Brief pause to show completion message
                        st.rerun()  # Force immediate refresh

    with tab2:
        st.header("❓ Ask Questions")
        if doc_count == 0:
            st.warning("⚠️ No documents in database. Please upload PDFs in the 'Upload & Process' tab first.")
        else:
            question = st.text_input("Enter your question:")
            if st.button("🔍 Ask") and question:
                answer, sources = answer_question(question, model, collection)
                st.session_state.chat_history.append({'question': question, 'answer': answer, 'sources': sources})
            for chat in reversed(st.session_state.chat_history):
                st.write(f"**Q:** {chat['question']}")
                st.write(f"**A:** {chat['answer']}")

    with tab3:
        st.header("📊 Database Info")
        st.metric("Total documents", doc_count)
        if collection is not None and st.button("🗑️ Clear All Documents"):
            collection.delete_many({})
            st.success("Database cleared.")
            st.rerun()  # Refresh after deletion

if __name__ == "__main__":
    main()
