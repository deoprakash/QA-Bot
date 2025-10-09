import streamlit as st
import pymongo
from dotenv import load_dotenv
import os
import requests
from sentence_transformers import SentenceTransformer
import PyPDF2
import io
from typing import List, Dict
import hashlib
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'embeddings_created' not in st.session_state:
    st.session_state.embeddings_created = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def init_connections():
    """Initialize MongoDB connection and embedding model."""
    try:
        # MongoDB connection
        mongoDB_api = os.getenv("MONGO_DB_URI")
        client = pymongo.MongoClient(mongoDB_api)
        db = client.pdf_qa_system
        collection = db.documents

        # Embedding model
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
                        chunks.append({
                            'text': current_chunk.strip(),
                            'page': page_num + 1,
                            'source': pdf_file.name
                        })
                    current_chunk = sentence + ". "
            if current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'page': page_num + 1,
                    'source': pdf_file.name
                })
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
        st.success(f"Successfully processed {len(chunks)} chunks. {inserted_count} new documents inserted.")
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
    """Query Groq API safely with error handling and correct payload."""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found in environment variables.")
            return "Error: GROQ_API_KEY not configured."

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 400:
            error_detail = response.json().get('error', {}).get('message', 'No details')
            st.error(f"Groq API Bad Request (400): {error_detail}")
            return f"Error: The request was malformed. Details: {error_detail}"
        
        response.raise_for_status() 
        
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "No answer from API.")

    except requests.exceptions.HTTPError as http_err:
        st.error(f"Groq API HTTP Error: {http_err} - {response.text}")
        return f"Error querying Groq API: {http_err}. Returning placeholder answer."
    except Exception as e:
        st.error(f"An unexpected error occurred while querying Groq: {e}")
        return f"Error querying Groq API: {e}. Returning placeholder answer."
    
def answer_question(question: str, model, collection) -> tuple:
    """Answer question using RAG, with controlled context size."""
    # Perform vector search, retrieve a few more documents than we might need
    search_results = vector_search(question, model, collection, k=5) 
    
    if not search_results:
        return "No relevant documents found. Please upload and process some PDF documents first.", []
    
    # Prepare context, but dynamically limit its size to avoid API errors
    context_parts = []
    total_context_length = 0
    # The llama-3.1-8b-instant model has an 8k token context window. 
    # We'll use a character limit as a safe approximation.
    max_context_length = 7000 

    used_sources = []
    for doc in search_results:
        text = doc.get('text', '')
        
        # Stop adding documents if the context is getting too long
        if total_context_length + len(text) > max_context_length:
            break 

        page = doc.get('page', 'Unknown')
        source = doc.get('source', 'Unknown')
        score = doc.get('score', 0)
        
        context_parts.append(f"Document (Page {page}")
        context_parts.append(f"Source: {source}")
        context_parts.append(f"Content: {text}")
        context_parts.append("---")
        
        total_context_length += len(text)
        used_sources.append(doc)
    
    if not context_parts:
         return "Could not find any documents small enough to fit in the context window.", []

    context = "\n".join(context_parts)
    
    # Create prompt
    prompt = f"""
Based on the following document excerpts, answer the question. If the answer is not clearly in the documents, say so.

Context:
{context}

Question: {question}

Answer (be concise and cite the source when possible):
"""
    
    # Get answer from Groq
    answer = query_groq(prompt)
    
    # Return the answer and only the sources that were actually used in the prompt
    return answer, used_sources

def main():
    client, db, collection, model = init_connections()
    if client is None or db is None or collection is None or model is None:
        st.error("Failed to initialize connections.")
        st.stop()

    st.title("📚 PDF Q&A System with MongoDB Vector Search")
    st.sidebar.title("🔧 System Status")

    try:
        doc_count = collection.count_documents({})
        st.sidebar.metric("📄 Documents in DB", doc_count)
    except:
        doc_count = 0

    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "❓ Ask Questions", "📊 Database Info"])

    with tab1:
        st.header("📤 Upload PDFs")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            st.write(f"{len(uploaded_files)} file(s) uploaded")
            if st.button("🚀 Process PDFs"):
                all_chunks = []
                for file in uploaded_files:
                    chunks = extract_text_from_pdf(file)
                    all_chunks.extend(chunks)
                if all_chunks:
                    create_embeddings(all_chunks, model, collection)

    with tab2:
        st.header("❓ Ask Questions")
        if doc_count == 0:
            st.warning("No documents in DB.")
        else:
            question = st.text_input("Enter your question:")
            if st.button("🔍 Ask") and question:
                answer, sources = answer_question(question, model, collection)
                st.session_state.chat_history.append({'question': question, 'answer': answer, 'sources': sources})
            if st.session_state.chat_history:
                for chat in reversed(st.session_state.chat_history):
                    st.write(f"**Q:** {chat['question']}")
                    st.write(f"**A:** {chat['answer']}")

    with tab3:
        st.header("📊 Database Info")
        st.write(f"Total documents: {doc_count}")
        if st.button("🗑️ Clear All Documents"):
            collection.delete_many({})
            st.success("Database cleared.")

if __name__ == "__main__":
    main()
