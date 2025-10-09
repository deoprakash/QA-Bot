# MongoDB Vector Search Index Configuration

To use the Streamlit app with vector search, you need to create a vector search index in MongoDB Atlas.

## Step 1: Create Vector Search Index

1. Go to MongoDB Atlas Dashboard
2. Navigate to your cluster → Search → Create Search Index
3. Choose "JSON Editor"
4. Use this configuration:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "source"
    },
    {
      "type": "filter", 
      "path": "page"
    }
  ]
}
```

## Step 2: Index Name

- **Index Name:** `vector_search_index`
- **Database:** `pdf_qa_system`
- **Collection:** `documents`

## Step 3: Environment Variables

Create/update your `.env` file:

```env
MONGO_DB_URI=your_mongodb_connection_string
GROQ_API_KEY=your_groq_api_key
```

## Step 4: Run the Application

```bash
# Install requirements
pip install -r streamlit_requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

## Features

### 📤 Upload & Process Tab
- Upload multiple PDF files
- Extract text and create chunks
- Generate embeddings using sentence-transformers
- Store in MongoDB with deduplication

### ❓ Ask Questions Tab
- Interactive Q&A interface
- Vector search through documents
- AI-powered answers using Groq
- Chat history with sources
- Source attribution with scores

### 📊 Database Info Tab
- Collection statistics
- Recent documents view
- Database management tools
- Clear all documents option

## Vector Search Fallback

If vector search fails (index not configured), the app falls back to MongoDB text search.

## Troubleshooting

1. **Vector Search Errors:** Ensure the vector index is created correctly
2. **Embedding Errors:** Check sentence-transformers installation
3. **PDF Errors:** Verify PyPDF2 installation and PDF file integrity
4. **API Errors:** Confirm GROQ_API_KEY is valid

## Database Schema

Each document stored has:
- `text`: Extracted text chunk
- `page`: Page number from PDF
- `source`: Original PDF filename
- `embedding`: 384-dimensional vector
- `doc_hash`: MD5 hash for deduplication
- `created_at`: Timestamp
- `embedding_model`: Model used for embeddings