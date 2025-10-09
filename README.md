# 📚 PDF Q&A System with MongoDB Vector Search

A **Streamlit-based interactive application** that allows users to upload PDF documents, generate **semantic embeddings**, store them in **MongoDB**, and perform **question answering** using **Groq’s Llama 3.1 model**.  
It uses **SentenceTransformers** for embeddings and **cosine similarity** or **MongoDB Atlas Vector Search** for retrieval.

---

## 🚀 Features

✅ Upload multiple PDFs and extract text  
✅ Create and store embeddings in MongoDB  
✅ Perform semantic search and retrieval  
✅ Ask questions and get AI-generated answers from Groq  
✅ Interactive UI built with Streamlit  
✅ MongoDB fallback to manual cosine similarity search  

---

## 🧠 Tech Stack

- **Frontend**: Streamlit  
- **Database**: MongoDB (Atlas or Local)  
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)  
- **Vector Search**: MongoDB `$vectorSearch` or manual cosine similarity  
- **LLM API**: Groq (Llama 3.1 8B Instant)  
- **Utilities**: PyPDF2, dotenv, numpy, sklearn

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/pdf-qa-system.git
cd pdf-qa-system
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Environment Variables
Create a `.env` file in your project root and add:
```bash
MONGO_DB_URI=your_mongodb_connection_string
GROQ_API_KEY=your_groq_api_key
HUGGINGFACE_API_KEY=hugging_face_access_token
```

### 4️⃣ Run Streamlit App
```bash
streamlit run app.py
```

---

## 🧩 Folder Structure
```
pdf-qa-system/
│
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── README.md             # Project documentation
└── data/                 # (Optional) store sample PDFs
```

---

## 💡 How It Works

1. Upload one or more PDF files.  
2. Extract text → split into chunks (~500 chars).  
3. Generate sentence embeddings using SentenceTransformers.  
4. Store embeddings + metadata in MongoDB.  
5. Ask questions — system retrieves similar chunks and queries Groq for summarized answers.

---

## 🧮 Example Workflow

1. Upload your **research paper PDFs**.  
2. Process them (embedding creation).  
3. Ask questions like:  
   > "What is the main finding of this paper?"  
4. The system fetches the most relevant chunks and generates an AI-based answer with context citations.

---

## 🔒 Security Notes

- API keys are loaded via `.env` (never hardcode keys).  
- Supports both **MongoDB Atlas** (cloud) and **local MongoDB**.  

---
## ✨ Author

**[Deo Prakash](https://www.linkedin.com/in/deo-prakash-152265225/)** 

---

## 🧾 License

This project is licensed under the Apache License.

---

## 🤝 Acknowledgements

- [Streamlit](https://streamlit.io/)  
- [SentenceTransformers](https://www.sbert.net/)  
- [Groq API](https://console.groq.com/)  
- [MongoDB Atlas Vector Search](https://www.mongodb.com/products/platform/atlas-vector-search)
