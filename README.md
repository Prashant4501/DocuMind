# 🧠 DocuMind

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-orange.svg)
![Groq](https://img.shields.io/badge/Groq-Llama_3-f55036.svg)

**DocuMind** is a blazing-fast Retrieval-Augmented Generation (RAG) backend API. It ingests your personal documents (PDFs and Text files), intelligently chunks and embeds them into a local vector database, and uses the ultra-fast Groq API (Llama 3) to answer questions based *strictly* on your documents.

---

## 🚀 How It Works (Architecture)

1. **Document Loading**: Reads `.pdf` and `.txt` files from the `documents loader/` directory.
2. **Text Splitting**: Splits large documents into manageable overlapping chunks.
3. **Embedding**: Converts text into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector Store**: Stores embeddings locally using **ChromaDB**.
5. **Retrieval & Generation**: When a user asks a question via the FastAPI `/ask` endpoint, it retrieves the most relevant document chunks and passes them as context to **Groq (Llama 3)** to generate a precise answer.

---

## 🛠️ Prerequisites

- Python 3.11 or higher
- A free [Groq API Key](https://console.groq.com/keys)
- A free [HuggingFace API Token](https://huggingface.co/settings/tokens)

---

## ⚙️ Setup & Installation

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/DocuMind.git
cd DocuMind
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

**3. Configure Environment Variables:**
Create a `.env` file in the root directory and add your API keys:
```env
GROQ_API_KEY="your_groq_api_key_here"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_api_token_here"
```

---

## 💻 Usage

### 1. Ingest Your Documents
Place your PDFs or text files inside the `documents loader/` folder. Then, build your vector database by running:
```bash
python create_database.py
```
*(You should see a message confirming the database was created successfully in the `chroma-db/` folder).*

### 2. Start the API Server
Launch the FastAPI backend:
```bash
python -m uvicorn api:app --reload
```
The server will start at `http://127.0.0.1:8000`.

### 3. Ask Questions!
Go to the interactive Swagger UI dashboard in your browser:
**👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

You can click the `POST /ask` endpoint, hit **Try it out**, and send a JSON query like this:
```json
{
  "query": "What are the main concepts in the deep learning document?"
}
```

---

## 📁 Project Structure

```text
📦 DocuMind
 ┣ 📂 documents loader/    # Place your raw .txt and .pdf files here
 ┣ 📂 chroma-db/           # Auto-generated local vector database
 ┣ 📜 api.py               # FastAPI server and endpoint routing
 ┣ 📜 create_database.py   # Ingestion script for parsing and embedding files
 ┣ 📜 main.py              # CLI testing script for LangChain pipeline
 ┣ 📜 requirements.txt     # Python dependencies
 ┗ 📜 .env                 # API Keys (Ignored by Git)
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
