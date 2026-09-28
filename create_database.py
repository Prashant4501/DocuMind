import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Initialize Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_db():
    print("Loading documents...")
    documents = []
    
    # Load notes.txt
    txt_path = "documents loader/notes.txt"
    if os.path.exists(txt_path):
        txt_loader = TextLoader(txt_path)
        documents.extend(txt_loader.load())
        print(f"Loaded {txt_path}")
        
    # Optional: Load PDFs if they exist (deeplearning.pdf, GRU.pdf)
    pdf_paths = ["documents loader/deeplearning.pdf", "documents loader/GRU.pdf"]
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            pdf_loader = PyPDFLoader(pdf_path)
            documents.extend(pdf_loader.load())
            print(f"Loaded {pdf_path}")

    if not documents:
        print("No documents found to index.")
        return

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks.")
    
    print("Creating Chroma vector store and generating embeddings...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma-db"
    )
    
    print("Database created successfully at 'chroma-db/'")

if __name__ == "__main__":
    create_db()
