from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG API", description="API for RAG project using Groq and ChromaDB")

# Request and Response Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    context: list[str]

# 1. Initialize Embedding Model (same as in db.py)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load existing Vector Store (ChromaDB)
vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

# Create a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Initialize LLM (using Groq)
llm = ChatGroq(model="llama3-8b-8192")

# 4. Create Prompt Template
template = """Answer the question based only on the following context:

{context}

Question: {question}

If you don't know the answer based on the context, just say that you don't know.
"""
prompt = ChatPromptTemplate.from_template(template)

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        # Retrieve documents to include in the response
        docs = retriever.invoke(request.query)
        context_texts = [doc.page_content for doc in docs]
        
        # Generate answer using the RAG chain
        answer = rag_chain.invoke(request.query)
        
        return QueryResponse(answer=answer, context=context_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG API. Use the /ask endpoint to post queries or /docs to see the Swagger UI."}
