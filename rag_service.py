import os
import tempfile
from typing import List, Tuple, Dict, Any
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# Import required LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain.schema.document import Document
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader

# --- Utility Functions ---

def load_document(uploaded_file: BytesIO, file_name: str) -> List[Document]:
    """Loads content from uploaded files (PDF, DOCX, PPTX)."""
    
    # Use temporary file to allow LangChain loaders access by path
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        if ext == ".pdf":
            print(f"Loading PDF: {file_name}")
            loader = PyPDFLoader(tmp_path)
        elif ext == ".docx":
            print(f"Loading DOCX: {file_name}")
            loader = Docx2txtLoader(tmp_path)
        elif ext == ".pptx":
            print(f"Loading PPTX: {file_name} (Requires 'unstructured' dependencies)")
            loader = UnstructuredPowerPointLoader(tmp_path)
        else:
            print(f"Skipping unsupported file type: {file_name}")
            return []
        
        return loader.load()
        
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return []
        
    finally:
        os.remove(tmp_path)

def load_web_url(url: str) -> List[Document]:
    """Loads content from a website URL."""
    try:
        print(f"Loading content from URL: {url}")
        loader = WebBaseLoader(url)
        return loader.load()
    except Exception as e:
        print(f"Error loading URL {url}: {e}")
        return []

# --- Main Service Functions ---

def setup_rag_pipeline(all_documents: List[Document]) -> Tuple[ChatGoogleGenerativeAI, Any, int]:
    """
    Creates embeddings, vector store (FAISS), and the RAG chain.
    Returns (llm, rag_chain, num_chunks).
    """
    if not all_documents:
        raise ValueError("No documents provided for RAG setup.")

    print("Setting up RAG pipeline...")
    
    # 1. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(all_documents)

    # 2. Initialize Embeddings and LLM
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

    # 3. Create FAISS Vector Store
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 4. Define the RAG Prompt Template
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert document analysis assistant named Gemini. Use the following context 
    to answer the user's question accurately and concisely. Always reference the source document 
    or URL if possible, or state if the information is not in the context.
    
    Context: {context}
    
    Question: {input}
    """)

    # 5. Create Document Chain and Retrieval Chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print(f"RAG system initialized with {len(docs)} chunks.")
    return llm, retrieval_chain, len(docs)

def generate_summary(llm: ChatGoogleGenerativeAI, documents: List[Document]) -> str:
    """Generates an executive summary of all loaded documents."""
    
    # Combine all document content into a single string
    full_text = "\n\n".join([doc.page_content for doc in documents])
    
    # Define a focused summarization prompt (limit content length for prompt safety)
    summary_prompt = f"""
    Please provide a comprehensive and concise executive summary of all the loaded content. 
    Focus on the main topics, key findings, and critical details mentioned. 
    The summary should be formatted as a short report.
    
    DOCUMENTS CONTENT (partial view):
    ---
    {full_text[:6000]} 
    ---
    """
    
    response = llm.invoke(summary_prompt)
    return response.content

def get_rag_answer(rag_chain: Any, question: str) -> Dict[str, Any]:
    """Retrieves context and generates an answer using the RAG chain."""
    response = rag_chain.invoke({"input": question})
    
    # Process sources for a clean list
    sources = []
    for doc in response["context"]:
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "N/A")
        sources.append({
            "source": os.path.basename(source),
            "page": page
        })
        
    return {
        "answer": response["answer"],
        "sources": sources
    }
