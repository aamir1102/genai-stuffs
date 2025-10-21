import streamlit as st
from typing import List
from io import BytesIO

# Import the RAG service functions (the simulated backend)
from rag_service import (
    load_document, 
    load_web_url, 
    setup_rag_pipeline, 
    generate_summary,
    get_rag_answer
)

# --- Configuration ---
# Set the page configuration for a professional look
st.set_page_config(
    page_title="Gemini RAG Document & Web Analyzer",
    page_icon="🤖",
    layout="wide"
)

# Initialize Session State for Chat History and RAG components
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "all_documents" not in st.session_state:
    st.session_state.all_documents = []
if "is_initialized" not in st.session_state:
    st.session_state.is_initialized = False
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

# --- Callbacks / UI Handlers ---

@st.cache_resource(show_spinner=False)
def get_cached_rag_pipeline(documents):
    """Caches the heavy RAG setup function from the service layer."""
    if not documents:
        st.warning("No documents to process.")
        return None, None, 0
    
    try:
        llm, rag_chain, num_chunks = setup_rag_pipeline(documents)
        st.success(f"RAG system initialized with {num_chunks} chunks.")
        st.session_state.is_initialized = True
        return llm, rag_chain, num_chunks
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {e}")
        return None, None, 0

def handle_source_processing(uploaded_files, url_input):
    """Callback for the 'Process Sources' button."""
    st.session_state.all_documents = []
    
    if not uploaded_files and not url_input:
        st.error("Please upload at least one file or enter a URL.")
        return

    with st.spinner("Loading documents from sources..."):
        # Load files
        for file in uploaded_files:
            st.session_state.all_documents.extend(load_document(file, file.name))
            
        # Load URL
        if url_input:
            st.session_state.all_documents.extend(load_web_url(url_input))

    # Setup RAG pipeline with all documents
    llm, rag_chain, num_chunks = get_cached_rag_pipeline(st.session_state.all_documents)
    
    st.session_state.llm = llm
    st.session_state.rag_chain = rag_chain
    st.session_state.num_chunks = num_chunks
    st.session_state.messages = [] # Clear chat history on new sources

def summarize_sources_ui():
    """Generates a summary of all loaded documents and updates UI."""
    if not st.session_state.llm or not st.session_state.all_documents:
        st.error("Please process your sources first using the 'Process Sources' button.")
        return
        
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Generating executive summary via RAG service..."):
            try:
                # Call the backend service function
                summary = generate_summary(st.session_state.llm, st.session_state.all_documents)
                
                # Append summary to chat history
                summary_message = f"**Executive Summary:**\n\n{summary}"
                st.session_state.messages.append({"role": "assistant", "content": summary_message})
                st.markdown(summary_message)
                
            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")

# --- Streamlit UI Components ---

# 1. Header and Sidebar
st.title("🤖 Gemini RAG Document & Web Analyzer")

# Sidebar for source management
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, and PPTX (experimental)",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True
    )

    st.header("2. Add Website URL")
    url_input = st.text_input("Enter URL (e.g., https://example.com)", key="url_input")

    st.markdown("---")

    # Button to process all sources
    st.button(
        "⚙️ Process Sources", 
        on_click=lambda: handle_source_processing(uploaded_files, url_input), 
        use_container_width=True,
        type="primary"
    )
    
    st.markdown("---")
    
    # Button for summarization
    st.button(
        "📝 Summarize All Sources", 
        on_click=summarize_sources_ui, 
        use_container_width=True
    )

    if st.session_state.is_initialized:
        st.success(f"Ready: {st.session_state.num_chunks} chunks processed.")
    else:
        st.warning("System not initialized. Click 'Process Sources' to begin.")
        
    st.markdown("---")
    st.caption("Powered by Google Gemini and LangChain.")

# 2. Main Chat Interface
if not st.session_state.is_initialized and not st.session_state.messages:
    st.info("Upload documents or enter a URL in the sidebar and click 'Process Sources' to start chatting.")
else:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents or website..."):
        
        # 1. Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Get RAG response
        if st.session_state.rag_chain:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Searching sources and generating response via RAG service..."):
                    try:
                        # Call the backend service function
                        response_data = get_rag_answer(st.session_state.rag_chain, prompt)
                        answer = response_data["answer"]
                        sources = response_data["sources"]
                        
                        # Format the source information cleanly
                        source_info = "\n\n**Sources:**\n"
                        for src in sources:
                            source_info += f"- **File/URL:** `{src['source']}`, **Chunk/Page:** `{src['page']}`\n"
                        
                        final_response = f"{answer}\n{source_info}"

                        # Display and save response
                        st.markdown(final_response)
                        st.session_state.messages.append({"role": "assistant", "content": final_response})
                        
                    except Exception as e:
                        error_message = f"An error occurred while running RAG: {e}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
        else:
            # Fallback if chat is attempted before initialization
            error_message = "The RAG system is not yet initialized. Please click 'Process Sources' first."
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            
