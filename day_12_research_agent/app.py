import streamlit as st
import os
from pathlib import Path
import time
from res import RAGSystem
import sys
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px dashed #dee2e6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'indexed' not in st.session_state:
    st.session_state.indexed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_path' not in st.session_state:
    st.session_state.documents_path = "documents"
if 'auto_indexed' not in st.session_state:
    st.session_state.auto_indexed = False

# Helper function to initialize and index
def initialize_system(documents_path, model_name, response_style, k_docs, 
                      similarity_threshold, chunk_size, chunk_overlap, force_reload=False):
    """Initialize RAG system and index documents"""
    try:
        # Create documents directory if it doesn't exist
        os.makedirs(documents_path, exist_ok=True)
        
        # Initialize RAG system
        if st.session_state.rag_system is None:
            with st.spinner("🔧 Initializing RAG system..."):
                st.session_state.rag_system = RAGSystem(use_local_models=True)
        
        # Check if there are any documents to index
        files = []
        if os.path.exists(documents_path):
            for filename in os.listdir(documents_path):
                if filename.endswith(('.pdf', '.txt')):
                    files.append(filename)
        
        if not files:
            return False, "No documents found in the directory. Please upload some documents first."
        
        # Index documents
        with st.spinner(f"📚 Indexing {len(files)} document(s)... This may take a few minutes."):
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = output_capture = StringIO()
            
            st.session_state.rag_system.index_documents(
                documents_path,
                force_reload=force_reload,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            sys.stdout = old_stdout
            output = output_capture.getvalue()
        
        # Setup QA chain
        with st.spinner(f"🤖 Loading {model_name} model..."):
            st.session_state.rag_system.setup_qa_chain(
                k=k_docs,
                model_name=model_name,
                response_style=response_style,
                similarity_threshold=similarity_threshold
            )
        
        st.session_state.indexed = True
        return True, output
        
    except Exception as e:
        return False, f"Error: {str(e)}"

# Header
st.markdown('<p class="main-header">📚 RAG Document Q&A System</p>', unsafe_allow_html=True)

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    st.subheader("🤖 Model Settings")
    model_name = st.selectbox(
        "Select LLM Model",
        [
            "google/flan-t5-base",
            "google/flan-t5-large",
            "google/flan-t5-xl",
        ],
        index=1,
        help="Larger models provide better quality but require more memory"
    )
    
    response_style = st.selectbox(
        "Response Style",
        ["conversational", "professional", "concise", "detailed", "generic"],
        index=0,
        help="Choose how the AI should respond"
    )
    
    # Retrieval settings
    st.subheader("🔍 Retrieval Settings")
    k_docs = st.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=10,
        value=4,
        help="More documents = more context but slower"
    )
    
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower = stricter filtering"
    )
    
    # Document settings
    st.subheader("📄 Document Processing")
    chunk_size = st.number_input(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Size of text chunks for processing"
    )
    
    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="Overlap between chunks to preserve context"
    )
    
    st.divider()
    
    # System status
    st.subheader("📊 System Status")
    if st.session_state.indexed:
        st.success("✅ System Ready")
        if st.session_state.rag_system and st.session_state.rag_system.indexed_files:
            st.metric("Indexed Files", len(st.session_state.rag_system.indexed_files))
    else:
        st.warning("⚠️ Upload documents to start")
    
    st.divider()
    
    # Re-index button
    if st.session_state.indexed:
        if st.button("🔄 Re-index Documents", use_container_width=True):
            success, message = initialize_system(
                st.session_state.documents_path,
                model_name,
                response_style,
                k_docs,
                similarity_threshold,
                chunk_size,
                chunk_overlap,
                force_reload=True
            )
            if success:
                st.success("✅ Documents re-indexed successfully!")
                st.rerun()
            else:
                st.error(message)
    
    # Clear history button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Documents", "ℹ️ About"])

# Tab 1: Chat Interface
with tab1:
    # Auto-initialize if documents exist but system not indexed
    if not st.session_state.auto_indexed:
        documents_path = st.session_state.documents_path
        if os.path.exists(documents_path):
            files = [f for f in os.listdir(documents_path) if f.endswith(('.pdf', '.txt'))]
            if files and not st.session_state.indexed:
                st.info(f"📚 Found {len(files)} document(s). Initializing system automatically...")
                success, message = initialize_system(
                    documents_path,
                    model_name,
                    response_style,
                    k_docs,
                    similarity_threshold,
                    chunk_size,
                    chunk_overlap
                )
                if success:
                    st.success("✅ System initialized successfully!")
                    st.session_state.auto_indexed = True
                    st.rerun()
                else:
                    st.error(message)
    
    if not st.session_state.indexed:
        st.info("👉 Please upload documents in the 'Documents' tab to get started")
    else:
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**🧑 You:** {question}")
                st.markdown(f"**🤖 Assistant:** {answer}")
                st.divider()
        
        # Question input
        with st.form(key="question_form", clear_on_submit=True):
            user_question = st.text_input(
                "Ask a question about your documents:",
                placeholder="e.g., What are the main topics covered in the documents?",
                key="user_question"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                submit_button = st.form_submit_button("🚀 Ask", use_container_width=True)
            with col2:
                show_sources = st.checkbox("Show sources", value=True)
        
        if submit_button and user_question:
            with st.spinner("🔍 Searching documents and generating response..."):
                try:
                    # Get response
                    response = st.session_state.rag_system.query(
                        user_question,
                        return_sources=show_sources
                    )
                    
                    # Display answer
                    st.markdown("### 💡 Answer:")
                    st.success(response['answer'])
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_question, response['answer']))
                    
                    # Show sources if requested
                    if show_sources and response.get('sources'):
                        with st.expander(f"📚 View {len(response['sources'])} Source(s)", expanded=False):
                            for idx, source in enumerate(response['sources'], 1):
                                st.markdown(f"**Source {idx}:** {source['metadata'].get('source_file', 'Unknown')}")
                                st.markdown(f"**Distance:** {source['metadata'].get('distance', 'N/A')}")
                                st.markdown(f"**Similarity:** {source['metadata'].get('similarity', 'N/A')}")
                                with st.container():
                                    st.text_area(
                                        f"Content {idx}",
                                        value=source['content'][:500] + "..." if len(source['content']) > 500 else source['content'],
                                        height=150,
                                        key=f"source_{idx}_{time.time()}"
                                    )
                                st.divider()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Tab 2: Document Management
with tab2:
    st.header("📁 Document Management")
    
    # Documents path configuration
    documents_path = st.text_input(
        "Documents Directory",
        value=st.session_state.documents_path,
        help="Folder where documents are stored"
    )
    
    if documents_path != st.session_state.documents_path:
        st.session_state.documents_path = documents_path
        st.session_state.indexed = False
        st.session_state.auto_indexed = False
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Documents")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Select one or more documents to upload"
        )
        
        if uploaded_files:
            st.info(f"📎 {len(uploaded_files)} file(s) selected")
            
            if st.button("💾 Save and Index Documents", use_container_width=True, type="primary"):
                # Create documents directory if it doesn't exist
                os.makedirs(documents_path, exist_ok=True)
                
                # Save uploaded files
                with st.spinner("💾 Saving files..."):
                    saved_files = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(documents_path, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_files.append(uploaded_file.name)
                    
                    st.success(f"✅ Saved {len(saved_files)} file(s)")
                
                # Automatically index the documents
                success, message = initialize_system(
                    documents_path,
                    model_name,
                    response_style,
                    k_docs,
                    similarity_threshold,
                    chunk_size,
                    chunk_overlap,
                    force_reload=False
                )
                
                if success:
                    st.success("✅ Documents indexed successfully!")
                    with st.expander("📝 Indexing Log"):
                        st.code(message)
                    st.balloons()
                    st.session_state.auto_indexed = True
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Current Documents")
        
        if st.session_state.rag_system and st.session_state.rag_system.indexed_files:
            indexed_files = st.session_state.rag_system.indexed_files
            
            st.metric("Total Files", len(indexed_files))
            
            st.markdown("---")
            
            # Display files in a nice format
            for filepath in indexed_files:
                filename = os.path.basename(filepath)
                file_type = "📄 PDF" if filepath.endswith('.pdf') else "📝 TXT"
                
                col_icon, col_name, col_action = st.columns([1, 4, 1])
                with col_icon:
                    st.markdown(file_type)
                with col_name:
                    st.markdown(f"**{filename}**")
                with col_action:
                    if st.button("🗑️", key=f"delete_{filename}"):
                        try:
                            os.remove(filepath)
                            st.success(f"Deleted {filename}")
                            st.session_state.indexed = False
                            st.session_state.auto_indexed = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {e}")
        else:
            st.info("📭 No documents indexed yet")
            
            # Show files in directory even if not indexed
            if os.path.exists(documents_path):
                files = [f for f in os.listdir(documents_path) if f.endswith(('.pdf', '.txt'))]
                if files:
                    st.warning(f"⚠️ Found {len(files)} file(s) in directory but not indexed")
                    if st.button("🔄 Index These Files", use_container_width=True):
                        success, message = initialize_system(
                            documents_path,
                            model_name,
                            response_style,
                            k_docs,
                            similarity_threshold,
                            chunk_size,
                            chunk_overlap
                        )
                        if success:
                            st.success("✅ Files indexed!")
                            st.rerun()
                        else:
                            st.error(message)

# Tab 3: About
with tab3:
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### 📚 RAG Document Q&A System
    
    This is a **Retrieval-Augmented Generation (RAG)** system that allows you to ask questions about your documents.
    
    #### 🎯 How It Works:
    
    1. **📤 Upload Documents**: Add your PDF or TXT files using the Documents tab
    2. **⚡ Auto-Index**: The system automatically processes and creates vector embeddings
    3. **💬 Ask Questions**: Type natural language questions about your documents
    4. **🤖 Get Answers**: The AI retrieves relevant information and generates answers
    
    #### ✨ Features:
    
    - **🚀 Automatic Indexing**: Just upload files and start asking questions
    - **Smart Retrieval**: Uses semantic search to find relevant document sections
    - **🎯 Relevance Filtering**: Only uses highly relevant content for answers
    - **🤖 Multiple Models**: Choose from different LLM models based on your needs
    - **⚙️ Customizable**: Adjust retrieval parameters, response style, and more
    - **📚 Source Tracking**: See which documents were used to generate each answer
    
    #### 🔧 Model Comparison:
    
    | Model | Quality | Speed | Memory |
    |-------|---------|-------|--------|
    | flan-t5-base | Basic | Fast ⚡ | 1GB |
    | flan-t5-large | Good ✅ | Medium | 3GB |
    | flan-t5-xl | Excellent ✅✅ | Slow 🐌 | 6GB |
    
    #### 📖 Tips for Best Results:
    
    - Use **flan-t5-large** or larger for better quality answers
    - Set **relevance threshold** to 0.5-0.7 for focused queries
    - Ask **specific questions** rather than broad ones
    - Use **generic style** for natural responses
    
    #### 🚀 Quick Start Guide:
    
    1. Go to the **Documents** tab
    2. Upload your PDF or TXT files
    3. Click "Save and Index Documents"
    4. Go to the **Chat** tab
    5. Start asking questions!
    
    #### 🛠️ Technology Stack:
    
    - **LangChain**: RAG pipeline
    - **Chroma**: Vector database
    - **HuggingFace**: Embeddings and LLMs
    - **Streamlit**: User interface
    
    ---
    
    **Made with ❤️ using LangChain and Streamlit**
    """)
    
    # System information
    with st.expander("🔍 Current Configuration"):
        config_data = {
            "Model": model_name,
            "Response Style": response_style,
            "Documents to Retrieve": k_docs,
            "Relevance Threshold": similarity_threshold,
            "Chunk Size": chunk_size,
            "Chunk Overlap": chunk_overlap,
            "Documents Path": documents_path,
            "System Indexed": st.session_state.indexed,
            "Number of Files": len(st.session_state.rag_system.indexed_files) if st.session_state.rag_system else 0
        }
        st.json(config_data)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
    💡 <strong>Quick Start:</strong> Upload documents → Automatic indexing → Start chatting!
    </div>
    """,
    unsafe_allow_html=True
)