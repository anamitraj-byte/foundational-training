"""
RAG (Retrieval-Augmented Generation) System using LangChain

This implementation covers document loading, text splitting, embeddings,
vector storage, and retrieval-based question answering.
"""

# Required installations:
# pip install langchain langchain-community langchain-text-splitters langchain-huggingface chromadb sentence-transformers pypdf python-dotenv

import os
from typing import List
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

# Load environment variables
load_dotenv()

class RAGSystem:
    """Complete RAG system with document loading, indexing, and querying."""
    
    def __init__(self, persist_directory: str = "./chroma_db", use_local_models: bool = True):
        """
        Initialize the RAG system.
        
        Args:
            persist_directory: Directory to store the vector database
            use_local_models: Whether to use local HuggingFace models (True) or OpenAI (False)
        """
        self.persist_directory = persist_directory
        self.use_local_models = use_local_models
        
        if use_local_models:
            # Use local HuggingFace embeddings (no API key needed)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            # Initialize LLM later when needed
            self.llm = None
        else:
            # Use OpenAI (requires API key)
            from langchain_openai import OpenAIEmbeddings, ChatOpenAI
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self, source_path: str, file_type: str = "pdf") -> List:
        """
        Load documents from a file or directory.
        
        Args:
            source_path: Path to file or directory
            file_type: Type of files to load (pdf, txt)
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        if os.path.isdir(source_path):
            if file_type == "pdf":
                # Load all PDFs in directory
                for filename in os.listdir(source_path):
                    if filename.endswith('.pdf'):
                        filepath = os.path.join(source_path, filename)
                        try:
                            loader = PyPDFLoader(filepath)
                            docs = loader.load()
                            documents.extend(docs)
                            print(f"✓ Loaded {filename}: {len(docs)} pages")
                        except Exception as e:
                            print(f"✗ Error loading {filename}: {e}")
            elif file_type == "txt":
                loader = DirectoryLoader(source_path, glob="**/*.txt", loader_cls=TextLoader)
                documents = loader.load()
        else:
            # Load single file
            try:
                if file_type == "pdf":
                    loader = PyPDFLoader(source_path)
                elif file_type == "txt":
                    loader = TextLoader(source_path)
                documents = loader.load()
                print(f"✓ Loaded {os.path.basename(source_path)}: {len(documents)} pages")
            except Exception as e:
                print(f"✗ Error loading file: {e}")
                return []
        
        # Print sample of loaded content
        if documents:
            print(f"\n📄 Total documents loaded: {len(documents)}")
            print(f"📝 Sample from first document (first 200 chars):")
            print(f"   {documents[0].page_content[:200]}...")
        else:
            print("⚠️  No documents were loaded!")
                
        return documents
    
    def split_documents(self, documents: List, chunk_size: int = 1000, 
                       chunk_overlap: int = 200) -> List:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"\n✂️  Split into {len(chunks)} chunks")
        print(f"📏 Chunk size: {chunk_size} chars, overlap: {chunk_overlap} chars")
        print(f"📝 Sample chunk (first 150 chars):")
        if chunks:
            print(f"   {chunks[0].page_content[:150]}...")
        return chunks
    
    def create_vectorstore(self, chunks: List):
        """
        Create and persist a vector store from document chunks.
        
        Args:
            chunks: List of document chunks
        """
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("Vector store created and persisted")
    
    def load_vectorstore(self):
        """Load an existing vector store from disk."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print("Vector store loaded from disk")
    
    def setup_qa_chain(self, chain_type: str = "stuff", k: int = 4, model_name: str = "google/flan-t5-base"):
        """
        Set up the question-answering chain.
        
        Args:
            chain_type: Type of chain (stuff, map_reduce, refine, map_rerank)
            k: Number of documents to retrieve
            model_name: HuggingFace model to use for local inference
                       Options: "google/flan-t5-base" (recommended), "google/flan-t5-small",
                               "microsoft/phi-2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Load or create one first.")
        
        # Initialize LLM if using local models
        if self.use_local_models and self.llm is None:
            print(f"Loading model: {model_name}...")
            
            # Determine the correct pipeline task and model class
            if "t5" in model_name.lower():
                # T5 models use text2text-generation
                from transformers import AutoModelForSeq2SeqLM
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                task = "text2text-generation"
            else:
                # Other models use text-generation
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                task = "text-generation"
            
            pipe = pipeline(
                task,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            print("Model loaded successfully")
        
        # Custom prompt template
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        
        Context: {context}
        
        Question: {question}
        
        Helpful Answer:"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        print("QA chain setup complete")
    
    def query(self, question: str, return_sources: bool = True):
        """
        Query the RAG system.
        
        Args:
            question: Question to ask
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optionally source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not set up. Call setup_qa_chain() first.")
        
        result = self.qa_chain({"query": question})
        
        response = {
            "answer": result["result"]
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        
        return response


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem()
    
    # Option 1: Load and index new documents
    documents = rag.load_documents("documents/Resume.pdf", file_type="pdf")
    chunks = rag.split_documents(documents)
    rag.create_vectorstore(chunks)
    
    # Option 2: Load existing vector store
    rag.load_vectorstore()
    
    # Set up QA chain
    rag.setup_qa_chain(k=4)
    
    # Query the system
    response = rag.query("What is the main topic of the documents?")
    print(f"Answer: {response['answer']}\n")
    print("Sources:")
    for i, source in enumerate(response['sources'], 1):
        print(f"\n{i}. {source['content'][:200]}...")
        print(f"   Metadata: {source['metadata']}")
