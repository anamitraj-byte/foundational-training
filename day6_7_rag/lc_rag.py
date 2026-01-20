import os
import hashlib
import json
from typing import List, Set
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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
    """Enhanced RAG system with multi-format support and incremental updates."""
    
    def __init__(self, persist_directory: str = "./chroma_db", 
                 metadata_file: str = "./indexed_files.json",
                 use_local_models: bool = True):
        """
        Initialize the RAG system.
        
        Args:
            persist_directory: Directory to store the vector database
            metadata_file: JSON file to track indexed files
            use_local_models: Whether to use local HuggingFace models (True) or OpenAI (False)
        """
        self.persist_directory = persist_directory
        self.metadata_file = metadata_file
        self.use_local_models = use_local_models
        
        if use_local_models:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.llm = None
        else:
            from langchain_openai import OpenAIEmbeddings, ChatOpenAI
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            
        self.vectorstore = None
        self.qa_chain = None
        self.indexed_files = self._load_indexed_files()
        
    def _load_indexed_files(self) -> dict:
        """Load the record of previously indexed files."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_indexed_files(self):
        """Save the record of indexed files."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.indexed_files, f, indent=2)
    
    def _get_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of a file to detect changes."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_all_files(self, source_path: str) -> List[tuple]:
        """
        Get all PDF and TXT files from the source path.
        
        Returns:
            List of tuples (filepath, file_type)
        """
        files = []
        
        if os.path.isdir(source_path):
            for filename in os.listdir(source_path):
                filepath = os.path.join(source_path, filename)
                if filename.endswith('.pdf'):
                    files.append((filepath, 'pdf'))
                elif filename.endswith('.txt'):
                    files.append((filepath, 'txt'))
        else:
            # Single file
            if source_path.endswith('.pdf'):
                files.append((source_path, 'pdf'))
            elif source_path.endswith('.txt'):
                files.append((source_path, 'txt'))
        
        return files
    
    def _get_new_or_modified_files(self, source_path: str) -> List[tuple]:
        """
        Identify new or modified files that need to be indexed.
        
        Returns:
            List of tuples (filepath, file_type) for files that need indexing
        """
        all_files = self._get_all_files(source_path)
        files_to_index = []
        
        for filepath, file_type in all_files:
            file_hash = self._get_file_hash(filepath)
            
            # Check if file is new or modified
            if filepath not in self.indexed_files or self.indexed_files[filepath] != file_hash:
                files_to_index.append((filepath, file_type))
                print(f"📄 New/Modified: {os.path.basename(filepath)}")
        
        return files_to_index
    
    def load_documents(self, source_path: str, force_reload: bool = False) -> List:
        """
        Load documents from files (PDF and TXT).
        
        Args:
            source_path: Path to file or directory
            force_reload: If True, reload all files regardless of changes
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        # Get files to index
        if force_reload:
            files_to_load = self._get_all_files(source_path)
            print("Force reload: loading all files")
        else:
            files_to_load = self._get_new_or_modified_files(source_path)
            if not files_to_load:
                print("✓ All files are up to date. No new documents to load.")
                return []
        
        # Load each file
        for filepath, file_type in files_to_load:
            try:
                if file_type == "pdf":
                    loader = PyPDFLoader(filepath)
                elif file_type == "txt":
                    loader = TextLoader(filepath, encoding='utf-8')
                
                docs = loader.load()
                
                # Add file information to metadata
                for doc in docs:
                    doc.metadata['source_file'] = os.path.basename(filepath)
                    doc.metadata['file_type'] = file_type
                
                documents.extend(docs)
                
                # Update indexed files record
                self.indexed_files[filepath] = self._get_file_hash(filepath)
                
                print(f"✓ Loaded {os.path.basename(filepath)}: {len(docs)} pages/sections")
                
            except Exception as e:
                print(f"✗ Error loading {os.path.basename(filepath)}: {e}")
        
        # Save updated index
        if documents:
            self._save_indexed_files()
            print(f"\n📚 Total new/modified documents loaded: {len(documents)}")
            print(f"📝 Sample from first document (first 200 chars):")
            print(f"   {documents[0].page_content[:200]}...")
        
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
        if not documents:
            print("No documents to split")
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"\n Split into {len(chunks)} chunks")
        print(f"Chunk size: {chunk_size} chars, overlap: {chunk_overlap} chars")
        if chunks:
            print(f"Sample chunk (first 150 chars):")
            print(f"   {chunks[0].page_content[:150]}...")
        return chunks
    
    def create_vectorstore(self, chunks: List):
        """
        Create and persist a vector store from document chunks.
        
        Args:
            chunks: List of document chunks
        """
        if not chunks:
            print("No chunks to create vectorstore from")
            return
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("✓ Vector store created and persisted")
    
    def update_vectorstore(self, chunks: List):
        """
        Add new chunks to an existing vector store.
        
        Args:
            chunks: List of new document chunks to add
        """
        if not chunks:
            print("No new chunks to add")
            return
        
        if self.vectorstore is None:
            print("No existing vectorstore found. Creating new one...")
            self.create_vectorstore(chunks)
        else:
            # Add documents to existing vectorstore
            self.vectorstore.add_documents(chunks)
            print(f"✓ Added {len(chunks)} new chunks to vector store")
    
    def load_vectorstore(self):
        """Load an existing vector store from disk."""
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("✓ Vector store loaded from disk")
        else:
            print("No existing vector store found")
    
    def index_documents(self, source_path: str, force_reload: bool = False,
                       chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Complete indexing pipeline: load, split, and index documents.
        
        Args:
            source_path: Path to documents directory or file
            force_reload: If True, reindex all files
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        print("=" * 60)
        print("Starting document indexing...")
        print("=" * 60)
        
        # Load documents
        documents = self.load_documents(source_path, force_reload)
        
        if not documents:
            print("\n✓ Index is up to date")
            return
        
        # Split into chunks
        chunks = self.split_documents(documents, chunk_size, chunk_overlap)
        
        if not chunks:
            return
        
        # Create or update vectorstore
        if self.vectorstore is None:
            self.load_vectorstore()
        
        if self.vectorstore is None:
            self.create_vectorstore(chunks)
        else:
            self.update_vectorstore(chunks)
        
        print("\n" + "=" * 60)
        print("✓ Indexing complete!")
        print("=" * 60)
    
    def setup_qa_chain(self, chain_type: str = "stuff", k: int = 4, 
                      model_name: str = "google/flan-t5-base"):
        """
        Set up the question-answering chain.
        
        Args:
            chain_type: Type of chain (stuff, map_reduce, refine, map_rerank)
            k: Number of documents to retrieve
            model_name: HuggingFace model to use for local inference
        """
        if self.vectorstore is None:
            self.load_vectorstore()
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized. Index documents first.")
        
        # Initialize LLM if using local models
        if self.use_local_models and self.llm is None:
            print(f"Loading model: {model_name}...")
            
            if "t5" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                task = "text2text-generation"
            else:
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
            print("✓ Model loaded successfully")
        
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
        print("✓ QA chain setup complete")
    
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
    
    def get_indexed_files_info(self):
        """Get information about currently indexed files."""
        print("\n📊 Indexed Files Summary:")
        print("=" * 60)
        if not self.indexed_files:
            print("No files indexed yet.")
        else:
            for filepath in self.indexed_files:
                file_type = "PDF" if filepath.endswith('.pdf') else "TXT"
                print(f"  • {os.path.basename(filepath)} ({file_type})")
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    print("🚀 Initializing Enhanced RAG System\n")
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Index documents (will only index new/modified files)
    rag.index_documents("documents")
    
    # Show indexed files
    rag.get_indexed_files_info()
    
    # Set up QA chain
    rag.setup_qa_chain(k=4)
    
    # Query the system
    while True:
        q = input("Enter query, q to quit")
        if q.lower == 'q':
            exit()
        else:
            print("\n💬 Querying the system...\n")
            response = rag.query("What is the main topic of the documents?")
            print(f"❓ Question: What is the main topic of the documents?")
            print(f"💡 Answer: {response['answer']}\n")
            
            print("📚 Sources:")
            for i, source in enumerate(response['sources'], 1):
                print(f"\n{i}. {source['content'][:200]}...")
                print(f"   📁 File: {source['metadata'].get('source_file', 'Unknown')}")
                print(f"   📄 Type: {source['metadata'].get('file_type', 'Unknown')}")
    
    # To force reindex all files (useful if you suspect issues):
    # rag.index_documents("documents", force_reload=True)