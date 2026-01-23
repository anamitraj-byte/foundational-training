import os
import hashlib
import json
from typing import List, Set
from pathlib import Path
from dotenv import load_dotenv
from imp_context import ImprovedContextExtractor

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

# Load environment variables
load_dotenv()

class RAGSystem:
    """Enhanced RAG system with explicit LLM response generation."""
    
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
        self.retriever = None
        self.indexed_files = self._load_indexed_files()
        self.context_extractor = ImprovedContextExtractor(self.embeddings)
        
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
        print(f"\n✂️ Split into {len(chunks)} chunks")
        print(f"📏 Chunk size: {chunk_size} chars, overlap: {chunk_overlap} chars")
        if chunks:
            print(f"📝 Sample chunk (first 150 chars):")
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
        print("📚 Starting document indexing...")
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
        print("✅ Indexing complete!")
        print("=" * 60)
    
    def setup_qa_chain(self, chain_type: str = "stuff", k: int = 4, 
                      model_name: str = "google/flan-t5-large",
                      response_style: str = "conversational",
                      relevance_threshold: float = 0.9):
        """
        Set up the question-answering chain with explicit LLM call.
        
        Args:
            chain_type: Type of chain (stuff, map_reduce, refine, map_rerank)
            k: Number of documents to retrieve
            model_name: HuggingFace model to use for local inference
            response_style: Style of response - 'conversational', 'professional', 'concise', 'detailed'
            relevance_threshold: Maximum distance threshold (lower = stricter). Default 0.7
                                 Distance metrics: 0 = identical, higher = less similar
        """
        if self.vectorstore is None:
            self.load_vectorstore()
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized. Index documents first.")
        
        # Store relevance threshold (for distance-based filtering)
        self.relevance_threshold = relevance_threshold
        
        # Setup retriever with MMR (Maximal Marginal Relevance) for diversity
        # We'll filter by distance in the query method
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Provides diversity in results
            search_kwargs={
                "k": k,
                "fetch_k": k * 3  # Fetch more candidates to filter from
            }
        )
        
        # Initialize LLM if using local models
        if self.use_local_models and self.llm is None:
            print(f"🤖 Loading model: {model_name}...")
            
            if "t5" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                task = "text2text-generation"
                # Increase max_length for T5 models to handle longer contexts
                max_length = 2048
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                task = "text-generation"
                max_length = 2048
            
            pipe = pipeline(
                task,
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                max_new_tokens=512,
                temperature=1,
                do_sample=True,
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            print("✓ Model loaded successfully")
        
        prompts = {
            "generic": """You are a helpful AI assistant.

        TASK:
        Answer the user's question using only the information provided in the context.

        CONSTRAINTS:
        - Use only relevant parts of the context
        - Do not copy sentences verbatim
        - Answer in your own words
        - Do not invent information
        - If multiple items are involved, present them as a list

        CONTEXT:
        {context}

        QUESTION:
        {question}

        FINAL ANSWER:"""
}

        # Select the appropriate template
        template = prompts.get(response_style, prompts["generic"])
        
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        
        # Create RAG chain with explicit LLM call and better context formatting
        def format_docs(docs):
            if not docs:
                return "No relevant information found in the documents."
            
            # Concatenate docs more efficiently and truncate if needed
            formatted = []
            total_chars = 0
            max_chars = 3000

            for i, doc in enumerate(docs):
                content = doc.page_content.strip()
                if total_chars + len(content) > max_chars:
                    # Truncate the last document to fit
                    remaining = max_chars - total_chars
                    if remaining > 100:  # Only add if meaningful content can fit
                        formatted.append(content[:remaining] + "...")
                    break
                formatted.append(content)
                total_chars += len(content)
            
            return "\n\n".join(formatted)
        
        self.qa_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        print(f"✓ QA chain setup complete with '{response_style}' style")
        print(f"✓ Relevance threshold set to {relevance_threshold} (lower = more strict)")
    
    def _execute_rag_pipeline(self, question: str, verbose: bool = False):
        """
        Core RAG pipeline execution - retrieves, filters, and generates answer.
        Used by both query() and query_with_details().
        
        Args:
            question: Question to ask
            verbose: Whether to print detailed progress information
            
        Returns:
            Dictionary with all RAG pipeline results including:
            - filtered_docs: Documents that passed relevance threshold
            - context: Extracted context text
            - full_prompt: Complete prompt sent to LLM
            - answer: Generated answer from LLM
            - all_retrieved: All initially retrieved documents with scores
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not set up. Call setup_qa_chain() first.")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"❓ Question: {question}")
            print(f"{'='*60}")
            print("\n[Step 1] 🔍 Retrieving relevant documents...")
        
        # Step 1: Retrieve documents with similarity scores
        retrieved_docs_with_scores = self.vectorstore.similarity_search_with_score(
            question, 
            k=self.retriever.search_kwargs.get("k", 4)
        )
        
        if verbose:
            print(f"✓ Retrieved {len(retrieved_docs_with_scores)} documents")
        else:
            print(f"\n🔍 Found {len(retrieved_docs_with_scores)} documents")
        
        # Step 2: Filter by relevance threshold
        filtered_docs = []
        for doc, distance in retrieved_docs_with_scores:
            if distance <= self.relevance_threshold:
                doc.metadata['distance'] = round(distance, 3)
                doc.metadata['similarity'] = round(1 - min(distance, 1), 3)
                filtered_docs.append(doc)
                
                if verbose:
                    print(f"  ✓ Kept: {doc.metadata.get('source_file', 'Unknown')} (distance: {distance:.3f})")
                else:
                    print(f"  ✓ Doc: {doc.metadata.get('source_file', 'Unknown')} (distance: {distance:.3f}, similarity: {doc.metadata['similarity']:.3f})")
            else:
                if verbose:
                    print(f"  ✗ Filtered: {doc.metadata.get('source_file', 'Unknown')} (distance: {distance:.3f})")
                else:
                    print(f"  ✗ Filtered out: {doc.metadata.get('source_file', 'Unknown')} (distance: {distance:.3f} > threshold {self.relevance_threshold})")
        
        print(f"\n📊 {len(filtered_docs)} documents pass relevance threshold ({self.relevance_threshold})")
        
        # Check if we have any relevant documents
        if not filtered_docs:
            print("⚠️  No documents meet the relevance threshold")
            return {
                "filtered_docs": [],
                "context": "",
                "full_prompt": "",
                "answer": "I don't have enough relevant information in the documents to answer this question. The available documents don't seem to be related to your query.",
                "all_retrieved": retrieved_docs_with_scores
            }
        
        # Step 3: Extract relevant context
        if verbose:
            print("\n[Step 2] 📋 Extracting relevant context")
        else:
            print("📝 Generating LLM response...")
        
        context = self.context_extractor.extract_relevant_contexts_with_embeddings(
            filtered_docs=filtered_docs,
            question=question,
            max_total_chars=1000
        )
        
        # Limit total context length
        if len(context) > 1000:
            context = context[:1000] + "..."
        
        if verbose:
            print(f"✓ Context prepared ({len(context)} characters)")
            print(f"\n📄 Extracted context:\n{context[:300]}...\n")
        else:
            print(f"📏 Context size: {len(context)} characters")
        
        # Step 4: Generate prompt
        if verbose:
            print("\n[Step 3] 📝 Creating prompt...")
        
        full_prompt = self.prompt.format(context=context, question=question)
        
        if verbose:
            print(f"✓ Prompt created ({len(full_prompt)} characters)")
            print(f"\n📄 Full prompt:\n{full_prompt}\n")
        
        # Step 5: Call LLM
        if verbose:
            print("[Step 4] 🤖 Calling LLM to generate answer...")
        
        answer = self.llm.invoke(full_prompt)
        
        # Clean up answer if needed
        if isinstance(answer, str):
            answer = answer.strip()
        
        if verbose:
            print("✓ LLM response received")
            print(f"\n{'='*60}")
            print(f"💡 Generated Answer:\n{answer}")
            print(f"{'='*60}\n")
        else:
            print("✅ LLM response generated")
        
        return {
            "filtered_docs": filtered_docs,
            "context": context,
            "full_prompt": full_prompt,
            "answer": answer,
            "all_retrieved": retrieved_docs_with_scores
        }

    def query(self, question: str, return_sources: bool = True):
        """
        Query the RAG system with explicit LLM generation and relevance filtering.
        
        Args:
            question: Question to ask
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with LLM-generated answer and optionally source documents
        """
        # Execute core RAG pipeline
        rag_results = self._execute_rag_pipeline(question, verbose=False)
        
        # Build response
        response = {
            "answer": rag_results["answer"],
            "raw_answer": rag_results["answer"]
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in rag_results["filtered_docs"]
            ]
            response["num_sources"] = len(rag_results["filtered_docs"])
        
        return response

    def query_with_details(self, question: str):
        """
        Query with detailed information about the RAG process.
        Uses the SAME smart filtering as the regular query method.
        
        Returns detailed information about retrieval and generation.
        """
        if self.retriever is None:
            raise ValueError("Retriever not set up. Call setup_qa_chain() first.")
        
        # Execute core RAG pipeline with verbose output
        rag_results = self._execute_rag_pipeline(question, verbose=True)
        
        return {
            "question": question,
            "answer": rag_results["answer"],
            "context": rag_results["context"],
            "prompt": rag_results["full_prompt"],
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in rag_results["filtered_docs"]
            ]
        }
    
    def get_indexed_files_info(self):
        """Get information about currently indexed files."""
        print("\n📊 Indexed Files Summary:")
        print("=" * 60)
        if not self.indexed_files:
            print("❌ No files indexed yet.")
        else:
            for filepath in self.indexed_files:
                file_type = "PDF" if filepath.endswith('.pdf') else "TXT"
                print(f"  ✓ {os.path.basename(filepath)} ({file_type})")
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    print("🚀 Initializing Enhanced RAG System with Explicit LLM Generation\n")
    
    # Initialize RAG system
    rag = RAGSystem(use_local_models=True)
    
    # Index documents (will only index new/modified files)
    rag.index_documents("documents")
    
    # Show indexed files
    rag.get_indexed_files_info()

    rag.setup_qa_chain(
        k=4, 
        model_name="google/flan-t5-large", 
        response_style="detailed",
        relevance_threshold=0.9
    )
    
    # Query the system
    print("\n" + "="*60)
    print("✅ Ready for queries!")
    print("="*60)
    
    while True:
        q = input("\n💬 Enter your question (or 'quit' to exit): ")
        if q.lower() in ['q', 'quit', 'exit']:
            print("👋 Goodbye!")
            break
        
        if q.strip():
            # Simple query
            response = rag.query(q)
            print(f"\n💡 Answer: {response['answer']}\n")
            
            # Optionally show detailed process
            show_details = input("🔍 Show detailed RAG process? (y/n): ")
            if show_details.lower() == 'y':
                rag.query_with_details(q)