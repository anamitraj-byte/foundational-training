import os
import hashlib
import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

# Load environment variables
load_dotenv()


class RAGSystem:
    """
    RAG system with unified cosine similarity metric throughout.
    
    Key improvements:
    1. Uses cosine similarity for both chunk and sentence filtering
    2. Consistent threshold interpretation (higher = more similar)
    3. Better performance tracking and debugging
    4. Optimized batch embedding for performance
    """
    
    def __init__(self, persist_directory: str = "./chroma_db", 
                 metadata_file: str = "./indexed_files.json",
                 use_local_models: bool = True):
        """Initialize the unified RAG system."""
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
        self._sentence_cache = {}  # Cache for sentence embeddings
        
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
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Returns:
            Similarity score between -1 and 1 (higher = more similar)
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _smart_sentence_split(self, text: str) -> List[str]:
        """Smart sentence splitting."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
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
            separators=["\n\n", "\n", ".", " ", ""]
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
        Create vector store with explicit cosine similarity configuration.
        """
        if not chunks:
            print("No chunks to create vectorstore from")
            return
        
        # IMPORTANT: Configure Chroma to use cosine similarity
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}  # ← KEY: Use cosine distance
        )
        print("✓ Vector store created with COSINE similarity metric")
    
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
        """Load existing vector store."""
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
                      similarity_threshold: float = 0.3):
        """
        Set up QA chain with UNIFIED cosine similarity threshold.
        
        Args:
            k: Number of chunks to retrieve
            model_name: HuggingFace model for local inference
            response_style: Style of response
            similarity_threshold: Minimum cosine similarity (0-1, higher = stricter)
                                 0.0 = orthogonal (unrelated)
                                 0.3 = somewhat similar (default)
                                 0.5 = moderately similar
                                 0.7 = very similar
                                 1.0 = identical
        """
        if self.vectorstore is None:
            self.load_vectorstore()
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized. Index documents first.")
        
        # Store threshold (now in cosine similarity scale)
        self.similarity_threshold = similarity_threshold
        
        # Setup retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",  # Use similarity search
            search_kwargs={"k": k}
        )
        
        # Initialize LLM if needed
        if self.use_local_models and self.llm is None:
            print(f"🤖 Loading model: {model_name}...")
            
            if "t5" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                task = "text2text-generation"
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
        
        template = """You are a helpful AI assistant.

TASK:
Answer the user's question using only the information provided in the context.
Keep the response answer-like rather than it being copy-pasted context.

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
        
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        
        def format_docs(docs):
            if not docs:
                return "No relevant information found in the documents."
            
            formatted = []
            total_chars = 0
            max_chars = 3000

            for doc in docs:
                content = doc.page_content.strip()
                if total_chars + len(content) > max_chars:
                    remaining = max_chars - total_chars
                    if remaining > 100:
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
        
        print(f"✓ QA chain setup with '{response_style}' style")
        print(f"✓ Cosine similarity threshold: {similarity_threshold} (0=unrelated, 1=identical)")
    
    def query(self, question: str, return_sources: bool = True, 
              sentence_level_reranking: bool = True,
              verbose: bool = True,
              return_detailed_info: bool = False) -> Dict:
        """
        Query with unified cosine similarity filtering.
        
        Args:
            question: Question to ask
            return_sources: Whether to return source documents
            sentence_level_reranking: Whether to do sentence-level similarity filtering
            verbose: Whether to print detailed progress
            return_detailed_info: Whether to include context, prompt, and full pipeline details
            
        Returns:
            Dictionary with answer and metadata
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not set up. Call setup_qa_chain() first.")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 STAGE 1: Chunk-Level Retrieval (Cosine Similarity)")
            print(f"{'='*60}")
        
        # Get question embedding
        question_embedding = np.array(self.embeddings.embed_query(question))
        
        # Retrieve chunks with scores
        retrieved_docs_with_scores = self.vectorstore.similarity_search_with_score(
            question, 
            k=self.retriever.search_kwargs.get("k", 4)
        )
        
        if verbose:
            print(f"Retrieved {len(retrieved_docs_with_scores)} chunks")
        
        # UNIFIED FILTERING: Convert distances to cosine similarity
        filtered_chunks = []
        for doc, distance in retrieved_docs_with_scores:
            # Chroma's cosine distance: distance = 1 - cosine_similarity
            similarity = 1 - distance
            
            doc.metadata['chunk_similarity'] = round(similarity, 3)
            doc.metadata['chunk_distance'] = round(distance, 3)
            
            if similarity >= self.similarity_threshold:
                filtered_chunks.append(doc)
                if verbose:
                    print(f"  ✓ Chunk from {doc.metadata.get('source_file', 'Unknown')}")
                    print(f"    Similarity: {similarity:.3f} (distance: {distance:.3f})")
            else:
                if verbose:
                    print(f"  ✗ Filtered: {doc.metadata.get('source_file', 'Unknown')}")
                    print(f"    Similarity: {similarity:.3f} < threshold {self.similarity_threshold}")
        
        if verbose:
            print(f"\n📊 {len(filtered_chunks)} chunks pass threshold (>= {self.similarity_threshold})")
        
        if not filtered_chunks:
            return {
                "question": question,
                "answer": "I don't have enough relevant information to answer this question.",
                "sources": [],
                "num_sources": 0,
                "chunk_similarities": [],
                "sentence_similarities": [],
                "context": "",
                "prompt": ""
            }
        
        # STAGE 2: Sentence-level reranking (optional)
        if sentence_level_reranking:
            if verbose:
                print(f"\n{'='*60}")
                print(f"🔍 STAGE 2: Sentence-Level Reranking (Cosine Similarity)")
                print(f"{'='*60}")
            
            context, sentence_similarities = self._extract_relevant_sentences(
                filtered_chunks, 
                question_embedding,
                max_chars=2000
            )
        else:
            # Use full chunks
            context = "\n\n".join([doc.page_content for doc in filtered_chunks])[:2000]
            sentence_similarities = []
        
        # Generate prompt (for detailed info)
        full_prompt = self.prompt.format(context=context, question=question) if return_detailed_info else ""
        
        # Generate answer
        if verbose:
            print(f"\n{'='*60}")
            print(f"🤖 Generating Answer")
            print(f"{'='*60}")
            print(f"Context length: {len(context)} chars")
        
        answer = self.qa_chain.invoke(question)
        
        response = {
            "question": question,
            "answer": answer.strip() if isinstance(answer, str) else answer,
            "num_sources": len(filtered_chunks),
            "chunk_similarities": [doc.metadata['chunk_similarity'] for doc in filtered_chunks],
            "sentence_similarities": sentence_similarities[:10]  # Top 10
        }
        
        # Add detailed info if requested
        if return_detailed_info:
            response["context"] = context
            response["prompt"] = full_prompt
        
        if return_sources:
            response["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in filtered_chunks
            ]
        
        return response
    
    def _extract_relevant_sentences(self, chunks: List, 
                                    question_embedding: np.ndarray,
                                    max_chars: int = 2000,
                                    top_k: int = 15) -> Tuple[str, List[float]]:
        """
        Extract most relevant sentences using COSINE SIMILARITY with batch embeddings.
        
        Returns:
            Tuple of (context_string, list_of_similarities)
        """
        all_sentences = []
        
        # Collect sentences from all chunks
        for chunk in chunks:
            sentences = self._smart_sentence_split(chunk.page_content)
            for sent in sentences:
                if len(sent.strip()) > 20:
                    all_sentences.append({
                        'text': sent,
                        'source': chunk.metadata.get('source_file', 'Unknown')
                    })
        
        if not all_sentences:
            return "", []
        
        print(f"Found {len(all_sentences)} sentences to analyze")
        
        # OPTIMIZED: Batch embed all sentences at once
        sentence_texts = [s['text'] for s in all_sentences]
        sentence_embeddings = self.embeddings.embed_documents(sentence_texts)
        
        # OPTIMIZED: Vectorized similarity calculation
        embeddings_matrix = np.array(sentence_embeddings)
        norms = np.linalg.norm(embeddings_matrix, axis=1)
        question_norm = np.linalg.norm(question_embedding)
        
        # Calculate all similarities at once
        similarities = np.dot(embeddings_matrix, question_embedding) / (norms * question_norm)
        
        # Add similarities to sentence data
        for i, sim in enumerate(similarities):
            all_sentences[i]['similarity'] = float(sim)
        
        # Sort by similarity (highest first)
        all_sentences.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Build context
        selected = []
        total_chars = 0
        selected_similarities = []
        
        for item in all_sentences[:top_k]:
            sent = item['text']
            sim = item['similarity']
            
            if total_chars + len(sent) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    selected.append(sent[:remaining] + "...")
                    selected_similarities.append(sim)
                break
            
            selected.append(sent)
            selected_similarities.append(sim)
            total_chars += len(sent)
            
            print(f"  ✓ Selected (similarity: {sim:.3f}): {sent[:80]}...")
        
        print(f"\nSelected {len(selected)} sentences (total: {total_chars} chars)")
        if selected_similarities:
            print(f"Similarity range: {min(selected_similarities):.3f} - {max(selected_similarities):.3f}")
        
        return "\n\n".join(selected), selected_similarities
    
    def query_with_details(self, question: str):
        """
        Query with detailed information about the RAG process.
        This is now just a convenience wrapper that calls query() with verbose options.
        
        Returns detailed information about retrieval and generation WITHOUT making duplicate LLM calls.
        """
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print(f"{'='*60}")
        
        # Call query() once with detailed info enabled
        response = self.query(
            question=question,
            return_sources=True,
            sentence_level_reranking=True,
            verbose=True,
            return_detailed_info=True
        )
        
        # Pretty print the detailed breakdown
        print(f"\n{'='*60}")
        print(f"📋 DETAILED BREAKDOWN")
        print(f"{'='*60}")
        
        print(f"\n[Context Used ({len(response['context'])} chars)]")
        print(f"{response['context'][:500]}...")
        
        print(f"\n[Full Prompt]")
        print(f"{response['prompt'][:800]}...")
        
        print(f"\n{'='*60}")
        print(f"💡 Generated Answer:")
        print(f"{'='*60}")
        print(f"{response['answer']}")
        
        print(f"\n{'='*60}")
        print(f"📊 Similarity Scores:")
        print(f"{'='*60}")
        print(f"Chunk similarities: {response['chunk_similarities']}")
        print(f"Top sentence similarities: {response['sentence_similarities'][:5]}")
        print(f"{'='*60}\n")
        
        return response
    
    def get_similarity_stats(self, question: str, k: int = 10):
        """
        Get detailed similarity statistics for debugging.
        Shows unified cosine similarity metrics.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        # DIAGNOSTIC: Check what distance metric Chroma is actually using
        try:
            collection = self.vectorstore._collection
            metadata = collection.metadata
            actual_metric = metadata.get('hnsw:space', 'unknown')
            print(f"\n⚠️  DIAGNOSTIC: Chroma is using '{actual_metric}' distance metric")
            if actual_metric != 'cosine':
                print(f"❌ WARNING: Expected 'cosine' but got '{actual_metric}'!")
        except Exception as e:
            print(f"⚠️  Could not check distance metric: {e}")
        
        # Get top k chunks
        results = self.vectorstore.similarity_search_with_score(question, k=k)
        
        print(f"\n{'='*80}")
        print(f"📊 SIMILARITY STATISTICS (Cosine Similarity - Unified Metric)")
        print(f"{'='*80}")
        print(f"Question: {question}")
        print(f"Threshold: {self.similarity_threshold}\n")
        
        stats = []
        for i, (doc, distance) in enumerate(results, 1):
            similarity = 1 - distance  # Assumes cosine distance
            
            print(f"[{i}] {doc.metadata.get('source_file', 'Unknown')}")
            print(f"    Raw Distance:      {distance:.4f}")
            print(f"    Cosine Similarity: {similarity:.4f} {'✓ PASS' if similarity >= self.similarity_threshold else '✗ FAIL'}")            
            print(f"    Preview: {doc.page_content[:100]}...")
            print()
            
            stats.append({
                'rank': i,
                'source': doc.metadata.get('source_file', 'Unknown'),
                'raw_distance': distance,
                'similarity': similarity,
                'passes_threshold': similarity >= self.similarity_threshold
            })
        
        print(f"{'='*80}\n")
        return stats
    
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
    
    def clear_cache(self):
        """Clear sentence embedding cache."""
        self._sentence_cache.clear()
        print("✓ Sentence cache cleared")


# Example usage
if __name__ == "__main__":
    print("🚀 Initializing Unified RAG System (Cosine Similarity Throughout)\n")
    
    # Initialize
    rag = RAGSystem(use_local_models=True)

    if os.path.exists("./chroma_db"):
        rag.load_vectorstore()
        rag.index_documents("documents")
    else:
        print("No existing vectorstore found. Creating new one with cosine metric...")
        # Force reload to create vectorstore from scratch
        rag.index_documents("documents", force_reload=True)
    
    
    # Show indexed files
    rag.get_indexed_files_info()
    
    # Setup QA chain with unified threshold (now vectorstore is guaranteed to exist)
    rag.setup_qa_chain(
        k=4,
        model_name="google/flan-t5-large",
        response_style="conversational",
        similarity_threshold=0.3  # 0.3 = moderately similar (recommended default)
    )
    
    # Query the system
    print("\n" + "="*60)
    print("✅ Ready for queries!")
    print("="*60)
    
    while True:
        q = input("\n💬 Enter your question (or 'quit' to exit, 'stats' for diagnostics, 'detailed' for verbose mode): ")
        
        if q.lower() in ['q', 'quit', 'exit']:
            print("👋 Goodbye!")
            break
        
        elif q.lower() == 'stats':
            test_q = input("Enter a question to analyze: ")
            if test_q.strip():
                rag.get_similarity_stats(test_q, k=5)
            continue
        
        elif q.lower() == 'detailed':
            test_q = input("Enter a question for detailed analysis: ")
            if test_q.strip():
                # This now only makes ONE LLM call but shows all details
                rag.query_with_details(test_q)
            continue
        
        elif q.strip():
            # Single query with all info included (ONE LLM call)
            response = rag.query(
                q, 
                sentence_level_reranking=True, 
                verbose=True,
                return_detailed_info=True  # Get context/prompt without extra LLM call
            )
            
            print(f"\n{'='*60}")
            print(f"💡 Answer:")
            print(f"{'='*60}")
            print(f"{response['answer']}\n")
            
            print(f"📊 Metadata:")
            print(f"  Sources used: {response['num_sources']}")
            print(f"  Chunk similarities: {response['chunk_similarities']}")
            print(f"  Top sentence similarities: {response['sentence_similarities'][:5]}")
            
            # Show context/prompt if user wants (no extra LLM call needed!)
            show_details = input("\n🔍 Show context & prompt? (y/n): ")
            if show_details.lower() == 'y':
                print(f"\n📄 Context ({len(response['context'])} chars):")
                print(f"{response['context'][:800]}...")
                print(f"\n📝 Full Prompt:")
                print(f"{response['prompt'][:800]}...")