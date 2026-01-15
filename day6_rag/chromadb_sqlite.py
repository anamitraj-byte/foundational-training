import chromadb
from chromadb.utils import embedding_functions
import os
from pathlib import Path
from typing import List, Dict, Tuple
import requests 
from dotenv import load_dotenv

load_dotenv()

# --- Previous setup code (keeping your existing setup) ---
CHROMA_DATA_PATH = "chromadb_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"
DOCUMENTS_PATH = "documents/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- RAG Configuration ---
MAX_CONTEXT_CHUNKS = 5  # Maximum number of chunks to include in context
MAX_CONTEXT_LENGTH = 4000  # Maximum characters for context

# --- Your existing chunking functions ---
def chunk_by_fixed_size(text: str, chunk_size: int = CHUNK_SIZE, 
                        overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    
    return chunks

def chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> List[str]:
    """Split text into chunks by sentence count."""
    # Simple sentence splitting (for better results, use nltk or spacy)
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = '. '.join(sentences[i:i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk + '.')
    
    return chunks


def chunk_by_paragraphs(text: str, paragraphs_per_chunk: int = 2) -> List[str]:
    """Split text into chunks by paragraph count."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    for i in range(0, len(paragraphs), paragraphs_per_chunk):
        chunk = '\n\n'.join(paragraphs[i:i + paragraphs_per_chunk])
        if chunk:
            chunks.append(chunk)
    
    return chunks


def chunk_recursive(text: str, max_chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Recursively split text by paragraphs, then sentences, then fixed size.
    This preserves natural boundaries when possible.
    """
    chunks = []
    
    # First try splitting by paragraphs
    paragraphs = chunk_by_paragraphs(text, paragraphs_per_chunk=1)
    
    for para in paragraphs:
        if len(para) <= max_chunk_size:
            chunks.append(para)
        else:
            # If paragraph is too long, split by sentences
            sentence_chunks = chunk_by_sentences(para, sentences_per_chunk=1)
            
            current_chunk = ""
            for sentence in sentence_chunks:
                if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # If single sentence is still too long, use fixed-size chunking
                    if len(sentence) > max_chunk_size:
                        chunks.extend(chunk_by_fixed_size(sentence, chunk_size=max_chunk_size, overlap=CHUNK_OVERLAP))
                        current_chunk = ""
                    else:
                        current_chunk = sentence + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
    
    return chunks



def load_documents_from_folder(folder_path: str) -> List[Dict]:
    """Load all .txt files from a folder."""
    documents_data = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Creating folder: {folder_path}")
        folder.mkdir(parents=True, exist_ok=True)
        print("Please add .txt files to this folder and run again.")
        return documents_data
    
    txt_files = list(folder.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {folder_path}")
        return documents_data
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents_data.append({
                    'filename': txt_file.name,
                    'content': content,
                    'path': str(txt_file)
                })
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
    
    return documents_data

def process_documents_with_chunking(documents_data: List[Dict], 
                                    chunking_strategy: str = 'recursive') -> Tuple[List[str], List[Dict], List[str]]:
    """
    Process documents and create chunks with metadata.
    
    Returns:
        chunks: List of text chunks
        metadatas: List of metadata dicts for each chunk
        ids: List of unique IDs for each chunk
    """
    chunks = []
    metadatas = []
    ids = []
    
    for doc in documents_data:
        # Choose chunking strategy
        if chunking_strategy == 'fixed_size':
            doc_chunks = chunk_by_fixed_size(doc['content'])
        elif chunking_strategy == 'sentences':
            doc_chunks = chunk_by_sentences(doc['content'])
        elif chunking_strategy == 'paragraphs':
            doc_chunks = chunk_by_paragraphs(doc['content'])
        else:  # recursive (default)
            doc_chunks = chunk_recursive(doc['content'])
        
        # Add chunks with metadata
        for i, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            metadatas.append({
                'filename': doc['filename'],
                'chunk_index': i,
                'total_chunks': len(doc_chunks),
                'source_path': doc['path']
            })
            # Create unique ID: filename_chunkIndex
            chunk_id = f"{doc['filename'].replace('.txt', '')}_{i}"
            ids.append(chunk_id)
    
    return chunks, metadatas, ids

# --- RAG Functions ---

def format_context_from_results(documents: List[str], metadatas: List[Dict], 
                                max_chunks: int = MAX_CONTEXT_CHUNKS,
                                max_length: int = MAX_CONTEXT_LENGTH) -> str:
    """
    Format ChromaDB search results into context for LLM.
    Includes source attribution and truncation.
    """
    context_parts = []
    total_length = 0
    
    for i, (doc, meta) in enumerate(zip(documents[:max_chunks], metadatas[:max_chunks])):
        # Create source attribution
        source_info = f"[Source: {meta['filename']}, Chunk {meta['chunk_index']+1}]"
        chunk_text = f"{source_info}\n{doc}\n"
        
        # Check if adding this chunk would exceed max length
        if total_length + len(chunk_text) > max_length:
            break
            
        context_parts.append(chunk_text)
        total_length += len(chunk_text)
    
    return "\n".join(context_parts)


