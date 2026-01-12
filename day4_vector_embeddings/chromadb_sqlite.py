import chromadb
from chromadb.utils import embedding_functions
import os
from pathlib import Path
from typing import List, Dict, Tuple

# --- Set up variables ---
CHROMA_DATA_PATH = "chromadb_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"
DOCUMENTS_PATH = "documents/"  # Folder containing your .txt files

# --- Chunking Configuration ---
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks to preserve context

# --- Chunking Strategies ---

def chunk_by_fixed_size(text: str, chunk_size: int = CHUNK_SIZE, 
                        overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():  # Only add non-empty chunks
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


# --- Load documents from .txt files ---

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


# --- Process documents with chunking ---

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


# --- Main execution ---

# Connect to ChromaDB
os.environ["ALLOW_RESET"] = "TRUE"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
client.reset()  # Clean up for testing

# Set up embedding function
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# Create collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"}
)

# Load documents from folder
print(f"Loading documents from {DOCUMENTS_PATH}...")
documents_data = load_documents_from_folder(DOCUMENTS_PATH)

if documents_data:
    print(f"Found {len(documents_data)} documents")
    
    # Process documents with chunking
    # Choose strategy: 'fixed_size', 'sentences', 'paragraphs', or 'recursive'
    chunks, metadatas, ids = process_documents_with_chunking(
        documents_data, 
        chunking_strategy='recursive'
    )
    
    print(f"Created {len(chunks)} chunks from documents")
    
    # Add chunks to collection
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    
    print("Documents successfully added to ChromaDB!")
    
    # --- Perform searches ---
    queries = []
    num_queries = int(input("Enter number of queries"))
    while num_queries > 0:
        query = input("Enter query:")
        queries.append(query)    
        num_queries -= 1
    query_results = collection.query(
        query_texts=queries,
        n_results=3  # Get top 3 results
    )
    
    # Print results
    for i, q in enumerate(queries):
        print(f'\n{"="*60}')
        print(f'Query: {q}')
        print(f'{"="*60}')
        for j, (doc, metadata) in enumerate(zip(query_results['documents'][i], 
                                                  query_results['metadatas'][i])):
            print(f'\n{j+1}. Source: {metadata["filename"]} (Chunk {metadata["chunk_index"] + 1}/{metadata["total_chunks"]})')
            print(f'   Text: {doc[:500]}...')  # Show first 200 characters
        print()
else:
    print("No documents to process. Please add .txt files to the documents folder.")
