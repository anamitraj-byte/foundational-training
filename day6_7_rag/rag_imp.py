import chromadb
from chromadb.utils import embedding_functions
import os
from pathlib import Path
from typing import Dict
import requests 
from dotenv import load_dotenv
import chromadb_sqlite as cdb

load_dotenv()

def create_rag_prompt(query: str, context: str, 
                     system_instruction: str = None) -> str:
    """
    Create a prompt for RAG with context and query.
    """
    if system_instruction is None:
        system_instruction = """You are a helpful assistant that answers questions based on the provided context.
Always cite which source you're using when you answer.
If the context doesn't contain enough information to answer the question, say so."""
    
    prompt = f"""{system_instruction}

Context from documents:
{context}

Question: {query}

Answer:"""
    
    return prompt


def query_groq(prompt: str, api_key: str, model: str = "llama-3.3-70b-versatile",
               max_tokens: int = 500) -> str:
    """
    Call Groq API with the RAG prompt.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",  # Fixed endpoint
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    


def query_anthropic(prompt: str, api_key: str, model: str = "claude-3-sonnet-20240229",
                   max_tokens: int = 1000) -> str:
    """
    Call Anthropic API with the RAG prompt.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()["content"][0]["text"]
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")


def rag_query(collection, query: str, llm_api_key: str = None, 
             llm_provider: str = "openai", n_results: int = 5) -> Dict:
    """
    Complete RAG pipeline: retrieve relevant chunks and generate answer.
    
    Args:
        collection: ChromaDB collection
        query: User's question
        llm_api_key: API key for LLM provider
        llm_provider: "openai" or "anthropic"
        n_results: Number of chunks to retrieve
    
    Returns:
        Dictionary with answer, sources, and context used
    """
    # Step 1: Retrieve relevant documents
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    # Step 2: Format context
    context = cdb.format_context_from_results(
        results['documents'][0],
        results['metadatas'][0]
    )
    
    # Step 3: Create prompt
    prompt = create_rag_prompt(query, context)
    
    # Step 4: Get LLM response
    if llm_api_key:
        if llm_provider == "openai":
            model = os.environ["GROQ_MODEL"]
            answer = query_groq(prompt, llm_api_key, model)
        elif llm_provider == "anthropic":
            answer = query_anthropic(prompt, llm_api_key)
        else:
            answer = "Unknown LLM provider"
    else:
        # Return the prompt if no API key (for testing)
        answer = "[No API key provided - showing prompt only]"
    
    return {
        'query': query,
        'answer': answer,
        'context': context,
        'sources': [meta['filename'] for meta in results['metadatas'][0]],
        'prompt': prompt
    }


def get_indexed_files(collection) -> set:
    """Get set of filenames that are already in the collection."""
    try:
        all_metadatas = collection.get()['metadatas']
        return set(meta['filename'] for meta in all_metadatas)
    except:
        return set()

# --- Main execution ---
if __name__ == "__main__":
    os.environ["ALLOW_RESET"] = "TRUE"
    client = chromadb.PersistentClient(path=cdb.CHROMA_DATA_PATH)
    
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cdb.EMBED_MODEL
    )
    
    collection = client.get_or_create_collection(
        name=cdb.COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Always check for documents (not just when collection is empty)
    print(f"Loading documents from {cdb.DOCUMENTS_PATH}...")
    documents_data = cdb.load_documents_from_folder(cdb.DOCUMENTS_PATH)
    
    if not documents_data:
        print("No documents found. Add .txt or .pdf files to documents/ folder")
        exit()
    
    # Check which files are already indexed
    indexed_files = get_indexed_files(collection)
    new_documents = [doc for doc in documents_data if doc['filename'] not in indexed_files]
    
    if new_documents:
        print(f"\nFound {len(new_documents)} new document(s) to index:")
        for doc in new_documents:
            print(f"  - {doc['filename']}")
        
        chunks, metadatas, ids = cdb.process_documents_with_chunking(
            new_documents, 
            chunking_strategy='recursive'
        )
        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        print(f"✓ Added {len(chunks)} chunks from {len(new_documents)} new document(s)")
    else:
        print(f"✓ All {len(documents_data)} document(s) already indexed")
    
    print(f"\nTotal chunks in collection: {collection.count()}")
    print(f"Indexed files: {', '.join(sorted(get_indexed_files(collection)))}")
    
    # --- Interactive RAG Query Loop ---
    print("\n" + "="*60)
    print("RAG System Ready!")
    print("="*60)

    API_KEY = os.environ["GROQ_API_KEY"]

    # Interactive mode
    print("\n" + "="*60)
    print("Enter your questions (or 'quit' to exit)")
    print("="*60)
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        result = rag_query(collection, query, llm_api_key=API_KEY, llm_provider="openai")
        
        print(f"\n📚 Sources: {', '.join(set(result['sources']))}")
        print(f"\n📄 Context used ({len(result['context'])} chars):")
        print(result['context'][:300] + "..." if len(result['context']) > 300 else result['context'])
        print(f"\n💬 Answer:")
        print(result['answer'])