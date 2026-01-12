import chromadb
from chromadb.utils import embedding_functions

# --- Set up variables ---
CHROMA_DATA_PATH = "chromadb_data/"  # Path where ChromaDB will store data
EMBED_MODEL = "all-MiniLM-L6-v2"  # Name of the pre-trained embedding model 
COLLECTION_NAME = "demo_docs"  # Name for our document collection

# --- Connect to ChromaDB ---
import os
os.environ["ALLOW_RESET"] = "TRUE"  # Enable resetting the ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)  # Create a ChromaDB client
# Clean up the DB, only for testing to avoid warnings when reinserting docs.
client.reset()

# --- Set up embedding function ---
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)  # Use a Sentence Transformer model for generating embeddings

# --- Create (or retrieve) the collection ---
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,  # Assign the embedding function to the collection
    metadata={"hnsw:space": "cosine"}  # Configure search optimization metadata 
)

# --- Prepare documents for storage ---
documents = [
    "The latest iPhone model comes with impressive features and a powerful camera.",
    "Exploring the beautiful beaches and vibrant culture of Bali is a dream for many travelers.",
    "Einstein's theory of relativity revolutionized our understanding of space and time.",
    "Traditional Italian pizza is famous for its thin crust, fresh ingredients, and wood-fired ovens.",
    "The American Revolution had a profound impact on the birth of the United States as a nation.",
    "Regular exercise and a balanced diet are essential for maintaining good physical health.",
    "Leonardo da Vinci's Mona Lisa is considered one of the most iconic paintings in art history.",
    "Climate change poses a significant threat to the planet's ecosystems and biodiversity.",
    "Startup companies often face challenges in securing funding and scaling their operations.",
    "Beethoven's Symphony No. 9 is celebrated for its powerful choral finale, 'Ode to Joy.'",
]

genres = [
    "technology",
    "travel",
    "science",
    "food",
    "history",
    "fitness",
    "art",
    "climate change",
    "business",
    "music",
]

# --- Add documents to the ChromaDB collection ---
collection.add(
    documents=documents,
    ids=[f"id{i}" for i in range(len(documents))],  # Generate unique document IDs
    metadatas=[{"genre": g} for g in genres]  # Associate genre metadata with each document
)

# --- Perform a search using a query  ---
q1 = "Find me some delicious food!"
q2 = "I am looking to buy a new Phone."
queries = [q1, q2]
query_results = collection.query(
    query_texts=queries,
    n_results=2,  # Retrieve the top 2 results
)

# --- Print the results  ---
for i, q in enumerate(queries):
  print(f'Query: {q}')
  print(f'Results:')
  for j, doc in enumerate(query_results['documents'][i]):
    print(f'{j+1}. {doc}')
  print('\n')