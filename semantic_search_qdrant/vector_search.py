from qdrant_client import QdrantClient, models

client = QdrantClient(":memory:")  # Qdrant is running from RAM.

docs = [
    "Qdrant has a LangChain integration for chatbots.",
    "Qdrant has a LlamaIndex integration for agents.",
]
metadata = [
    {"source": "langchain-docs"},
    {"source": "llamaindex-docs"},
]
ids = [42, 2]

model_name = "BAAI/bge-small-en"
client.create_collection(
    collection_name="test_collection",
    vectors_config=models.VectorParams(
        size=client.get_embedding_size(model_name), 
        distance=models.Distance.COSINE
    ),  # size and distance are model dependent
)

metadata_with_docs = [
    {"document": doc, "source": meta["source"]} for doc, meta in zip(docs, metadata)
]
client.upload_collection(
    collection_name="test_collection",
    vectors=[models.Document(text=doc, model=model_name) for doc in docs],
    payload=metadata_with_docs,
    ids=ids,
)


search_result = client.query_points(
    collection_name="test_collection",
    query=models.Document(
        text="Which integration is best for agents?", 
        model=model_name
    )
).points
print(search_result)