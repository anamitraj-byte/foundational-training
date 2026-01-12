import os
import numpy as np
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)


#Cosine similarity = (dot product of the vectors A and B) / (L2 norm(A) * L2 norm(B))
# L2 norm(A) = square root of sum of squares of all values in the vector
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Get user input
main_sentence = input("Enter the main sentence: ")
compare_sentences_count = int(input("Enter the number of sentences you wish to compare against the main sentence: "))
other_sentences = []
while compare_sentences_count > 0:
    sentence = input("Enter sentence: ")
    other_sentences.append(sentence)
    compare_sentences_count -= 1

# Get embeddings for all sentences
all_sentences = [main_sentence] + other_sentences
embeddings = client.feature_extraction(
    text=all_sentences,
    model=MODEL,
)

# Extract main sentence embedding and other embeddings
main_embedding = np.array(embeddings[0])
other_embeddings = [np.array(emb) for emb in embeddings[1:]]

# Calculate cosine similarities
print("\nSimilarity Scores:")
print("-" * 50)
for i, sentence in enumerate(other_sentences):
    similarity = cosine_similarity(main_embedding, other_embeddings[i])
    print(f"{sentence}: {similarity:.4f}")