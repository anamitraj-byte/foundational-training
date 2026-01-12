import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

main_sentence = input("Enter the main sentence:")
compare_sentences_count = int(input("Enter the number of sentences you wish to compare against the main sentence:"))
other_sentences = []
while compare_sentences_count > 0:
    sentence = input("Enter sentence:")
    other_sentences.append(sentence)
    compare_sentences_count -= 1

result = client.sentence_similarity(
    main_sentence,
    other_sentences = other_sentences,
    model=MODEL,
)

for i in range(len(other_sentences)):
    print(other_sentences[i], ":", result[i])