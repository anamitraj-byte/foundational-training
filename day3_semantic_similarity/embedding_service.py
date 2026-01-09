import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class SemanticSimilarityChecker:
    def __init__(self, sentences=None, model_path=None):
        """
        Initialize the semantic similarity checker.
        
        Args:
            sentences: List of sentences to train on (if training new model)
            model_path: Path to load pre-trained model
        """
        if model_path:
            self.model = Word2Vec.load(model_path)
        elif sentences:
            self.train_model(sentences)
        else:
            # Use a pre-trained model (requires downloading)
            print("No model provided. Train with sentences or load a pre-trained model.")
            self.model = None
    
    def train_model(self, sentences, vector_size=100, window=5, min_count=1, workers=4):
        """
        Train a Word2Vec model on provided sentences.
        
        Args:
            sentences: List of sentences (strings)
            vector_size: Dimensionality of word vectors
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
        """
        # Preprocess sentences into tokens
        processed_sentences = [simple_preprocess(sent) for sent in sentences]
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=processed_sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=10
        )
        print(f"Model trained on {len(sentences)} sentences")
        print(f"Vocabulary size: {len(self.model.wv)}")
    
    def word_similarity(self, word1, word2):
        """
        Calculate similarity between two words.
        
        Returns:
            Similarity score (0-1), or None if word not in vocabulary
        """
        try:
            similarity = self.model.wv.similarity(word1.lower(), word2.lower())
            return similarity
        except KeyError as e:
            print(f"Word not in vocabulary: {e}")
            return None
    
    def get_word_vector(self, word):
        """Get the vector representation of a word."""
        try:
            return self.model.wv[word.lower()]
        except KeyError:
            return None
    
    def sentence_vector(self, sentence):
        """
        Get vector representation of a sentence by averaging word vectors.
        
        Args:
            sentence: Input sentence (string)
            
        Returns:
            numpy array representing sentence vector
        """
        words = simple_preprocess(sentence)
        word_vectors = []
        
        for word in words:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])
        
        if not word_vectors:
            return np.zeros(self.model.vector_size)
        
        # Average all word vectors
        return np.mean(word_vectors, axis=0)
    
    def sentence_similarity(self, sentence1, sentence2):
        """
        Calculate similarity between two sentences.
        
        Returns:
            Similarity score (0-1)
        """
        vec1 = self.sentence_vector(sentence1).reshape(1, -1)
        vec2 = self.sentence_vector(sentence2).reshape(1, -1)
        
        similarity = cosine_similarity(vec1, vec2)[0][0]
        return similarity
    
    def most_similar_words(self, word, top_n=5):
        """
        Find most similar words to the given word.
        
        Args:
            word: Input word
            top_n: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        try:
            similar = self.model.wv.most_similar(word.lower(), topn=top_n)
            return similar
        except KeyError:
            print(f"Word '{word}' not in vocabulary")
            return []
    
    def save_model(self, path):
        """Save the trained model to disk."""
        self.model.save(path)
        print(f"Model saved to {path}")


# Example usage
if __name__ == "__main__":
    # Sample training data
    training_sentences = [
        "The cat sat on the mat",
        "The dog played in the park",
        "I love machine learning and artificial intelligence",
        "Python is a great programming language",
        "Natural language processing is fascinating",
        "The quick brown fox jumps over the lazy dog",
        "I enjoy coding in Python every day",
        "Machine learning models require lots of data",
        "The cat and dog are friends",
        "Artificial intelligence is changing the world",
        "Programming languages help us build software",
        "Natural language understanding is challenging",
        "The park is where children play",
        "Software development requires practice",
        "Data science combines statistics and programming"
    ]
    
    # Create and train the checker
    checker = SemanticSimilarityChecker(sentences=training_sentences)
    
    print("\n" + "="*60)
    print("WORD SIMILARITY EXAMPLES")
    print("="*60)
    
    # Test word similarity
    word_pairs = [
        ("cat", "dog"),
        ("python", "programming"),
        ("machine", "artificial"),
        ("learning", "intelligence")
    ]
    
    for word1, word2 in word_pairs:
        sim = checker.word_similarity(word1, word2)
        if sim is not None:
            print(f"Similarity between '{word1}' and '{word2}': {sim:.4f}")
    
    print("\n" + "="*60)
    print("SENTENCE SIMILARITY EXAMPLES")
    print("="*60)
    
    # Test sentence similarity
    sentence_pairs = [
        ("I love programming", "I enjoy coding"),
        ("The cat is sleeping", "The dog is playing"),
        ("Machine learning is cool", "Artificial intelligence is amazing")
    ]
    
    for sent1, sent2 in sentence_pairs:
        sim = checker.sentence_similarity(sent1, sent2)
        print(f"\nSentence 1: '{sent1}'")
        print(f"Sentence 2: '{sent2}'")
        print(f"Similarity: {sim:.4f}")
    
    print("\n" + "="*60)
    print("MOST SIMILAR WORDS")
    print("="*60)
    
    # Find similar words
    test_words = ["python", "cat", "learning"]
    for word in test_words:
        print(f"\nWords most similar to '{word}':")
        similar = checker.most_similar_words(word, top_n=3)
        for sim_word, score in similar:
            print(f"  {sim_word}: {score:.4f}")
