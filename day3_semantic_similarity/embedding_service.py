import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import warnings
warnings.filterwarnings('ignore')

class SemanticSimilarityChecker:
    def __init__(self, sentences=None, model_path=None, pretrained_name=None):
        """
        Initialize the semantic similarity checker.
        
        Args:
            sentences: List of sentences to train on (if training new model)
            model_path: Path to load custom pre-trained model (.model or .bin file)
            pretrained_name: Name of gensim pre-trained model to download
                           (e.g., 'word2vec-google-news-300', 'glove-wiki-gigaword-100')
        """
        self.model = None
        self.wv = None  # Word vectors
        
        if pretrained_name:
            print("I am pretraining")
            self.load_pretrained_gensim(pretrained_name)
        elif model_path:
            print("I am custom loading")
            self.load_custom_model(model_path)
        elif sentences:
            print("I am training")
            self.train_model(sentences)
        else:
            print("No model provided. Use pretrained_name, model_path, or sentences.")
    
    def load_pretrained_gensim(self, model_name):
        """
        Load a pre-trained model from gensim's collection.
        
        Popular models:
        - 'word2vec-google-news-300': Google News (100B words, 300-dim)
        - 'glove-wiki-gigaword-100': Wikipedia + Gigaword (6B tokens, 100-dim)
        - 'glove-wiki-gigaword-200': Wikipedia + Gigaword (6B tokens, 200-dim)
        - 'glove-twitter-25': Twitter (2B tweets, 25-dim)
        """
        print(f"Downloading pre-trained model: {model_name}")
        print("This may take a while for the first download...")
        self.wv = api.load(model_name)
        self.vector_size = self.wv.vector_size
        print(f"Model loaded! Vocabulary size: {len(self.wv)}")
        print(f"Vector size: {self.vector_size}")
    
    def load_custom_model(self, model_path):
        """
        Load a custom pre-trained model.
        
        Supports:
        - Gensim Word2Vec models (.model)
        - Google's original Word2Vec format (.bin)
        - KeyedVectors format
        """
        print(f"Loading model from: {model_path}")
        try:
            if model_path.endswith('.bin'):
                # Google's original Word2Vec binary format
                self.wv = KeyedVectors.load_word2vec_format(model_path, binary=True)
            elif model_path.endswith('.txt'):
                # Text format
                self.wv = KeyedVectors.load_word2vec_format(model_path, binary=False)
            else:
                # Gensim's native format
                self.model = Word2Vec.load(model_path)
                self.wv = self.model.wv
            
            self.vector_size = self.wv.vector_size
            print(f"Model loaded! Vocabulary size: {len(self.wv)}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    @staticmethod
    def list_available_models():
        """List all available pre-trained models from gensim."""
        print("\nAvailable pre-trained models from gensim:")
        print("=" * 70)
        info = api.info()
        for name, details in info['models'].items():
            if 'word2vec' in name or 'glove' in name or 'fasttext' in name:
                print(f"\nName: {name}")
                print(f"  Description: {details.get('description', 'N/A')[:100]}...")
                print(f"  File size: {details.get('file_size', 'N/A')}")
                print(f"  Parameters: {details.get('parameters', 'N/A')}")
    
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
        self.wv = self.model.wv
        self.vector_size = vector_size
        print(f"Model trained on {len(sentences)} sentences")
        print(f"Vocabulary size: {len(self.wv)}")
    
    def word_similarity(self, word1, word2):
        """
        Calculate similarity between two words.
        
        Returns:
            Similarity score (0-1), or None if word not in vocabulary
        """
        try:
            similarity = self.wv.similarity(word1.lower(), word2.lower())
            return similarity
        except KeyError as e:
            print(f"Word not in vocabulary: {e}")
            return None
    
    def get_word_vector(self, word):
        """Get the vector representation of a word."""
        try:
            return self.wv[word.lower()]
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
            if word in self.wv:
                word_vectors.append(self.wv[word])
        
        if not word_vectors:
            return np.zeros(self.vector_size)
        
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
            similar = self.wv.most_similar(word.lower(), topn=top_n)
            return similar
        except KeyError:
            print(f"Word '{word}' not in vocabulary")
            return []
    
    def analogy(self, positive, negative):
        """
        Solve word analogies (e.g., king - man + woman = queen).
        
        Args:
            positive: List of positive words
            negative: List of negative words
            
        Returns:
            List of (word, similarity_score) tuples
        """
        try:
            result = self.wv.most_similar(positive=positive, negative=negative, topn=5)
            return result
        except KeyError as e:
            print(f"Word not in vocabulary: {e}")
            return []
    
    def save_model(self, path):
        """Save the trained model to disk."""
        self.model.save(path)
        print(f"Model saved to {path}")


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("SEMANTIC SIMILARITY CHECKER")
    print("="*70)
    print("\nLoading pre-trained model (this may take a moment)...")
    
    # Load a lightweight model for quick testing
    # For better results, use: pretrained_name='word2vec-google-news-300'
    checker = SemanticSimilarityChecker(pretrained_name='glove-wiki-gigaword-100')
    
    print("\n" + "="*70)
    print("SENTENCE SIMILARITY CHECKER")
    print("="*70)
    print("\nThis tool compares two sentences and shows how similar they are.")
    print("Similarity score ranges from 0 (completely different) to 1 (identical)")
    print("\nType 'quit' or 'exit' to stop.\n")
    
    while True:
        print("-" * 70)
        
        # Get first sentence
        sentence1 = input("Enter first sentence: ").strip()
        if sentence1.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Semantic Similarity Checker!")
            break
        
        if not sentence1:
            print("Please enter a valid sentence.\n")
            continue
        
        # Get second sentence
        sentence2 = input("Enter second sentence: ").strip()
        if sentence2.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Semantic Similarity Checker!")
            break
        
        if not sentence2:
            print("Please enter a valid sentence.\n")
            continue
        
        # Calculate similarity
        print("\nCalculating similarity...")
        similarity_score = checker.sentence_similarity(sentence1, sentence2)
        
        # Display results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Sentence 1: {sentence1}")
        print(f"Sentence 2: {sentence2}")
        print(f"\nSimilarity Score: {similarity_score:.4f}")
        
        # Interpretation
        if similarity_score >= 0.8:
            interpretation = "Very similar - these sentences have nearly the same meaning"
        elif similarity_score >= 0.6:
            interpretation = "Similar - these sentences are related and share common themes"
        elif similarity_score >= 0.4:
            interpretation = "Somewhat similar - these sentences have some overlap in meaning"
        elif similarity_score >= 0.2:
            interpretation = "Slightly similar - these sentences have minimal overlap"
        else:
            interpretation = "Not similar - these sentences are about different topics"
        
        print(f"Interpretation: {interpretation}")
        
        # Show which words were found in vocabulary
        words1 = simple_preprocess(sentence1)
        words2 = simple_preprocess(sentence2)
        
        found_words1 = [w for w in words1 if w in checker.wv]
        found_words2 = [w for w in words2 if w in checker.wv]
        
        print(f"\nWords from sentence 1 in vocabulary: {len(found_words1)}/{len(words1)}")
        print(f"Words from sentence 2 in vocabulary: {len(found_words2)}/{len(words2)}")
        
        print("\n")