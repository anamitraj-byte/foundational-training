import re
from typing import List
import numpy as np

class ImprovedContextExtractor:
    """Enhanced context extraction with better relevance scoring."""
    
    def __init__(self, embeddings_model=None):
        """
        Args:
            embeddings_model: Optional embeddings model for semantic similarity
        """
        self.embeddings_model = embeddings_model
        
    def smart_sentence_split(self, text: str) -> List[str]:
        """
        Improved sentence splitting that handles edge cases.
        """
        # Replace common abbreviations temporarily
        text = re.sub(r'\bDr\.', 'Dr<DOT>', text)
        text = re.sub(r'\bMr\.', 'Mr<DOT>', text)
        text = re.sub(r'\bMrs\.', 'Mrs<DOT>', text)
        text = re.sub(r'\bMs\.', 'Ms<DOT>', text)
        text = re.sub(r'\be\.g\.', 'e<DOT>g<DOT>', text)
        text = re.sub(r'\bi\.e\.', 'i<DOT>e<DOT>', text)
        text = re.sub(r'\bvs\.', 'vs<DOT>', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_question_entities(self, question: str) -> dict:
        """
        Extract key entities and terms from the question.
        Returns keywords with weights.
        """
        # Remove common stop words
        stop_words = {
            'what', 'when', 'where', 'who', 'how', 'why', 'is', 'are', 
            'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in',
            'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about'
        }
        
        words = question.lower().split()
        
        # Weight different types of words
        keywords = {}
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^\w]', '', word)
            if word_clean and word_clean not in stop_words and len(word_clean) > 2:
                # Give higher weight to:
                # 1. Capitalized words (proper nouns)
                # 2. Words at the end of question (often the focus)
                # 3. Longer words (more specific)
                weight = 1.0
                
                if word[0].isupper():  # Proper noun
                    weight *= 1.5
                
                position_weight = 1.0 + (i / len(words)) * 0.5  # Later words weighted more
                weight *= position_weight
                
                length_weight = min(len(word_clean) / 10, 1.5)  # Longer = more specific
                weight *= length_weight
                
                keywords[word_clean] = weight
        
        return keywords
    
    def score_sentence_relevance(self, sentence: str, question_keywords: dict) -> float:
        """
        Score a sentence's relevance to the question.
        Uses multiple factors for better ranking.
        """
        sentence_lower = sentence.lower()
        score = 0.0
        
        # Factor 1: Keyword presence with weights
        for keyword, weight in question_keywords.items():
            if keyword in sentence_lower:
                # Count occurrences (but with diminishing returns)
                count = sentence_lower.count(keyword)
                score += weight * (1 + np.log1p(count - 1) * 0.5)
        
        # Factor 2: Keyword density (what % of sentence is relevant)
        sentence_words = set(re.findall(r'\w+', sentence_lower))
        keyword_overlap = len(sentence_words & set(question_keywords.keys()))
        if len(sentence_words) > 0:
            density = keyword_overlap / len(sentence_words)
            score += density * 2.0
        
        # Factor 3: Penalize very short or very long sentences
        sentence_length = len(sentence.split())
        if sentence_length < 5:
            score *= 0.5  # Too short, likely incomplete
        elif sentence_length > 100:
            score *= 0.7  # Too long, likely not focused
        
        # Factor 4: Bonus for sentences with multiple keywords close together
        keyword_positions = []
        for keyword in question_keywords.keys():
            pos = sentence_lower.find(keyword)
            if pos != -1:
                keyword_positions.append(pos)
        
        if len(keyword_positions) >= 2:
            # Check proximity of keywords
            keyword_positions.sort()
            avg_distance = np.mean(np.diff(keyword_positions))
            if avg_distance < 50:  # Keywords within 50 chars
                score += 1.0
        
        return score
    
    def extract_relevant_contexts(
        self, 
        filtered_docs: List, 
        question: str,
        max_total_chars: int = 2000,
        sentences_per_doc: int = 5,
        min_relevance_score: float = 0.1
    ) -> str:
        """
        Extract most relevant contexts from documents using improved strategy.
        
        Args:
            filtered_docs: List of documents that passed distance threshold
            question: The user's question
            max_total_chars: Maximum total characters for context
            sentences_per_doc: Max sentences to consider per document
            min_relevance_score: Minimum relevance score to include sentence
            
        Returns:
            Formatted context string
        """
        # Extract question keywords with weights
        question_keywords = self.extract_question_entities(question)
        
        if not question_keywords:
            # Fallback: use simple keyword extraction
            question_keywords = {
                word.lower(): 1.0 
                for word in question.split() 
                if len(word) > 3
            }
        
        print(f"🔑 Key terms: {list(question_keywords.keys())}")
        
        # Collect all relevant sentences from all docs
        all_scored_sentences = []
        
        for doc in filtered_docs:
            # Smart sentence splitting
            sentences = self.smart_sentence_split(doc.page_content)
            
            # Score each sentence
            for sentence in sentences:
                if len(sentence.strip()) > 20:  # Skip very short fragments
                    score = self.score_sentence_relevance(sentence, question_keywords)
                    
                    if score >= min_relevance_score:
                        all_scored_sentences.append({
                            'sentence': sentence,
                            'score': score,
                            'doc_metadata': doc.metadata,
                            'doc_similarity': doc.metadata.get('similarity', 0.5)
                        })
        
        # Sort by relevance score (highest first)
        all_scored_sentences.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"📊 Found {len(all_scored_sentences)} relevant sentences")
        
        if not all_scored_sentences:
            # Fallback to document beginnings
            print("⚠️ No sentences scored high enough, using document beginnings")
            return "\n\n".join([doc.page_content[:300] for doc in filtered_docs[:2]])
        
        # Build context by selecting top sentences up to max_total_chars
        selected_contexts = []
        total_chars = 0
        docs_used = set()
        
        for item in all_scored_sentences:
            sentence = item['sentence']
            doc_id = item['doc_metadata'].get('source_file', 'unknown')
            
            # Diversity: try not to take too many from the same doc
            if doc_id in docs_used and len(docs_used) < len(filtered_docs):
                # Already used this doc, prefer others if available
                continue
            
            if total_chars + len(sentence) > max_total_chars:
                # Check if we can fit a truncated version
                remaining = max_total_chars - total_chars
                if remaining > 100:
                    selected_contexts.append(sentence[:remaining] + "...")
                break
            
            selected_contexts.append(sentence)
            total_chars += len(sentence)
            docs_used.add(doc_id)
            
            if len(selected_contexts) >= 10:  # Max 10 sentences total
                break
        
        print(f"✓ Selected {len(selected_contexts)} most relevant sentences ({total_chars} chars)")
        
        # Format contexts with some structure
        context = "\n\n".join(selected_contexts)
        return context
    
    def extract_relevant_contexts_with_embeddings(
        self,
        filtered_docs: List,
        question: str,
        max_total_chars: int = 2000
    ) -> str:
        """
        Use semantic embeddings for even better context extraction.
        Requires embeddings_model to be set.
        """
        if self.embeddings_model is None:
            raise ValueError("Embeddings model not provided")
        
        # Get question embedding
        question_embedding = self.embeddings_model.embed_query(question)
        
        # Collect sentences with their embeddings
        all_sentences_with_embeddings = []
        
        for doc in filtered_docs:
            sentences = self.smart_sentence_split(doc.page_content)
            
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    # Get sentence embedding
                    sentence_embedding = self.embeddings_model.embed_query(sentence)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(question_embedding, sentence_embedding) / (
                        np.linalg.norm(question_embedding) * np.linalg.norm(sentence_embedding)
                    )
                    
                    all_sentences_with_embeddings.append({
                        'sentence': sentence,
                        'similarity': similarity,
                        'doc_metadata': doc.metadata
                    })
        
        # Sort by similarity
        all_sentences_with_embeddings.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Build context
        selected_contexts = []
        total_chars = 0
        
        for item in all_sentences_with_embeddings[:15]:  # Top 15 most similar
            sentence = item['sentence']
            
            if total_chars + len(sentence) > max_total_chars:
                remaining = max_total_chars - total_chars
                if remaining > 100:
                    selected_contexts.append(sentence[:remaining] + "...")
                break
            
            selected_contexts.append(sentence)
            total_chars += len(sentence)
        
        print(f"✓ Selected {len(selected_contexts)} semantically similar sentences")
        
        return "\n\n".join(selected_contexts)