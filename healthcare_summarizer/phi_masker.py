import re
from datetime import datetime, timedelta
import random
from typing import Set, List, Tuple

class PHIMasker:
    """Class to handle masking of Protected Health Information (PHI)."""
    
    def __init__(self, use_consistent_hashing=True, name_threshold=0.3):
        """
        Initialize the PHI masker.
        
        Args:
            use_consistent_hashing: Not used in this version, kept for compatibility
            name_threshold: Minimum score to consider something a name (0.0-1.0)
                          Lower = more aggressive masking, Higher = more conservative
        """
        self.use_consistent_hashing = use_consistent_hashing
        self.date_offset = None
        self.name_threshold = name_threshold
        self.exclusions = self._load_exclusions()
        self.medical_context_words = {
            'patient', 'doctor', 'nurse', 'appointment', 'prescription',
            'diagnosis', 'treatment', 'medical', 'hospital', 'clinic',
            'therapy', 'medication', 'symptoms', 'condition', 'procedure'
        }
    
    def _load_exclusions(self) -> Set[str]:
        """Load common English words that aren't names."""
        return {
            # Common words
            'Perfect', 'Good', 'Great', 'Bad', 'Nice', 'Fine', 'Okay',
            'Yes', 'No', 'Maybe', 'Sure', 'Hello', 'Hi', 'Hey', 'Bye',
            'Thank', 'Thanks', 'Please', 'Sorry', 'Welcome', 'Excuse',
            # Medical terms
            'Blood', 'Heart', 'Body', 'White', 'Red', 'Sugar', 'Test',
            'Pressure', 'Rate', 'Mass', 'Room', 'Care', 'Emergency',
            'Medical', 'Record', 'History', 'Insurance', 'Operating',
            'Intensive', 'Patient', 'Doctor', 'Nurse', 'Hospital',
            # Months
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December',
            # Days
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
            # Common phrases
            'During', 'After', 'Before', 'While', 'Because', 'However',
            'Therefore', 'Meanwhile', 'Otherwise', 'Although', 'Unless'
        }
    
    def _is_likely_name(self, word: str, context: str) -> float:
        """
        Score a word's likelihood of being a name (0.0 to 1.0).
        Higher score = more likely to be a name.
        """
        score = 0.0
        word_lower = word.lower()
        
        # Rule out common words
        if word in self.exclusions:
            return 0.0
        
        # Must be capitalized and alphabetic
        if not (word[0].isupper() and word.isalpha()):
            return 0.0
        
        # Length heuristics (names are typically 2-15 characters)
        if len(word) < 2 or len(word) > 15:
            return 0.0
        
        # Check context indicators
        # Strong indicators (vocative, possessive, introductions)
        if re.search(rf'(good\s+job|thank\s+you|thanks|hello|hi|hey|bye),\s*{re.escape(word)}\b', 
                     context, re.IGNORECASE):
            score += 0.5
        
        if re.search(rf'\b{re.escape(word)},\s+(can|could|would|please|I|you|let|may)\b', 
                     context, re.IGNORECASE):
            score += 0.4
        
        if re.search(rf'\b(patient|named|called|I\'m|my name is|this is)\s+{re.escape(word)}\b', 
                     context, re.IGNORECASE):
            score += 0.6
        
        if re.search(rf'\b{re.escape(word)}\'s\b', context):
            score += 0.3
        
        # Title prefix
        if re.search(rf'\b(Dr\.|Mr\.|Mrs\.|Ms\.)\s+{re.escape(word)}\b', context):
            score += 0.6
        
        # Followed by another capitalized word (likely last name)
        if re.search(rf'\b{re.escape(word)}\s+([A-Z][a-z]+)\b', context):
            next_word_match = re.search(rf'\b{re.escape(word)}\s+([A-Z][a-z]+)\b', context)
            if next_word_match and next_word_match.group(1) not in self.exclusions:
                score += 0.4
        
        # Sentence beginning (weaker signal)
        if re.search(rf'(^|[.!?]\s+){re.escape(word)}\b', context):
            score += 0.1
        
        # Medical context reduces likelihood (might be a medical term)
        context_lower = context.lower()
        if any(med_word in context_lower for med_word in self.medical_context_words):
            # But only penalize if the word itself looks medical
            if word_lower.endswith(('osis', 'itis', 'ology', 'ectomy', 'gram', 'pathy')):
                score -= 0.3
        
        # Unusual character patterns common in non-English names
        has_double_consonants = bool(re.search(r'([bcdfghjklmnpqrstvwxyz])\1', word_lower))
        has_unusual_start = word_lower.startswith(('zh', 'kh', 'ng', 'mb', 'nd', 'nj'))
        
        if has_double_consonants or has_unusual_start:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _group_consecutive_names(self, replacements: List[Tuple[int, int, str, float]], 
                                 text: str) -> List[Tuple[int, int, str, float]]:
        """Group consecutive capitalized words that are likely part of the same name."""
        if not replacements:
            return []
        
        grouped = []
        i = 0
        
        while i < len(replacements):
            start, end, word, score = replacements[i]
            
            # Look ahead to see if next word is close and also a name
            if i + 1 < len(replacements):
                next_start, next_end, next_word, next_score = replacements[i + 1]
                
                # If words are separated by just a space (full name like "John Smith")
                between = text[end:next_start]
                if between.strip() == '' and len(between) <= 2:
                    # Merge them
                    end = next_end
                    word = word + ' ' + next_word
                    score = max(score, next_score)
                    i += 1  # Skip the next one
            
            grouped.append((start, end, word, score))
            i += 1
        
        return grouped
    
    def mask_names(self, text: str) -> str:
        """
        Mask person names with <CONFIDENTIAL> using probabilistic scoring.
        Works with names from any language/culture.
        """
        # Find all capitalized words
        word_pattern = r'\b([A-Z][a-z]+)\b'
        matches = list(re.finditer(word_pattern, text))
        
        replacements = []
        
        for match in matches:
            word = match.group(1)
            start, end = match.span()
            
            # Get surrounding context (50 chars on each side)
            context_start = max(0, start - 50)
            context_end = min(len(text), end + 50)
            context = text[context_start:context_end]
            
            # Score this word
            likelihood = self._is_likely_name(word, context)
            
            if likelihood >= self.name_threshold:
                replacements.append((start, end, word, likelihood))
        
        # Handle multi-word names (e.g., "John Smith")
        grouped_replacements = self._group_consecutive_names(replacements, text)
        
        # Apply replacements in reverse order
        for start, end, _, _ in reversed(grouped_replacements):
            text = text[:start] + '<CONFIDENTIAL>' + text[end:]
        
        return text
    
    def mask_names_with_feedback(self, text: str) -> Tuple[str, List[dict]]:
        """
        Mask names and return what was masked for review/debugging.
        
        Returns:
            Tuple of (masked_text, list of candidate words with scores)
        """
        word_pattern = r'\b([A-Z][a-z]+)\b'
        matches = list(re.finditer(word_pattern, text))
        
        candidates = []
        replacements = []
        
        for match in matches:
            word = match.group(1)
            start, end = match.span()
            
            context_start = max(0, start - 50)
            context_end = min(len(text), end + 50)
            context = text[context_start:context_end]
            
            likelihood = self._is_likely_name(word, context)
            
            candidates.append({
                'word': word,
                'score': likelihood,
                'masked': likelihood >= self.name_threshold,
                'context': context
            })
            
            if likelihood >= self.name_threshold:
                replacements.append((start, end, word, likelihood))
        
        grouped_replacements = self._group_consecutive_names(replacements, text)
        
        for start, end, _, _ in reversed(grouped_replacements):
            text = text[:start] + '<CONFIDENTIAL>' + text[end:]
        
        return text, candidates
    
    def mask_dates(self, text):
        """Mask dates with <CONFIDENTIAL>."""
        # Pattern for various date formats
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',      # YYYY-MM-DD
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        ]
        
        for pattern in date_patterns:
            text = re.sub(pattern, '<CONFIDENTIAL>', text)
        
        return text
    
    def mask_phone_numbers(self, text):
        """Mask phone numbers."""
        # Pattern for various phone number formats
        phone_pattern = r'\b(\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-.\s]??\d{4})\b'
        text = re.sub(phone_pattern, '<CONFIDENTIAL>', text)
        return text
    
    def mask_ssn(self, text):
        """Mask Social Security Numbers."""
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        text = re.sub(ssn_pattern, '<CONFIDENTIAL>', text)
        return text
    
    def mask_mrn(self, text):
        """Mask Medical Record Numbers (common patterns)."""
        # Pattern for MRN (assuming format like MRN: 123456 or MRN#123456)
        mrn_pattern = r'\b(MRN|Medical Record Number|Patient ID)[\s:#]*(\d+)\b'
        text = re.sub(mrn_pattern, r'\1: <CONFIDENTIAL>', text, flags=re.IGNORECASE)
        return text
    
    def mask_addresses(self, text):
        """Mask street addresses."""
        # Pattern for street addresses (basic)
        address_pattern = r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b'
        text = re.sub(address_pattern, '<CONFIDENTIAL>', text, flags=re.IGNORECASE)
        
        # ZIP codes
        zip_pattern = r'\b\d{5}(-\d{4})?\b'
        text = re.sub(zip_pattern, '<CONFIDENTIAL>', text)
        
        return text
    
    def mask_emails(self, text):
        """Mask email addresses."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '<CONFIDENTIAL>', text)
        return text
    
    def mask_ages_over_89(self, text):
        """Mask ages over 89 (HIPAA requirement)."""
        age_pattern = r'\b(\d+)[\s-]*(year[\s-]*old|years old|y\.?o\.?|age)\b'
        
        def replace_age(match):
            age = int(match.group(1))
            if age > 89:
                return f"<CONFIDENTIAL> {match.group(2)}"
            return match.group(0)
        
        text = re.sub(age_pattern, replace_age, text, flags=re.IGNORECASE)
        return text
    
    def apply_all_masking(self, text):
        """Apply all masking techniques in sequence."""
        text = self.mask_names(text)
        text = self.mask_dates(text)
        text = self.mask_phone_numbers(text)
        text = self.mask_ssn(text)
        text = self.mask_mrn(text)
        text = self.mask_addresses(text)
        text = self.mask_emails(text)
        text = self.mask_ages_over_89(text)
        return text
