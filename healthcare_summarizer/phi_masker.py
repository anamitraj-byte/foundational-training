import re
from git import List
import hashlib
from datetime import datetime, timedelta
import random
import spacy

class PHIMasker:
    """Class to handle masking of Protected Health Information (PHI)."""
    
    def __init__(self, use_consistent_hashing=True):
        """
        Initialize the PHI masker.
        
        Args:
            use_consistent_hashing: If True, same names will always get same masked values
        """

        self.use_consistent_hashing = use_consistent_hashing
        self.name_mapping = {}
        self.date_offset = None
        
    def _get_consistent_replacement(self, original, category, replacement_list):
        """Generate consistent replacement for the same original value."""
        if not self.use_consistent_hashing:
            return random.choice(replacement_list)
        
        key = f"{category}:{original}"
        if key not in self.name_mapping:
            # Use hash to deterministically pick from replacement list
            hash_val = int(hashlib.md5(original.encode()).hexdigest(), 16)
            self.name_mapping[key] = replacement_list[hash_val % len(replacement_list)]
        return self.name_mapping[key]
    
    def mask_names(self, text):
        """Mask person names with generic replacements."""
        # Common name patterns
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 
                      'Robert', 'Lisa', 'James', 'Mary']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 
                     'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
        
        # Pattern for common name formats
        # Dr./Mr./Mrs./Ms. + First Last or just First Last
        name_pattern = r'\b(Dr\.|Mr\.|Mrs\.|Ms\.)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
        
        def replace_full_name(match):
            title = match.group(1)
            first = match.group(2)
            last = match.group(3)
            
            new_first = self._get_consistent_replacement(first, 'first', first_names)
            new_last = self._get_consistent_replacement(last, 'last', last_names)
            return f"{title} {new_first} {new_last}"
        
        text = re.sub(name_pattern, replace_full_name, text)
        
        # Pattern for names without titles
        simple_name_pattern = r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
        matches = re.finditer(simple_name_pattern, text)

        replacements = []
        for match in matches:
            # Skip common medical terms that might match pattern
            full_match = match.group(0)
            if full_match not in ['Blood Pressure', 'Heart Rate', 'Body Mass']:
                first = match.group(1)
                last = match.group(2)
                new_first = self._get_consistent_replacement(first, 'first', first_names)
                new_last = self._get_consistent_replacement(last, 'last', last_names)
                replacements.append((match.span(), f"{new_first} {new_last}"))
        
        # Apply replacements in reverse order to maintain positions
        for (start, end), replacement in reversed(replacements):
            text = text[:start] + replacement + text[end:]
        
        return text
    

    def mask_dates(self, text):
        """Shift dates by a random offset to maintain temporal relationships."""
        if self.date_offset is None:
            # Random offset between -365 and +365 days
            self.date_offset = timedelta(days=random.randint(-365, 365))
        
        # Pattern for various date formats
        date_patterns = [
            (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', '%m/%d/%Y'),  # MM/DD/YYYY
            (r'\b(\d{4})-(\d{2})-(\d{2})\b', '%Y-%m-%d'),      # YYYY-MM-DD
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b', '%B %d %Y'),
        ]
        
        def shift_date(match, date_format):
            try:
                date_str = match.group(0).replace(',', '')
                original_date = datetime.strptime(date_str, date_format)
                shifted_date = original_date + self.date_offset
                
                # Format back to original format
                if ',' in match.group(0):
                    return shifted_date.strftime(date_format.replace(' %Y', ', %Y'))
                return shifted_date.strftime(date_format)
            except:
                return match.group(0)  # Return original if parsing fails
        
        for pattern, date_format in date_patterns:
            text = re.sub(pattern, lambda m: shift_date(m, date_format), text)
        
        return text
    
    def mask_phone_numbers(self, text):
        """Mask phone numbers."""
        # Pattern for various phone number formats
        phone_pattern = r'\b(\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-.\s]??\d{4})\b'
        text = re.sub(phone_pattern, '555-XXX-XXXX', text)
        return text
    
    def mask_ssn(self, text):
        """Mask Social Security Numbers."""
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        text = re.sub(ssn_pattern, 'XXX-XX-XXXX', text)
        return text
    
    def mask_mrn(self, text):
        """Mask Medical Record Numbers (common patterns)."""
        # Pattern for MRN (assuming format like MRN: 123456 or MRN#123456)
        mrn_pattern = r'\b(MRN|Medical Record Number|Patient ID)[\s:#]*(\d+)\b'
        text = re.sub(mrn_pattern, r'\1: [MASKED]', text, flags=re.IGNORECASE)
        return text
    
    def mask_addresses(self, text):
        """Mask street addresses."""
        # Pattern for street addresses (basic)
        address_pattern = r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b'
        text = re.sub(address_pattern, '[ADDRESS MASKED]', text, flags=re.IGNORECASE)
        
        # ZIP codes
        zip_pattern = r'\b\d{5}(-\d{4})?\b'
        text = re.sub(zip_pattern, 'XXXXX', text)
        
        return text
    
    def mask_emails(self, text):
        """Mask email addresses."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '[EMAIL MASKED]', text)
        return text
    
    def mask_ages_over_89(self, text):
        """Mask ages over 89 (HIPAA requirement)."""
        age_pattern = r'\b(\d+)[\s-]*(year[\s-]*old|years old|y\.?o\.?|age)\b'
        
        def replace_age(match):
            age = int(match.group(1))
            if age > 89:
                return f"90+ {match.group(2)}"
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
