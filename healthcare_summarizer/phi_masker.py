import re
from datetime import datetime, timedelta
import random

class PHIMasker:
    """Class to handle masking of Protected Health Information (PHI)."""
    
    def __init__(self, use_consistent_hashing=True):
        """
        Initialize the PHI masker.
        
        Args:
            use_consistent_hashing: Not used in this version, kept for compatibility
        """
        self.use_consistent_hashing = use_consistent_hashing
        self.date_offset = None
    
    def mask_names(self, text):
        """Mask person names with <CONFIDENTIAL>."""
        # Pattern for common name formats with titles
        # Dr./Mr./Mrs./Ms. + First Last
        name_pattern = r'\b(Dr\.|Mr\.|Mrs\.|Ms\.)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
        text = re.sub(name_pattern, r'\1 <CONFIDENTIAL>', text)
        
        # Pattern for names without titles
        simple_name_pattern = r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
        matches = re.finditer(simple_name_pattern, text)

        replacements = []
        for match in matches:
            # Skip common medical terms that might match pattern
            full_match = match.group(0)
            if full_match not in ['Blood Pressure', 'Heart Rate', 'Body Mass']:
                replacements.append((match.span(), '<CONFIDENTIAL>'))
        
        # Apply replacements in reverse order to maintain positions
        for (start, end), replacement in reversed(replacements):
            text = text[:start] + replacement + text[end:]
        
        return text
    
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