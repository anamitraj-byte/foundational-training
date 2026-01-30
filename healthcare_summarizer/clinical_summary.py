import os
import json
from dotenv import load_dotenv
import docx
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import PyPDF2
from openai import OpenAI

from phi_masker import PHIMasker

load_dotenv()


def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = []
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text.append(page.extract_text())
    return '\n'.join(text)


def extract_transcript(file_path):
    """Extract transcript text based on file extension."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Please use .docx or .pdf files.")


def generate_clinical_summary(transcript_text, use_masked_data=False):
    """Generate clinical summary in JSON format."""
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )
    
    system_content = (
        "You are a medical documentation assistant specialized in creating structured clinical summaries. "
        "Extract and organize information from doctor-patient conversation transcripts.\n\n"
        
        "Return your output as a JSON object with the following structure:\n"
        "{\n"
        '  "chief_complaint": "string",\n'
        '  "diagnoses": ["item1", "item2"],\n'
        '  "medications": [{"name": "med_name", "dose": "dosage", "frequency": "freq"}],\n'
        '  "follow_up": ["action1", "action2"],\n'
        '  "monitoring": ["item1", "item2"],\n'
        '  "soap": {\n'
        '    "subjective": "text",\n'
        '    "objective": "text",\n'
        '    "assessment": "text",\n'
        '    "plan": "text"\n'
        '  }\n'
        "}\n\n"
        
        "CONTENT REQUIREMENTS:\n"
        "- Be concise, accurate, and use standard medical terminology\n"
        "- If information is not present, use 'Not documented'\n"
        "- If max token limit is reached, shorten content instead of cutting off mid-section\n"
        "- Return ONLY valid JSON, no markdown code blocks or additional text\n\n"
        
        "Note: This transcript contains masked PHI (Protected Health Information) for privacy protection."
    )
    
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b:groq",
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": f"Please create a structured clinical summary from the following doctor-patient transcript:\n\n{transcript_text}"
            },
        ],
        max_tokens=2000,
        temperature=0.3,
    )
    
    return completion.choices[0].message.content

def create_medical_summary_docx(json_data, output_path):
    """
    Convert any JSON structure to DOCX file dynamically.
    Automatically processes headings, paragraphs, lists, and nested structures.
    
    Args:
        json_data: Dictionary or JSON string with any structure
        output_path: Path where DOCX file should be saved
    
    Returns:
        str: Path to created DOCX file
    """
    # Parse JSON if it's a string
    if isinstance(json_data, str):
        json_str = json_data.strip()
        if json_str.startswith('```'):
            json_str = json_str.replace('```json', '').replace('```', '').strip()
        data = json.loads(json_str)
    else:
        data = json_data
    
    # Create document
    doc = Document()
    
    # Add title
    title = doc.add_heading('Clinical Summary', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    
    # Process the JSON structure recursively
    process_json_to_docx(doc, data, level=1)
    
    # Save document
    doc.save(output_path)
    print(f"Clinical summary DOCX saved to: {output_path}")
    return output_path


def process_json_to_docx(doc, data, level=1):
    """
    Recursively process JSON data and add to DOCX document.
    
    Args:
        doc: Document object
        data: Data to process (dict, list, or primitive)
        level: Current heading level (1 or 2)
    """
    if isinstance(data, dict):
        # Process dictionary - each key becomes a heading
        for key, value in data.items():
            # Convert key to readable heading (handle snake_case and underscores)
            heading_text = format_key_as_heading(key)
            
            # Add heading
            doc.add_heading(heading_text, level)
            
            # Process the value
            if isinstance(value, dict):
                # Nested dictionary - recurse with increased level
                process_json_to_docx(doc, value, level=min(level + 1, 2))
            elif isinstance(value, list):
                # List of items
                process_list_to_docx(doc, value)
            else:
                # Simple value - add as paragraph
                process_text_to_docx(doc, value)
            
            # Add spacing after section
            doc.add_paragraph()
            
    elif isinstance(data, list):
        # If data itself is a list (shouldn't happen at top level, but handle it)
        process_list_to_docx(doc, data)
    else:
        # Simple value
        process_text_to_docx(doc, data)


def format_key_as_heading(key):
    """
    Convert JSON key to readable heading.
    Examples:
        'chief_complaint' -> 'CHIEF COMPLAINT'
        'soap' -> 'SOAP'
        'follow_up' -> 'FOLLOW UP'
    """
    # Replace underscores with spaces
    heading = key.replace('_', ' ')
    # Convert to uppercase
    heading = heading.upper()
    return heading


def process_list_to_docx(doc, items):
    """
    Process a list of items and add to document.
    
    Args:
        doc: Document object
        items: List of items (can be strings, dicts, or mixed)
    """
    if not items:
        doc.add_paragraph('Not documented')
        return
    
    for item in items:
        if isinstance(item, dict):
            # Dictionary item - format as structured bullet
            formatted_text = format_dict_item(item)
            doc.add_paragraph(formatted_text, style='List Bullet')
        elif isinstance(item, str):
            # String item - add as bullet point, handling newlines
            lines = item.split('\n')
            for line in lines:
                if line.strip():
                    doc.add_paragraph(line.strip(), style='List Bullet')
        else:
            # Other types - convert to string
            doc.add_paragraph(str(item), style='List Bullet')


def format_dict_item(item_dict):
    """
    Format a dictionary item as a readable string.
    Example: {'name': 'Aspirin', 'dose': '100mg', 'frequency': 'daily'}
             -> 'Aspirin - 100mg - daily'
    """
    if not item_dict:
        return 'Not documented'
    
    # Join all values with ' - '
    values = [str(v) for v in item_dict.values() if v]
    return ' - '.join(values) if values else 'Not documented'


def process_text_to_docx(doc, text):
    """
    Process text content and add to document, handling newlines and formatting.
    
    Args:
        doc: Document object
        text: Text content (string or other primitive)
    """
    if text is None or text == '':
        doc.add_paragraph('Not documented')
        return
    
    # Convert to string if not already
    text_str = str(text)
    
    # Handle 'Not documented' case
    if text_str.lower() == 'not documented':
        doc.add_paragraph('Not documented')
        return
    
    # Split by newlines and add each line as a separate paragraph
    lines = text_str.split('\n')
    for line in lines:
        line = line.strip()
        if line:  # Only add non-empty lines
            doc.add_paragraph(line)


# Optional: Even more customizable version with configuration

def create_medical_summary_docx_configurable(json_data, output_path, config=None):
    """
    Convert JSON to DOCX with optional configuration for special handling.
    
    Args:
        json_data: Dictionary or JSON string
        output_path: Path where DOCX file should be saved
        config: Optional configuration dict for special handling
                Example: {
                    'title': 'Custom Title',
                    'exclude_keys': ['internal_id', 'metadata'],
                    'key_mappings': {'soap': 'SOAP Note'},
                    'list_keys': ['diagnoses', 'medications']  # Force these as lists
                }
    """
    if config is None:
        config = {}
    
    # Parse JSON if it's a string
    if isinstance(json_data, str):
        json_str = json_data.strip()
        if json_str.startswith('```'):
            json_str = json_str.replace('```json', '').replace('```', '').strip()
        data = json.loads(json_str)
    else:
        data = json_data
    
    # Create document
    doc = Document()
    
    # Add title (customizable)
    title_text = config.get('title', 'Clinical Summary')
    title = doc.add_heading(title_text, 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    
    # Process with configuration
    exclude_keys = config.get('exclude_keys', [])
    key_mappings = config.get('key_mappings', {})
    
    process_json_with_config(doc, data, config, level=1)
    
    # Save document
    doc.save(output_path)
    print(f"Clinical summary DOCX saved to: {output_path}")
    return output_path


def process_json_with_config(doc, data, config, level=1):
    """Process JSON with configuration options."""
    exclude_keys = config.get('exclude_keys', [])
    key_mappings = config.get('key_mappings', {})
    
    if isinstance(data, dict):
        for key, value in data.items():
            # Skip excluded keys
            if key in exclude_keys:
                continue
            
            # Use custom heading if mapped, otherwise auto-format
            heading_text = key_mappings.get(key, format_key_as_heading(key))
            
            doc.add_heading(heading_text, level)
            
            if isinstance(value, dict):
                process_json_with_config(doc, value, config, level=min(level + 1, 2))
            elif isinstance(value, list):
                process_list_to_docx(doc, value)
            else:
                process_text_to_docx(doc, value)
            
            doc.add_paragraph()

def save_summary_to_file(summary, output_path):
    """Save the clinical summary to a text file (for backup/debugging)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Clinical summary text saved to: {output_path}")


if __name__ == "__main__":
    # Configuration
    input_file = "transcript.pdf"  # or "transcript.docx"
    output_json_file = "clinical_summary.json"
    output_txt_file = "clinical_summary.txt"
    output_docx_file = "clinical_summary.docx"
    
    try:
        # Extract transcript from file
        print(f"Extracting transcript from {input_file}...")
        transcript = extract_transcript(input_file)
        print(f"Transcript extracted successfully. Length: {len(transcript)} characters\n")

        # Apply PHI masking
        print("Applying PHI masking...")
        masker = PHIMasker(use_consistent_hashing=True)
        masked_transcript = masker.apply_all_masking(transcript)
        print(f"Masking applied. Masked transcript length: {len(masked_transcript)} characters\n")
        
        # Optionally save masked transcript
        with open("masked_transcript.txt", 'w', encoding='utf-8') as f:
            f.write(masked_transcript)
        print("Masked transcript saved to: masked_transcript.txt\n")
        
        # Generate summary from masked data (returns JSON)
        print("Generating clinical summary from masked data...")
        summary_json = generate_clinical_summary(masked_transcript, use_masked_data=True)
        
        # Save raw JSON response for debugging
        save_summary_to_file(summary_json, output_txt_file)
        
        # Parse and save as formatted JSON
        try:
            if isinstance(summary_json, str):
                json_str = summary_json.strip()
                if json_str.startswith('```'):
                    json_str = json_str.replace('```json', '').replace('```', '').strip()
                parsed_json = json.loads(json_str)
            else:
                parsed_json = summary_json
                
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=2)
            print(f"Structured JSON saved to: {output_json_file}\n")
            
            # Print summary
            print("\n" + "="*80)
            print("CLINICAL SUMMARY (JSON FORMAT)")
            print("="*80 + "\n")
            print(json.dumps(parsed_json, indent=2))
            print("\n" + "="*80 + "\n")
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON response: {e}")
            print("Raw response saved to text file for inspection.")
            parsed_json = None
        
        # Create DOCX file
        if parsed_json:
            print("Creating DOCX file...")
            create_medical_summary_docx(parsed_json, output_docx_file)
            print(f"\n✓ DOCX file created successfully: {output_docx_file}")
        else:
            print("Skipping DOCX creation due to JSON parsing error.")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found. Please check the file path.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()