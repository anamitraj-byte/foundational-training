import os
from dotenv import load_dotenv
import docx
import PyPDF2
from openai import OpenAI
from phi_masker import PHIMasker
import md_to_docx

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
        
        "Return your output as clean, well-formatted MARKDOWN text.\n\n"
        
        "STRUCTURE:\n"
        "# Clinical Summary\n\n"
        "## Chief Complaint\n"
        "[Brief description]\n\n"
        
        "## Key Problems/Diagnoses\n"
        "[Use bullet points for multiple items]\n\n"
        
        "## Current Medications\n"
        "[Use a Markdown table if multiple medications with details]\n"
        "[Use bullet points if simple list]\n\n"
        
        "## Follow-Up Actions\n"
        "[Use numbered list if sequential starting from 1, bullets if not]\n\n"
        
        "## Monitoring Needs\n"
        "[Bullet points or brief paragraph]\n\n"
        
        "## SOAP Note\n"
        "### Subjective\n"
        "[Patient's reported symptoms and history]\n\n"
        
        "### Objective\n"
        "[Physical findings, vital signs - consider table for vitals]\n\n"
        
        "### Assessment\n"
        "[Clinical impression, use bullet points]\n\n"
        
        "### Plan\n"
        "[Use bullet points for action items]\n\n"
        
        "MARKDOWN FORMATTING GUIDELINES:\n"
        "- Use # for title, ## for main sections, ### for subsections\n"
        "- Use Markdown tables for structured data (medications, vitals, lab results):\n"
        "  | Column1 | Column2 |\n"
        "  |---------|----------|\n"
        "  | Data1   | Data2   |\n"
        "- Use - for unordered bullet points\n"
        "- Use 1., 2., 3. for numbered lists (sequential steps). Start from 1 for every new subheading.\n"
        "- Use **bold** for important terms or findings\n"
        "- Use *italic* for patient quotes or emphasis\n"
        "- Use blank lines between sections\n"
        "- For multi-line text within sections, use line breaks naturally\n\n"
        
        "CONTENT REQUIREMENTS:\n"
        "- Be concise, accurate, and use standard medical terminology\n"
        "- If information is not present, write 'Not documented'\n"
        "- If max token limit is reached, make sure to complete the summary with all critical information. Do not stop abruptly.\n"
        "- Return ONLY Markdown text - no code fences (```), no preamble, no explanation\n"
        "- Start your response directly with: # Clinical Summary\n\n"
        
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

if __name__ == "__main__":
    # Configuration
    input_file = "transcript.pdf"
    output_md_file = "clinical_summary.md"
    output_docx_file = "clinical_summary.docx"
    
    try:
        # Extract and mask transcript
        print(f"Extracting transcript from {input_file}...")
        transcript = extract_transcript(input_file)
        
        print("Applying PHI masking...")
        masker = PHIMasker(use_consistent_hashing=True)
        masked_transcript = masker.apply_all_masking(transcript)
        
        # Generate Markdown summary
        print("Generating clinical summary in Markdown format...")
        markdown_summary = generate_clinical_summary(masked_transcript, use_masked_data=True)
        while len(markdown_summary) == 0:
            print("Received empty summary, retrying...")
            markdown_summary = generate_clinical_summary(masked_transcript, use_masked_data=True)
        
        # Clean up response (remove code fences if LLM added them)
        markdown_summary = markdown_summary.strip()
        if markdown_summary.startswith('```'):
            markdown_summary = markdown_summary.replace('```markdown', '').replace('```', '').strip()
        
        # Save Markdown
        with open(output_md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_summary)
        print(f"Markdown summary saved: {output_md_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("CLINICAL SUMMARY (MARKDOWN)")
        print("="*80 + "\n")
        print(markdown_summary)
        print("\n" + "="*80 + "\n")
        
        # Convert Markdown to DOCX
        print("Converting Markdown to DOCX...")
        md_to_docx.markdown_to_docx(markdown_summary, output_docx_file)
        print(f"\n✓ DOCX file created successfully: {output_docx_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()