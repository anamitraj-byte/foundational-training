import os
from dotenv import load_dotenv
import docx
import PyPDF2
from openai import OpenAI
from phi_masker import PHIMasker
import md_to_docx
from groq import Groq
from google import genai
from google.genai import types

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
    # client = OpenAI(
    #     base_url="https://router.huggingface.co/v1",
    #     api_key=os.environ["HF_TOKEN"],
    # )
    # client = Groq()
    
    system_content = (
    "You are a medical documentation assistant specialized in creating structured clinical summaries from doctor-patient conversation transcripts.\n"
    "This transcript contains masked PHI (Protected Health Information) for privacy protection. Conversations may be multilingual.\n\n"

    "Your responsibility is to extract and organize ONLY what is explicitly stated in the transcript. Do not infer, assume, extrapolate, or add context that is not clearly documented.\n\n"

    "## WORD COUNT REQUIREMENTS (EXTREMELY IMPORTANT)\n"
    "Each summary section below has a specific word range that must be respected.\n\n"

    "While writing each section, consciously track its length and refine wording so the final content naturally fits within the required range.\n"
    "If a section feels long, reduce redundancy and tighten language rather than adding more detail.\n\n"

    "Brief Summary: 150–300 words\n"
    "Goldilocks Summary: 200–350 words\n"
    "Detailed Summary: 300–400 words\n"
    "Super Detailed Summary: 400–500 words\n"
    "SOAP Note (entire SOAP section combined): 200–300 words\n\n"

    "Ensure every section is complete, clinically accurate, and comfortably within its range. Do not pad content to reach minimums and do not exceed maximums.\n\n"

    "## CORE PRINCIPLES\n"
    "Extract only documented information; never infer or extrapolate\n"
    "If information is not present, write \"Not documented\"\n"
    "Preserve clinical negations exactly (e.g., \"no fever\" must not become \"fever\")\n"
    "Resolve conflicting statements using the most recent clinician assessment\n"
    "Use standard medical terminology and abbreviations appropriately\n"
    "Exclude herbal teas and general supplements from Current Medications\n"
    "Omit all patient and clinician identifying information\n"
    "Output clean Markdown only — no code fences, no explanations, no preamble\n\n"

    "Begin the response exactly with:\n"
    "# Clinical Summary\n\n"

    "## OUTPUT STRUCTURE\n\n"

    "# Clinical Summary\n\n"

    "## Quick Reference\n\n"

    "### Chief Complaint\n"
    "[Concise primary presenting concern]\n\n"

    "### Key Problems/Diagnoses\n"
    "[Bulleted list of active clinical issues]\n\n"

    "### Current Medications\n"
    "[Markdown table for multiple medications with dosing details; bulleted list for simple entries]\n\n"

    "### Follow-Up Actions\n"
    "[Numbered list for sequential tasks; bullets for non-sequential items]\n\n"

    "### Monitoring Needs\n"
    "[Parameters requiring ongoing surveillance]\n\n"

    "---\n\n"

    "## Goldilocks Summary\n"
    "[200–350 words | Balanced clinical detail for standard documentation, must stick to the word count]\n\n"

    "**Assessment:**\n"
    "Patient's personal history\n"
    "Anthropometric data\n"
    "Dietary history\n"
    "Physical activity patterns\n\n"

    "**Diagnosis:**\n"
    "[Working diagnoses and differential considerations]\n\n"

    "**Intervention:**\n"
    "Food and/or nutrient delivery\n"
    "Education on specific nutrition guidelines\n"
    "Counseling strategies\n\n"

    "**Monitoring and Evaluation:**\n"
    "Progress evaluation\n"
    "Reviewing goals and outcomes\n"
    "Follow-up care plan\n\n"

    "---\n\n"

    "## Brief Summary\n"
    "[150–300 words | Condensed overview using standard abbreviations, must stick to the word count]\n\n"

    "**Assessment:**\n"
    "Essential patient background\n"
    "Key anthropometric data\n"
    "Critical dietary history\n"
    "Physical activity patterns\n\n"

    "**Diagnosis:**\n"
    "[Primary conditions and concerns]\n\n"

    "**Intervention:**\n"
    "Core recommendations\n"
    "Education on nutrition guidelines\n"
    "Counseling strategies\n"
    "Coordination of nutrition care\n\n"

    "**Monitoring and Evaluation:**\n"
    "Progress tracking\n"
    "Reviewing goals and outcomes\n"
    "Follow-up care plan\n\n"

    "---\n\n"

    "## Detailed Summary\n"
    "[300–400 words | Comprehensive clinical narrative with context, must stick to the word count.]\n\n"

    "**Assessment:**\n"
    "Thorough patient background\n"
    "Anthropometric data with interpretation\n"
    "Detailed dietary patterns and examples\n"
    "Physical activity profile and limitations\n\n"

    "**Diagnosis:**\n"
    "[Comprehensive diagnostic impressions]\n\n"

    "**Intervention:**\n"
    "Detailed nutrition prescriptions\n"
    "Education and counseling strategies\n\n"

    "**Monitoring and Evaluation:**\n"
    "Detailed progress tracking\n"
    "Reviewing goals and outcomes\n"
    "Follow-up timeline\n\n"

    "---\n\n"

    "## Super Detailed Summary\n"
    "[400–500 words | Maximum clinical detail including patient quotes when documented, must stick to the word count]\n\n"

    "**Assessment:**\n"
    "Exhaustive patient history\n"
    "Complete anthropometric context\n"
    "Extensive dietary history with symptoms and emotional responses\n"
    "Physical activity history, preferences, and limitations\n\n"

    "**Diagnosis:**\n"
    "[Complete diagnostic formulation with clinical reasoning]\n\n"

    "**Intervention:**\n"
    "Comprehensive nutrition strategies with rationale\n"
    "Education and counseling documentation\n\n"

    "**Monitoring and Evaluation:**\n"
    "Detailed progress evaluation\n"
    "Reviewing goals and outcomes\n"
    "Follow-up metrics and timelines\n\n"

    "---\n\n"

    "## SOAP Note\n"
    "[200–300 words | Entire SOAP section combined]\n\n"

    "**Client/Patient Name:** <confidential>\n"
    "**Assessment Date:** <confidential>\n"
    "**Submitted By:** <confidential>\n\n"

    "### Subjective\n"
    "[Patient-reported symptoms, concerns, and relevant history]\n\n"

    "### Objective\n"
    "[Observable findings, measurements, and vital signs when documented]\n\n"

    "### Assessment\n"
    "[Clinical impressions and diagnostic reasoning]\n\n"

    "### Plan\n"
    "[Bulleted action items with interventions and follow-up]\n\n"

    "---\n\n"

    "## MARKDOWN FORMATTING STANDARDS\n"
    "Headers: # (title), ## (main sections), ### (subsections)\n"
    "Tables: Use Markdown tables for structured clinical data (medications, vitals, labs)\n"
    "Lists: Use - for bullets and 1., 2., 3. for numbered lists (restart numbering per section)\n"
    "Emphasis: Use **bold** for important findings and italic for patient quotes\n"
    "Spacing: Leave blank lines between sections for readability\n\n"

    "Before finalizing, briefly review each section and adjust wording so it fits comfortably within its intended word range while remaining clinically accurate and complete.\n"
    "Return only the final Markdown output."
)
    
    # GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    # HF_MODEL = "openai/gpt-oss-20b:groq"
    
    # completion = client.chat.completions.create(
    #     model=GROQ_MODEL,
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": system_content,
    #         },
    #         {
    #             "role": "user",
    #             "content": f"Please create a structured clinical summary from the following doctor-patient transcript:\n\n{transcript_text}"
    #         },
    #     ],
    #     max_tokens=8192,
    #     temperature=0.1,
    # )

    
    # return completion.choices[0].message.content

    client = genai.Client()

            # Create user message
    user_message = types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"Please create a structured clinical summary from the following doctor-patient transcript:\n\n{transcript_text}")]
            )

    contents = [user_message]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=system_content,
            temperature=0.1,
        )
    )

    return response.text if response else ""

if __name__ == "__main__":
    # Configuration
    input_file = "transcript.pdf"
    output_md_file = "clinical_summary.md"
    output_docx_file = "clinical_summary.docx"
    masked_transcript_file = "masked_transcript.txt"
    
    try:
        # Extract and mask transcript
        print(f"Extracting transcript from {input_file}...")
        transcript = extract_transcript(input_file)

        print("Applying PHI masking...")
        masker = PHIMasker(use_consistent_hashing=True)
        masked_transcript = masker.apply_all_masking(transcript)
        
        # Save masked transcript
        with open(masked_transcript_file, 'w', encoding='utf-8') as f:
            f.write(masked_transcript)
        f.close()
        
        # Generate Markdown summary
        print("Generating clinical summary in Markdown format...")
        markdown_summary = generate_clinical_summary(masked_transcript, use_masked_data=True)
        MAX_RETRIES = 5
        retries = 0
        while len(markdown_summary) == 0 and retries < MAX_RETRIES:
            print("Received empty summary, retrying...")
            markdown_summary = generate_clinical_summary(masked_transcript, use_masked_data=True)
            retries += 1

        if retries == MAX_RETRIES:
            print("Reached maximum retry limit, please try again")
        
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