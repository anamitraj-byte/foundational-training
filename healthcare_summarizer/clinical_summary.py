import os
from dotenv import load_dotenv
import docx
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
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )
    
    # Add note about masked data if applicable
    system_content = (
        "You are a medical documentation assistant specialized in creating structured clinical summaries. "
        "Extract and organize information from doctor-patient conversation transcripts into a well-formatted document.\n\n"
        
        "Format your output as text suitable for a DOCX file with the following sections:\n\n"
        
        "CHIEF COMPLAINT\n"
        "Main reason for visit\n\n"
        
        "KEY PROBLEMS/DIAGNOSES\n"
        "Current medical issues identified\n\n"
        
        "CURRENT MEDICATIONS\n"
        "List all medications mentioned with dosages\n\n"
        
        "FOLLOW-UP ACTIONS\n"
        "Appointments, tests, referrals scheduled\n\n"
        
        "MONITORING NEEDS\n"
        "What should be tracked (vitals, symptoms, labs)\n\n"
        
        "SOAP NOTE\n"
        "Subjective: Patient's reported symptoms and history\n"
        "Objective: Physical findings, vital signs, test results\n"
        "Assessment: Clinical impression and diagnoses\n"
        "Plan: Treatment plan and recommendations\n\n"
        
        "FORMATTING REQUIREMENTS:\n"
        "- Use section headings in ALL CAPS\n"
        "- Use simple bullet points with hyphens (-) or asterisks (*)\n"
        "- Avoid special characters that may not render properly in DOCX\n"
        "- Use standard line breaks between sections\n"
        "- Keep formatting simple and clean\n"
        "- Do not use markdown formatting (no ##, **, etc.)\n\n"
        
        "CONTENT REQUIREMENTS:\n"
        "- Be concise, accurate, and use standard medical terminology\n"
        "- If information is not present in the transcript, note it as 'Not documented'\n"
        "- If the max token limit is reached, shorten the result instead of cutting off mid-section\n"
        "- Ensure all content is appropriate for a professional medical document\n\n"
        
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


def save_summary_to_file(summary, output_path):
    """Save the clinical summary to a text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Clinical summary saved to: {output_path}")



if __name__ == "__main__":
    # Configuration
    input_file = "transcript.pdf"  # or "transcript.pdf"
    output_file = "clinical_summary.txt"
    masked_output_file = "clinical_summary_masked.txt"
    
    
    try:
        # Extract transcript from file
        print(f"Extracting transcript from {input_file}...")
        transcript = extract_transcript(input_file)
        print(f"Transcript extracted successfully. Length: {len(transcript)} characters\n")


        print("Applying PHI masking...")
        masker = PHIMasker(use_consistent_hashing=True)
        masked_transcript = masker.apply_all_masking(transcript)
        print(f"Masking applied. Masked transcript length: {len(masked_transcript)} characters\n")
        
        # Optionally save masked transcript
        with open("masked_transcript.txt", 'w', encoding='utf-8') as f:
            f.write(masked_transcript)
        print("Masked transcript saved to: masked_transcript.txt\n")
        
        # Generate summary from masked data
        print("Generating clinical summary from masked data...")
        masked_summary = generate_clinical_summary(masked_transcript, use_masked_data=True)
        
        # Print and save masked summary
        print("\n" + "="*80)
        print("CLINICAL SUMMARY (WITH PHI MASKING)")
        print("="*80 + "\n")
        print(masked_summary)
        print("\n" + "="*80)
        
        save_summary_to_file(masked_summary, masked_output_file)

        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found. Please check the file path.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
