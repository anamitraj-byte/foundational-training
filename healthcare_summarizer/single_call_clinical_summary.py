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
    client = Groq()
    
    system_content = (
        "IMPORTANT: This transcript contains masked PHI (Protected Health Information) for privacy protection."
        "You are a medical documentation assistant specialized in creating structured clinical summaries. "
        "Extract and organize information from doctor-patient conversation transcripts. The conversation may be multilingual\n\n"

        "Return your output as clean, well-formatted MARKDOWN text.\n\n"

        "CRITICAL RULES:\n"
        "- Do NOT infer or hallucinate information\n"
        "- If information is missing or unclear, write 'Not documented'\n"
        "- Preserve negations exactly (e.g., 'no fever' ≠ fever)\n"
        "- If statements conflict, use the most recent clinician statement\n"
        "- Use standard medical terminology\n"
        "- Be concise and clinically precise\n"
        "- Complete all sections - do not stop abruptly if max token limit is reached\n"
        "- Output ONLY Markdown - no code fences (```), no preamble, no explanation\n"
        "- Follow the specified length for each summary"
        "- Do NOT skip any subheading/heading"
        "- Start your response directly with: # Clinical Summary\n\n"

        "CONTENT REQUIREMENTS:\n"
        "- Be concise, accurate, and use standard medical terminology\n"
        "- If information is not present, write 'Not documented'\n"
        "- IMPORTANT: Do not stop abruptly if max token limit is reached, make sure to complete the summary with all critical information.\n"
        "- Return ONLY Markdown text - no code fences (```), no preamble, no explanation\n"
        "- SOAP Notes must be detailed\n"
        "- Start your response directly with: # Clinical Summary\n\n"

        "STRUCTURE:\n"
        "# Clinical Summary\n\n"

        "## Quick Reference\n\n"

        "### Chief Complaint\n"
        "[Brief description]\n\n"

        "### Key Problems/Diagnoses\n"
        "[Use bullet points for multiple items]\n\n"

        "### Current Medications\n"
        "[Use a Markdown table if multiple medications with details]\n"
        "[Use bullet points if simple list]\n\n"

        "### Follow-Up Actions\n"
        "[Use numbered list if sequential starting from 1, bullets if not]\n\n"

        "### Monitoring Needs\n"
        "[Bullet points or brief paragraph]\n\n"

        "---\n\n"

        "## Goldilocks Summary\n"
        "[300-400 words]\n"
        "**Assessment:**\n"
        "- Patient's personal history: [Brief overview of relevant personal history, work stress, injuries, illnesses]\n"
        "- Anthropometric data: [Current weight and any changes]\n"
        "- Dietary history: [Recent dietary patterns, challenges, successes, current meal examples]\n"
        "- Physical activity patterns: [Current activity level, limitations, previous exercise habits]\n\n"

        "**Diagnosis:**\n"
        "[List possible diagnoses or conditions]\n\n"

        "**Intervention:**\n"
        "- Food and/or nutrient delivery: [Specific dietary recommendations, meal suggestions, strategies]\n"
        "- Education on specific nutrition guidelines: [Educational points covered about nutrition, metabolism, behavior change]\n"
        "- Counseling strategies: [Therapeutic approaches used, motivational interviewing, psychoeducation]\n\n"

        "**Monitoring and Evaluation:**\n"
        "- Progress evaluation: [Current symptoms, improvements, tracking methods]\n"
        "- Reviewing goals and outcomes: [Acknowledgment of progress and setbacks]\n"
        "- Follow-up care plan: [Specific action items, exercise plans, next appointment]\n\n"

        "---\n\n"

        "## Brief Summary\n"
        "[200-300 words]\n"
        "**Assessment:**\n"
        "- Patient's personal history: [Condensed version with abbreviations where appropriate (Hx, ~)]\n"
        "- Anthropometric data: [Weight and changes]\n"
        "- Dietary history: [Key dietary events and patterns]\n"
        "- Physical activity patterns: [Current status and history]\n\n"

        "**Diagnosis:**\n"
        "[List possible conditions]\n\n"

        "**Intervention:**\n"
        "- Food and/or nutrient delivery: [Abbreviated recommendations]\n"
        "- Education on specific nutrition guidelines: [Key educational points]\n"
        "- Counseling strategies: [Brief description of approach]\n"
        "- Coordination of nutrition care: [Any care coordination notes]\n\n"

        "**Monitoring and Evaluation:**\n"
        "- Progress evaluation: [Tracking methods]\n"
        "- Reviewing goals and outcomes: [Specific goals]\n"
        "- Follow-up care plan: [Next steps and appointment]\n\n"

        "---\n\n"

        "## Detailed Summary\n"
        "[400-500 words]\n"
        "**Assessment:**\n"
        "- Patient's personal history: [Comprehensive personal history with more context and details]\n"
        "- Anthropometric data: [Weight with context about changes]\n"
        "- Dietary history: [Detailed dietary patterns, specific meals, symptoms, preferences]\n"
        "- Physical activity patterns, preferences, and limitations: [Comprehensive activity history]\n\n"

        "**Diagnosis:**\n"
        "[Detailed list of possible conditions with descriptive context]\n\n"

        "**Intervention:**\n"
        "- Food and/or nutrient delivery: [Detailed recommendations with examples]\n"
        "- Education on specific nutrition guidelines, physical activity, health behaviors or other nutritional advice: [Comprehensive educational content with explanations]\n"
        "- Counseling strategies: [Specific therapeutic approaches used]\n\n"

        "**Monitoring and Evaluation:**\n"
        "- Progress evaluation: [How progress will be tracked]\n"
        "- Reviewing goals and outcomes: [Specific accomplishments and acknowledgments]\n"
        "- Follow-up care plan: [Detailed plan with timing and specifics]\n\n"

        "---\n\n"

        "## Super Detailed Summary\n"
        "[500-600 words]\n"
        "**Assessment:**\n"
        "- Patient's personal history: [Extremely comprehensive with patient quotes and specific details]\n"
        "- Anthropometric data: [Weight with full context and patient's own descriptions]\n"
        "- Dietary history: [Extensive detail including patient quotes, specific symptoms, emotional responses]\n"
        "- Physical activity patterns, preferences, and limitations: [Complete history with specific details]\n\n"

        "**Diagnosis:**\n"
        "[Comprehensive list with full medical terminology and context]\n\n"

        "**Intervention:**\n"
        "- Food and/or nutrient delivery: [Extensive recommendations with multiple options and details]\n"
        "- Education on specific nutrition guidelines, physical activity, health behaviors or other nutritional advice: [Complete educational content with patient quotes and detailed explanations]\n"
        "- Counseling strategies: [Detailed description of therapeutic approaches]\n\n"

        "**Monitoring and Evaluation:**\n"
        "- Progress evaluation: [Detailed current status with patient quotes]\n"
        "- Reviewing goals and outcomes: [Complete acknowledgment with context]\n"
        "- Follow-up care plan: [Comprehensive plan with specific timing and tracking methods]\n\n"

        "---\n\n"

        "## SOAP Note\n"
        "**Client/Patient Name:** <confidential>  \n"
        "**Assessment Date:** <confidential>  \n"
        "**Submitted By:** <confidential>\n\n"
        "### Subjective\n"
        "[Patient's reported symptoms and history]\n\n"
        
        "### Objective\n"
        "[Physical findings, vital signs - consider table for vitals]\n\n"
        
        "### Assessment\n"
        "[Clinical impression, use bullet points, keep it detailed]\n\n"
        
        "### Plan\n"
        "[Use bullet points for action items]\n\n"

        "---\n\n"

        "MARKDOWN FORMATTING GUIDELINES:\n"
        "- Use # for title, ## for main sections, ### for subsections\n"
        "- Use Markdown tables for structured data (medications, vitals, lab results)\n"
        "- Use - for unordered bullet points\n"
        "- Use 1., 2., 3. for numbered lists (start from 1 for each section)\n"
        "- Use **bold** for section headers and important findings\n"
        "- Use *italic* for patient quotes or emphasis\n"
        "- Use --- to separate major sections\n"
        "- Use blank lines between sections\n"
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

    return response.text

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