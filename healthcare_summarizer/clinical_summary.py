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

def build_contents(new_user_input: str) -> list:
    """Convert chat history to API-compatible format."""
    contents = []
    
    # Add new user input
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=new_user_input)]
        )
    )
    
    return contents


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


def generate_summary_section(transcript_text, section_type, system_prompt, max_tokens=2048):
    """Generate a specific section of the clinical summary."""
    # client = Groq()

    GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    MAX_RETRIES = 3
    retries = 0
    
    while retries < MAX_RETRIES:
        try:
            # completion = client.chat.completions.create(
            #     model=GROQ_MODEL,
            #     messages=[
            #         {"role": "system", "content": system_prompt},
            #         {
            #             "role": "user",
            #             "content": f"Create a {section_type} from this doctor-patient transcript:\n\n{transcript_text}"
            #         }
            #     ],
            #     max_tokens=max_tokens,
            #     temperature=0.1,
            # )
            
            # result = completion.choices[0].message.content.strip()

            client = genai.Client()

            # Create user message
            user_message = types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"Create a {section_type} from this doctor-patient transcript:\n\n{transcript_text}")]
            )

            contents = [user_message]

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="text/plain",
                    system_instruction=system_prompt,
                    temperature=0.1,
                )
            )

            result = response.text
            
            # Clean up response (remove code fences if LLM added them)
            if result.startswith('```'):
                result = result.replace('```markdown', '').replace('```', '').strip()
            
            if len(result) > 0:
                return result
            else:
                print(f"  Received empty response for {section_type}, retrying...")
                retries += 1
                
        except Exception as e:
            print(f"  Error generating {section_type}: {e}")
            retries += 1
    
    return f"*Error: Could not generate {section_type} after {MAX_RETRIES} attempts*"


def generate_clinical_summary(transcript_text, use_masked_data=False):
    """Generate clinical summary by calling LLM separately for each section."""
    
    base_instructions = (
        "IMPORTANT: This transcript contains masked PHI (Protected Health Information) for privacy protection. "
        "You are a medical documentation assistant specialized in creating structured clinical summaries. "
        "The conversation may be multilingual.\n\n"
        
        "CRITICAL RULES:\n"
        "- Do NOT infer or hallucinate information\n"
        "- If information is missing or unclear, write 'Not documented'\n"
        "- Preserve negations exactly (e.g., 'no fever' ≠ fever)\n"
        "- If statements conflict, use the most recent clinician statement\n"
        "- Use standard medical terminology\n"
        "- Be concise and clinically precise\n"
        "- Output ONLY Markdown - no code fences (```), no preamble, no explanation\n"
        "- Do NOT use patient and doctor names\n"
        "- Complete the entire section - do not stop abruptly\n\n"
    )
    
    sections = {
        "quick_reference": {
            "prompt": base_instructions + (
                "Create a Quick Reference section with the following subsections:\n\n"
                "### Chief Complaint\n"
                "[Brief description]\n\n"
                "### Key Problems/Diagnoses\n"
                "[Use bullet points for multiple items]\n\n"
                "### Current Medications\n"
                "[Use a Markdown table if multiple medications with details, bullet points if simple list]\n\n"
                "### Follow-Up Actions\n"
                "[Use numbered list if sequential starting from 1, bullets if not]\n\n"
                "### Monitoring Needs\n"
                "[Bullet points or brief paragraph]\n\n"
                "Target length: 150-200 words. Use Markdown formatting."
            ),
            "max_tokens": 1024
        },

                
        "detailed": {
            "prompt": base_instructions + (
                "Create a Detailed Summary (300-400 words) with these sections:\n\n"
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
                "Use Markdown formatting. Target: 400-500 words."
            ),
            "max_tokens": 2560
        },
        
        "super_detailed": {
            "prompt": base_instructions + (
                "Create a Super Detailed Summary (400-500 words) with these sections:\n\n"
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
                "Include patient quotes where relevant. Markdown format. Target: 500-600 words."
            ),
            "max_tokens": 3072
        },
        
        "goldilocks": {
            "prompt": base_instructions + (
                "Create a Goldilocks Summary (200-300 words) with these sections:\n\n"
                "**Assessment:**\n"
                "- Patient's personal history: [Brief overview of relevant personal history, work stress, injuries, illnesses]\n"
                "- Anthropometric data: [Current weight and any changes]\n"
                "- Dietary history: [Recent dietary patterns, challenges, successes, current meal examples]\n"
                "- Physical activity patterns: [Current activity level, limitations, previous exercise habits]\n\n"
                
                "**Diagnosis:**\n"
                "[List possible diagnoses or conditions]\n\n"
                
                "**Intervention:**\n"
                "- Food and/or nutrient delivery: [Specific dietary recommendations, meal suggestions, strategies]\n"
                "- Education on specific nutrition guidelines: [Educational points covered]\n"
                "- Counseling strategies: [Therapeutic approaches used]\n\n"
                
                "**Monitoring and Evaluation:**\n"
                "- Progress evaluation: [Current symptoms, improvements, tracking methods]\n"
                "- Reviewing goals and outcomes: [Acknowledgment of progress and setbacks]\n"
                "- Follow-up care plan: [Specific action items, next appointment]\n\n"
                "Use Markdown formatting. Target: 300-400 words."
            ),
            "max_tokens": 2048
        },
        
        "brief": {
            "prompt": base_instructions + (
                "Create a Brief Summary (100-200 words) with these sections:\n\n"
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
                "Use medical abbreviations. Markdown format. Target: 200-300 words."
            ),
            "max_tokens": 1536
        },
        
        "soap": {
            "prompt": base_instructions + (
                "Create a detailed SOAP Note with these sections:\n\n"
                "**Client/Patient Name:** <confidential>  \n"
                "**Assessment Date:** <confidential>  \n"
                "**Submitted By:** <confidential>\n\n"
                "### Subjective\n"
                "[Patient's reported symptoms, concerns, and history in detail]\n\n"
                
                "### Objective\n"
                "[Physical findings, vital signs, measurements - use Markdown table for vitals if available]\n\n"
                
                "### Assessment\n"
                "[Clinical impression and differential diagnosis - use bullet points, keep it detailed]\n\n"
                
                "### Plan\n"
                "[Treatment plan, medications, follow-up - use bullet points for action items]\n\n"
                "Use Markdown formatting. Target: 300-400 words. Be thorough and detailed."
            ),
            "max_tokens": 2048
        }
    }
    
    results = {}
    
    print("\nGenerating clinical summary sections...")
    print("=" * 60)
    
    for section_name, config in sections.items():
        section_display_name = section_name.replace('_', ' ').title()
        print(f"\n[{section_display_name}]")
        
        result = generate_summary_section(
            transcript_text, 
            section_display_name,
            config["prompt"],
            config["max_tokens"]
        )
        
        results[section_name] = result
        print(f"  ✓ Generated ({len(result.split())} words)")
    
    print("\n" + "=" * 60)
    print("All sections generated successfully!\n")
    
    # Consolidate into final markdown
    final_markdown = f"""# Clinical Summary

## Quick Reference

{results['quick_reference']}

---

## Goldilocks Summary

{results['goldilocks']}

---

## Brief Summary

{results['brief']}

---

## Detailed Summary

{results['detailed']}

---

## Super Detailed Summary

{results['super_detailed']}

---

## SOAP Note

{results['soap']}
"""
    
    return final_markdown


if __name__ == "__main__":
    # Configuration
    input_file = "transcript_new_with_non_english.pdf"
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
        print(f"Masked transcript saved: {masked_transcript_file}")
        
        # Generate clinical summary with modular approach
        markdown_summary = generate_clinical_summary(masked_transcript, use_masked_data=True)
        
        # Save Markdown
        with open(output_md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_summary)
        print(f"\n✓ Markdown summary saved: {output_md_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("CLINICAL SUMMARY (MARKDOWN)")
        print("="*80 + "\n")
        print(markdown_summary)
        print("\n" + "="*80 + "\n")
        
        # Convert Markdown to DOCX
        print("Converting Markdown to DOCX...")
        md_to_docx.markdown_to_docx(markdown_summary, output_docx_file)
        print(f"✓ DOCX file created successfully: {output_docx_file}")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()