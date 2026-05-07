import os
import pyperclip
from anthropic import Anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import re
import argparse
import subprocess
import shutil
import traceback
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Model configuration
LIGHTWEIGHT_MODEL = "claude-haiku-4-5"  # For extraction tasks
HEAVY_DUTY_MODEL = "claude-sonnet-4-6"  # For CV tailoring


# Color output functions
def inform(message):
    """Print an informational message (no color)."""
    print(message)


def success(message):
    """Print a success message (green)."""
    print(f"\033[92m{message}\033[0m")


def warn(message):
    """Print a warning message (yellow)."""
    print(f"\033[93m{message}\033[0m")


def error(message):
    """Print an error message (red)."""
    print(f"\033[91m{message}\033[0m")


def get_clipboard_content():
    """Gets the content of the clipboard."""
    return pyperclip.paste()


def extract_position_name(position_text):
    """Extracts a concise position name using a lightweight LLM."""
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    try:
        response = client.messages.create(
            model=LIGHTWEIGHT_MODEL,
            max_tokens=50,
            system="""You are a helpful assistant that extracts job position titles. Respond with only a concise position identifier suitable for a filename.

Format: Use lowercase with underscores, combining role, domain/specialization (if relevant), and company.

Examples:
- "Applied ML Engineer (Speech) at Apple" → "applied_ml_speech_apple"
- "Research Scientist - ML & Computer Vision at Meta" → "research_ml_computer_vision_meta"
- "Software Engineer, Full Stack at Google" → "swe_fullstack_google"
- "Machine Learning for Science at AI2" → "ml_for_science_ai2"

Be concise and avoid unnecessary words. In case you cannot infer the position name, return 'UNDETECTABLE'.""",
            messages=[
                {
                    "role": "user",
                    "content": f"Extract the position title and company name from this job description:\n\n{position_text[:1000]}",
                },
            ],
        )
        return response.content[0].text.strip()
    except Exception as e:
        error(f"Position extraction failed: {e}")
        return "Position (extraction failed)"


def extract_position_content(position_text):
    """Extracts only the relevant position description content, removing navigation, footers, etc."""
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    try:
        response = client.messages.create(
            model=LIGHTWEIGHT_MODEL,
            max_tokens=2000,
            system="You are a helpful assistant that extracts job descriptions. Extract ONLY the relevant job position content including: job title, company name, job description, requirements, responsibilities, qualifications, benefits, and any other job-related information. Remove any website navigation, footers, sidebars, cookies notices, or unrelated content. Return only the clean job description.",
            messages=[
                {
                    "role": "user",
                    "content": f"Extract the job description content from this text:\n\n{position_text}",
                },
            ],
        )
        return response.content[0].text.strip()
    except Exception as e:
        warn(f"Content extraction failed ({e}), using original text")
        return position_text


def check_suitability(position_text, cv_path):
    """Checks if the candidate is suitable for the position and returns (is_suitable, reason)."""
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    with open(cv_path, "r") as f:
        cv_content = f.read()

    try:
        response = client.messages.create(
            model=LIGHTWEIGHT_MODEL,
            max_tokens=200,
            system="""You are a job application advisor. Analyze whether a candidate's CV matches a job position.

Respond in the following format:
SUITABLE: [YES/NO]
REASON: [Brief explanation in 1-2 sentences]

Consider:
- Required qualifications vs. candidate's education/experience
- Technical skills match
- Years of experience required vs. actual experience
- Domain expertise alignment
- Seniority level match

Be realistic but not overly harsh. Minor gaps are acceptable.""",
            messages=[
                {
                    "role": "user",
                    "content": f"Job Position:\n\n{position_text}\n\nCandidate CV:\n\n{cv_content}\n\nIs this candidate suitable for this position?",
                },
            ],
        )

        result = response.content[0].text.strip()

        # Parse the response
        lines = result.split("\n")
        suitable_line = next((l for l in lines if l.startswith("SUITABLE:")), "")
        reason_line = next((l for l in lines if l.startswith("REASON:")), "")

        assert "YES" in suitable_line.upper() or "NO" in suitable_line.upper()
        assert not ("YES" in suitable_line.upper() and "NO" in suitable_line.upper())

        is_suitable = "YES" in suitable_line.upper()
        reason = reason_line.replace("REASON:", "").strip() if reason_line else result

        return is_suitable, reason
    except Exception as e:
        warn(f"Suitability check failed ({e}), proceeding anyway")
        return True, "Suitability check failed"


def find_variants(variants_dir):
    """Finds all existing CV variants."""
    variants = []
    variants_path = Path(variants_dir)
    if not variants_path.exists():
        return variants

    for variant_path in variants_path.iterdir():
        if variant_path.is_dir():
            cv_path = variant_path / "cv.tex"
            position_path = variant_path / "position.txt"
            if cv_path.exists() and position_path.exists():
                variants.append(
                    {
                        "name": variant_path.name,
                        "cv_path": str(cv_path),
                        "position_path": str(position_path),
                    }
                )
    return variants


def find_similar_variants(position_text, variants, n=2):
    """Finds the most similar variants to the given position text."""
    if not variants:
        return []

    variant_positions = []
    for variant in variants:
        with open(variant["position_path"], "r") as f:
            variant_positions.append(f.read())

    vectorizer = TfidfVectorizer().fit_transform([position_text] + variant_positions)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    scores = cosine_matrix[0][1:]

    top_indices = scores.argsort()[-n:][::-1]
    return [variants[i] for i in top_indices]


def generate_cv(position_text, main_cv, similar_variants, model_name):
    """Generates a new CV using the Anthropic Claude model."""
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Read bibliography file for context
    bib_content = ""
    bib_path = Path("pub.bib")
    if bib_path.exists():
        bib_content = bib_path.read_text()

    system_prompt = """You are an expert CV writer. Your task is to tailor a CV for a specific job position based on the provided examples and a main CV.

CRITICAL RULES:
1. Output ONLY the raw LaTeX code - no markdown code blocks, no explanations, no formatting markers like ```latex or ```.
2. DO NOT invent or add ANY information that is not explicitly present in the main CV. This includes:
   - Work experience details
   - Projects
   - Skills
   - Dates
   - Achievements
   - Publications (the \\bibliography command references a .bib file - do NOT modify the publications section or \\nocite{*} command)
3. EDUCATION SECTION: The Education section MUST remain completely unchanged. Do NOT modify, reorder, rephrase, or alter any part of the Education section in any way.
4. You can ONLY:
   - Reorder sections to highlight most relevant experience (EXCEPT Education which must stay unchanged)
   - Adjust emphasis on certain bullet points in the Experience section
   - Remove bullet points from any section, or compress/highlight bullet points anywhere to convey the suitability better
   - Rephrase existing content in the Experience section to better match the job description (without changing facts)
   - Modify the Summary section to align with the position
   - Select and emphasize relevant work experiences, projects, and skills
   - COMPRESS detailed explanations from the main CV to fit space constraints or improve readability
   - SELECTIVELY INCLUDE OR EXCLUDE experiences based on relevance (e.g., include high school robotics experience like RoboCup for robotics positions where it might matter, but exclude it for pure software engineering positions, unless relevant)
   - Remove or condense less relevant experiences to make room for more pertinent ones
5. The bibliography file (pub.bib) is provided for your reference so you understand what publications exist, but you MUST NOT modify the publications list or citations.
6. "Lying" or inventing information is strictly forbidden and will result in rejection of the output
7. STRONGLY DISCOURAGED: Avoid changing LaTeX formatting, spacing, or structural commands (like \\sectiontitle, \\entrytitle, geometry settings, etc.). The existing formatting is carefully tuned and easy to break. Focus on content rather than presentation.
8. Prefer to compress the CV to be as short as possible, but without losing important information
9. DO NOT use markdown syntax (e.g. **xyz**, _abc_, #heading). Use LaTeX, e.g. \\textbf{xyz}
10. DO NOT mention the company name, or mention projects/results if they don't improve the overall picture. For example, you might want to mention "used AdamW with weight decay" only if the position is concerned with some research on precisely AdamW, since it's an assumption that everybody uses AdamW to train models these days.
11. Try to keep the CV as concise as possible; the HR does not have the time to read through the whole thing!

Your output must be valid LaTeX that can be directly compiled."""

    messages = []

    # Add context about publications if bib file exists
    if bib_content:
        messages.append(
            {
                "role": "user",
                "content": f"For reference, here are the actual publications (DO NOT modify the publications section):\n\n{bib_content}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": "Understood. I will not modify the publications section and will reference this information when tailoring the CV.",
            }
        )
    else:
        warn(
            "Bib content was not extracted. The tailoring model is unaware of the publication list."
        )

    for variant in similar_variants:
        with open(variant["cv_path"], "r") as f:
            cv_content = f.read()
        with open(variant["position_path"], "r") as f:
            position_content = f.read()
        messages.append(
            {
                "role": "user",
                "content": f"For the following position:\n\n{position_content}\n\nGenerate a tailored CV.",
            }
        )
        messages.append({"role": "assistant", "content": cv_content})

    with open(main_cv, "r") as f:
        main_cv_content = f.read()

    messages.append(
        {
            "role": "user",
            "content": f"For the following position:\n\n{position_text}\n\nGenerate a tailored CV from the following CV:\n\n{main_cv_content}",
        }
    )

    response = client.messages.create(
        model=model_name,
        max_tokens=8000,
        system=system_prompt,
        messages=messages,
    )
    
    try:
        generated_content = response.content[0].text.strip()
    except (IndexError, AttributeError) as e:
        error(f"Failed to extract content from Claude response: {e}")
        error(f"Response content structure: {response.content}")
        if response.content:
            error(f"First content item: {response.content[0]}")
            error(f"Content item type: {type(response.content[0])}")
            if hasattr(response.content[0], '__dict__'):
                error(f"Content item attributes: {response.content[0].__dict__}")
        raise ValueError(f"Unexpected Claude API response structure: {e}")

    # Post-process: Extract LaTeX code from markdown blocks if present
    # Try to find ```latex ... ``` blocks first
    latex_blocks = re.findall(r"```latex\s*\n(.*?)\n```", generated_content, re.DOTALL)
    if latex_blocks:
        # Use the largest block
        generated_content = max(latex_blocks, key=len)
    else:
        # Try generic ``` ... ``` blocks
        code_blocks = re.findall(r"```\s*\n(.*?)\n```", generated_content, re.DOTALL)
        if code_blocks:
            # Use the largest block
            generated_content = max(code_blocks, key=len)
        # Otherwise, use the content as-is

    return generated_content.strip()


def generate_pdf(variant_dir):
    """Generates a PDF from the CV LaTeX file using pdflatex and bibtex."""
    variant_path = Path(variant_dir)
    cv_tex_path = variant_path / "cv.tex"

    if not cv_tex_path.exists():
        raise FileNotFoundError(f"cv.tex not found in {variant_dir}")

    try:
        # First pdflatex run
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "cv.tex"],
            cwd=str(variant_path),
            check=True,
            capture_output=True,
            text=True,
        )

        # Run bibtex
        subprocess.run(
            ["bibtex", "cv"],
            cwd=str(variant_path),
            check=True,
            capture_output=True,
            text=True,
        )

        # Second pdflatex run to resolve references
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "cv.tex"],
            cwd=str(variant_path),
            check=True,
            capture_output=True,
            text=True,
        )

        pdf_path = variant_path / "cv.pdf"
        if pdf_path.exists():
            return str(pdf_path)
        else:
            raise FileNotFoundError(
                "PDF generation completed but cv.pdf was not created"
            )

    except subprocess.CalledProcessError as e:
        error(f"Error during PDF generation: {e}")
        error(f"stdout: {e.stdout}")
        error(f"stderr: {e.stderr}")
        error("\nTraceback:")
        error(traceback.format_exc())
        raise


def main():
    """The main function."""
    parser = argparse.ArgumentParser(description="Tailor a CV to a job description.")
    parser.add_argument(
        "--model",
        type=str,
        default=HEAVY_DUTY_MODEL,
        help=f"The Anthropic Claude model to use for CV generation (default: {HEAVY_DUTY_MODEL}).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Save the full variant to variants/ directory. If not set, only the PDF will be saved to pdfs/.",
    )
    args = parser.parse_args()

    inform("Reading clipboard content...")
    position_text = get_clipboard_content()
    if not position_text:
        error("ERROR: Clipboard is empty.")
        return

    inform(f"\nExtracting position name (using {LIGHTWEIGHT_MODEL})...")
    position_name = extract_position_name(position_text)
    inform(f"Position: {position_name}")

    if position_name == "UNDETECTABLE":
        error(
            "ERROR: Could not extract position name from clipboard content. Please ensure the clipboard contains a valid job description."
        )
        raise ValueError("Position name undetectable")

    # Generate variant name and check if it already exists
    now = datetime.datetime.now()
    clean_name = position_name.replace(" ", "_").replace("/", "_").lower()
    clean_name = re.sub(r"[^\w\-_]", "", clean_name)
    summary_name = f"{clean_name}_{now.year}_{now.month}"

    # Determine where to save based on --write flag
    variants_dir = Path("variants")
    if args.write:
        new_variant_dir = variants_dir / summary_name

        if new_variant_dir.exists():
            error(f"\nERROR: Variant already exists: {new_variant_dir}")
            print("Do you want to override the existing variant? (yes/no): ", end="")

            user_input = input().strip().lower()
            if user_input not in ["yes", "y"]:
                inform("\nAborted by user.")
                return
            inform("Overriding existing variant...")
    else:
        # Use temporary directory
        new_variant_dir = Path("/tmp") / summary_name
        if new_variant_dir.exists():
            shutil.rmtree(new_variant_dir)
        inform(f"Using temporary directory: {new_variant_dir}")

    inform(f"\nExtracting clean position content (using {LIGHTWEIGHT_MODEL})...")
    clean_position_text = extract_position_content(position_text)
    inform(f"Extracted {len(clean_position_text)} characters of position content")

    inform(f"\nChecking suitability (using {LIGHTWEIGHT_MODEL})...")
    main_cv_path = "cv.tex"
    is_suitable, reason = check_suitability(clean_position_text, main_cv_path)

    if is_suitable:
        success(f"Suitability check passed: {reason}")
    else:
        warn("Warning: You may not be suitable for this position!\n")
        warn(f"Reason: {reason}")
        print("\nDo you want to proceed anyway? (yes/no): ", end="")

        user_input = input().strip().lower()
        if user_input not in ["yes", "y"]:
            inform("\nAborted by user.")
            return

    inform("\nFinding existing CV variants...")
    variants = find_variants(variants_dir)
    inform(f"Found {len(variants)} existing variants")

    inform("\nFinding similar variants...")
    similar_variants = find_similar_variants(clean_position_text, variants)
    if similar_variants:
        inform(f"Found {len(similar_variants)} similar variants:")
        for variant in similar_variants:
            inform(f"  - {variant['name']}")
    else:
        inform("No similar variants found")

    inform(f"\nGenerating tailored CV using {args.model}...")
    new_cv = generate_cv(
        clean_position_text, main_cv_path, similar_variants, args.model
    )
    success("CV generation complete!")

    inform("\nSaving CV variant...")
    new_variant_dir.mkdir(parents=True, exist_ok=True)

    (new_variant_dir / "cv.tex").write_text(new_cv)
    (new_variant_dir / "position.txt").write_text(clean_position_text)

    # Link bibliography file if it exists
    bib_file = Path("pub.bib")
    if bib_file.exists():
        bib_link = new_variant_dir / bib_file.name
        if not bib_link.exists():
            bib_link.symlink_to(bib_file.resolve())
            inform(f"Linked {bib_file} to variant directory")

    inform(f"Variant saved to: {new_variant_dir}")

    inform("\nGenerating PDF...")
    try:
        pdf_path = generate_pdf(new_variant_dir)
        success(f"PDF generated successfully: {pdf_path}")
    except Exception as e:
        error(f"ERROR: Failed to generate PDF: {e}")
        warn(f"The LaTeX file is saved at: {new_variant_dir / 'cv.tex'}")
        error("\nTraceback:")
        error(traceback.format_exc())
        raise

    # Link or copy PDF to pdfs/ directory
    pdfs_dir = Path("pdfs")
    pdfs_dir.mkdir(exist_ok=True)
    final_pdf_path = pdfs_dir / f"{summary_name}.pdf"

    if args.write:
        # When storing variant, create symlink instead of copying
        inform("\nLinking PDF to pdfs/ directory...")
        # Remove existing symlink or file if it exists
        if final_pdf_path.exists() or final_pdf_path.is_symlink():
            final_pdf_path.unlink()
        # Create symlink to the PDF in the variant directory
        final_pdf_path.symlink_to(Path(pdf_path).resolve())
        success(f"PDF linked to: {final_pdf_path}")
    else:
        # When using temp directory, copy the PDF before cleanup
        inform("\nCopying PDF to pdfs/ directory...")
        shutil.copy2(pdf_path, final_pdf_path)
        success(f"PDF saved to: {final_pdf_path}")

    if args.write:
        success(f"\nSUCCESS: Generated new CV variant in: {new_variant_dir}")
    else:
        # Clean up temporary directory
        inform(f"Cleaning up temporary directory: {new_variant_dir}")
        shutil.rmtree(new_variant_dir)

    success(f"Final PDF location: {final_pdf_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error(f"\nFATAL ERROR: {e}")
        error("\nTraceback:")
        error(traceback.format_exc())
        exit(1)
