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
HEAVY_DUTY_MODEL = "claude-sonnet-4-6"  # For CV / cover-letter tailoring


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


def find_variants(variants_dir, artifact_filename):
    """Finds all existing variants of the given artifact type.

    artifact_filename is the filename of the artifact inside each variant
    directory (e.g. 'cv.tex' or 'cover_letter.tex').
    """
    variants = []
    variants_path = Path(variants_dir)
    if not variants_path.exists():
        return variants

    for variant_path in variants_path.iterdir():
        if variant_path.is_dir():
            artifact_path = variant_path / artifact_filename
            position_path = variant_path / "position.txt"
            if artifact_path.exists() and position_path.exists():
                variants.append(
                    {
                        "name": variant_path.name,
                        "artifact_path": str(artifact_path),
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
        with open(variant["artifact_path"], "r") as f:
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


def generate_cover_letter(position_text, main_cv, similar_variants, model_name):
    """Generates a new cover letter using the Anthropic Claude model.

    Cover letters differ from CVs structurally: they are prose, four paragraphs,
    and require *selection* (pick one or two threads from the CV and develop them)
    rather than *reorganisation* (reorder structured CV sections). The system
    prompt enforces STAR structure for the strongest thread, bans generic phrases,
    and requires honest framing of gaps.
    """
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    with open(main_cv, "r") as f:
        main_cv_content = f.read()

    system_prompt = """You are an expert cover-letter writer. You tailor cover letters for specific job positions, drawing only on the candidate's actual CV.

# Hard rules

1. Output ONLY the raw LaTeX code. No markdown blocks, no explanations, no formatting markers like ```latex or ```.
2. DO NOT invent or add ANY information that is not explicitly present in the CV. This is the most important rule. Overclaiming is grounds for rejection of the output. This includes work experience, projects, skills, dates, quantitative claims, achievements.
3. Use the candidate's actual prose register: formal British English unless the CV indicates otherwise, no contractions, no hedging, every sentence carries information.
4. The cover letter must be in LaTeX, compilable with pdflatex, with the candidate's contact details (from the CV) at the top in a sender block. Use exactly the date provided in the user message — do not invent or guess a date.

# Structural rules

5. Four paragraphs, in this order:
   - **Opening (2 sentences).** Who the candidate is, what role they are applying for, and the one-sentence reason for fit. No generic enthusiasm. State the concrete match between candidate and role.
   - **Strongest thread (4-7 sentences).** Pick the ONE experience from the CV that maps most directly onto the job description and develop it using STAR structure:
     - Situation: the context of the work (project, problem, dataset).
     - Task: the specific responsibility or problem to solve.
     - Action: what the candidate did, with technical specifics drawn from the CV.
     - Result: the quantified outcome (accuracy, latency, scale, publication, etc.) and the explicit transferable link to the role.
   - **Secondary threads (3-5 sentences).** Two more experiences that broaden the picture. Mention briefly, do not develop in full. Use these to demonstrate range without diluting the main thread.
   - **Close (2 sentences).** Why this employer specifically (one concrete reason from the job description). Availability and willingness to discuss further.

6. SELECT, do not enumerate. The CV will have many experiences. Pick the two or three most relevant. Mentioning everything dilutes everything.

7. Acknowledge gaps honestly when they exist. If the candidate's current research is on a different topic to the role, say so and explain what transfers (methodology, research practice, related domain experience). Pretending the gap does not exist is visible to any careful reader.

# Stylistic rules

8. Banned phrases (these mark a letter as generic or AI-generated within seconds):
   - "I am passionate about..."
   - "I am excited to apply..."
   - "I believe I would be a great fit..."
   - "As you can see from my CV..."
   - "I am writing to express my interest in..."
   Do not use these. State substance directly.

9. No bullet points in the body. The letter is prose. Bullets in a cover letter signal the writer could not be bothered to write sentences.

10. Quantify everywhere quantification is honest. Numbers in the CV (accuracy, latency, scale, rank) belong in the letter. "98% detection accuracy" beats "high accuracy"; "processed terabyte-scale data" beats "processed large amounts of data".

11. Do not pad. If the strongest version of the letter is 350 words, do not stretch to 500. Length is set by content, not by appearance.

# LaTeX rules

12. Use a simple, clean letter format. Sender block (candidate's name, location, email, website if present) at the top. Date below. Recipient block ("Hiring Team" or specific name if the job description names one). Salutation. Body. Sign-off and signature.
13. Do not use the `letter` document class unless you produce a complete, self-contained, compilable document. The `article` class with manual layout is more reliable.
14. Use \\textbf{} for emphasis sparingly, only on technical terms that match the job description.
15. Keep margins reasonable (around 2.2cm), 11pt font.

# What you can do

- Select which experiences to develop
- Decide which transferable skills to emphasise
- Frame the candidate's current research in terms relevant to the role
- Choose technical vocabulary that matches the job description (when honest)

# What you cannot do

- Invent experiences, projects, skills, accomplishments, dates, or quantities
- Claim domain expertise the CV does not support
- Modify or invent education, publications, or employment history
- Mention things from the CV that do not improve the case for this specific role"""

    messages = []

    # Few-shot from prior cover letters in the same voice, if any
    for variant in similar_variants:
        with open(variant["artifact_path"], "r") as f:
            cl_content = f.read()
        with open(variant["position_path"], "r") as f:
            position_content = f.read()
        messages.append(
            {
                "role": "user",
                "content": f"For the following position:\n\n{position_content}\n\nGenerate a tailored cover letter.",
            }
        )
        messages.append({"role": "assistant", "content": cl_content})

    today_str = datetime.date.today().strftime("%-d %B %Y")
    messages.append(
        {
            "role": "user",
            "content": (
                f"Today's date is {today_str}. Use exactly this date in the letter.\n\n"
                f"For the following position:\n\n{position_text}\n\n"
                f"Generate a tailored cover letter from the following CV:\n\n{main_cv_content}"
            ),
        }
    )

    response = client.messages.create(
        model=model_name,
        max_tokens=4000,
        system=system_prompt,
        messages=messages,
    )

    try:
        generated_content = response.content[0].text.strip()
    except (IndexError, AttributeError) as e:
        error(f"Failed to extract content from Claude response: {e}")
        raise ValueError(f"Unexpected Claude API response structure: {e}")

    # Post-process: strip markdown code fences if the model added them despite instructions
    latex_blocks = re.findall(r"```latex\s*\n(.*?)\n```", generated_content, re.DOTALL)
    if latex_blocks:
        generated_content = max(latex_blocks, key=len)
    else:
        code_blocks = re.findall(r"```\s*\n(.*?)\n```", generated_content, re.DOTALL)
        if code_blocks:
            generated_content = max(code_blocks, key=len)

    return generated_content.strip()


def generate_pdf(variant_dir, tex_filename):
    """Generates a PDF from a LaTeX file using pdflatex (and bibtex for CVs).

    tex_filename is the name of the .tex file (e.g. 'cv.tex' or 'cover_letter.tex').
    For CVs we run pdflatex -> bibtex -> pdflatex to resolve bibliography references.
    For cover letters we run pdflatex twice (no bibliography).
    """
    variant_path = Path(variant_dir)
    tex_path = variant_path / tex_filename

    if not tex_path.exists():
        raise FileNotFoundError(f"{tex_filename} not found in {variant_dir}")

    base_name = Path(tex_filename).stem
    is_cv = (tex_filename == "cv.tex")

    try:
        # First pdflatex run
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_filename],
            cwd=str(variant_path),
            check=True,
            capture_output=True,
            text=True,
        )

        # Bibtex pass for CV (cover letter has no bibliography)
        if is_cv:
            subprocess.run(
                ["bibtex", base_name],
                cwd=str(variant_path),
                check=True,
                capture_output=True,
                text=True,
            )

        # Second pdflatex run to resolve references
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_filename],
            cwd=str(variant_path),
            check=True,
            capture_output=True,
            text=True,
        )

        pdf_path = variant_path / f"{base_name}.pdf"
        if pdf_path.exists():
            return str(pdf_path)
        else:
            raise FileNotFoundError(
                f"PDF generation completed but {base_name}.pdf was not created"
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
    parser = argparse.ArgumentParser(description="Tailor a CV or cover letter to a job description.")
    parser.add_argument(
        "--model",
        type=str,
        default=HEAVY_DUTY_MODEL,
        help=f"The Anthropic Claude model to use for generation (default: {HEAVY_DUTY_MODEL}).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Save the full variant to the variants directory. If not set, only the PDF is saved to the pdfs directory.",
    )
    parser.add_argument(
        "--cover-letter",
        action="store_true",
        help="Generate a cover letter instead of a CV. The source CV is still read from cv.tex.",
    )
    args = parser.parse_args()

    # Set artifact-type-specific filenames and directories
    if args.cover_letter:
        artifact_label = "cover letter"
        artifact_filename = "cover_letter.tex"
        variants_dir = Path("cover_letters")
        pdfs_dir = Path("cover_letter_pdfs")
        generator_fn = generate_cover_letter
    else:
        artifact_label = "CV"
        artifact_filename = "cv.tex"
        variants_dir = Path("variants")
        pdfs_dir = Path("pdfs")
        generator_fn = generate_cv

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
        tmp_prefix = "cover_letter_" if args.cover_letter else ""
        new_variant_dir = Path("/tmp") / f"{tmp_prefix}{summary_name}"
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

    inform(f"\nFinding existing {artifact_label} variants...")
    variants = find_variants(variants_dir, artifact_filename)
    inform(f"Found {len(variants)} existing variants")

    inform("\nFinding similar variants...")
    similar_variants = find_similar_variants(clean_position_text, variants)
    if similar_variants:
        inform(f"Found {len(similar_variants)} similar variants:")
        for variant in similar_variants:
            inform(f"  - {variant['name']}")
    else:
        inform("No similar variants found")

    inform(f"\nGenerating tailored {artifact_label} using {args.model}...")
    new_artifact = generator_fn(
        clean_position_text, main_cv_path, similar_variants, args.model
    )
    success(f"{artifact_label.capitalize()} generation complete!")

    inform(f"\nSaving {artifact_label} variant...")
    new_variant_dir.mkdir(parents=True, exist_ok=True)

    (new_variant_dir / artifact_filename).write_text(new_artifact)
    (new_variant_dir / "position.txt").write_text(clean_position_text)

    # Link bibliography file if it exists (only relevant for CV)
    if not args.cover_letter:
        bib_file = Path("pub.bib")
        if bib_file.exists():
            bib_link = new_variant_dir / bib_file.name
            if not bib_link.exists():
                bib_link.symlink_to(bib_file.resolve())
                inform(f"Linked {bib_file} to variant directory")

    inform(f"Variant saved to: {new_variant_dir}")

    inform("\nGenerating PDF...")
    try:
        pdf_path = generate_pdf(new_variant_dir, artifact_filename)
        success(f"PDF generated successfully: {pdf_path}")
    except Exception as e:
        error(f"ERROR: Failed to generate PDF: {e}")
        warn(f"The LaTeX file is saved at: {new_variant_dir / artifact_filename}")
        error("\nTraceback:")
        error(traceback.format_exc())
        raise

    # Link or copy PDF (and .tex source) to pdfs directory
    pdfs_dir.mkdir(exist_ok=True)
    final_pdf_path = pdfs_dir / f"{summary_name}.pdf"
    final_tex_path = pdfs_dir / f"{summary_name}.tex"
    tex_source_path = new_variant_dir / artifact_filename

    if args.write:
        # When storing variant, create symlinks instead of copying
        inform(f"\nLinking PDF and .tex to {pdfs_dir}/ directory...")
        for link_path, target_path in (
            (final_pdf_path, Path(pdf_path)),
            (final_tex_path, tex_source_path),
        ):
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(target_path.resolve())
        success(f"PDF linked to: {final_pdf_path}")
        success(f"TeX linked to: {final_tex_path}")
    else:
        # When using temp directory, copy both before cleanup
        inform(f"\nCopying PDF and .tex to {pdfs_dir}/ directory...")
        shutil.copy2(pdf_path, final_pdf_path)
        shutil.copy2(tex_source_path, final_tex_path)
        success(f"PDF saved to: {final_pdf_path}")
        success(f"TeX saved to: {final_tex_path}")

    if args.write:
        success(f"\nSUCCESS: Generated new {artifact_label} variant in: {new_variant_dir}")
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