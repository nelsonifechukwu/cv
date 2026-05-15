# Elijah Nelson — CV

A LaTeX CV plus a small Python script that tailors it to a job description using Claude.

[View the PDF](./cv.pdf)

---

## Build the CV

```bash
pdflatex cv.tex && bibtex cv && pdflatex cv.tex
```

Or use `just build`.

Dependencies (Debian/Ubuntu):

```bash
sudo apt install texlive-full
sudo tlmgr install fontawesome5
```

## Tailor to a job

The `tailor.py` script reads a job description from your clipboard, generates a tailored CV or cover letter via Claude, and compiles it.

### Setup

1. Put your Anthropic API key in `.env`:

   ```env
   ANTHROPIC_API_KEY=sk-ant-...
   ```

2. Install Python deps with `uv sync`.

### Usage

Copy a job description to your clipboard, then:

```bash
# CV → pdfs/<position>.pdf  (+ .tex source)
uv run tailor.py

# Cover letter → cover_letter_pdfs/<position>.pdf  (+ .tex source)
uv run tailor.py --cover-letter

# Persist the full variant under variants/ or cover_letters/ for future few-shot context
uv run tailor.py --write
uv run tailor.py --cover-letter --write

# Override the generation model (default: claude-sonnet-4-6)
uv run tailor.py --model claude-opus-4-7
```

### Flags

| Flag | Effect |
| --- | --- |
| `--cover-letter` | Generate a cover letter instead of a CV |
| `--write` | Save the full variant directory; without this, only the final PDF + `.tex` are kept |
| `--model MODEL` | Anthropic model for generation (default `claude-sonnet-4-6`) |

Saved variants are reused as few-shot examples (selected by TF-IDF similarity), so the output improves over time when you run with `--write`.

---

*LaTeX template forked from [knyazer/cv](https://github.com/knyazer/cv) by Roman Knyazhitskiy.*
