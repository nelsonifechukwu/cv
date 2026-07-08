# CV

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

# Tailor to the role AND the company (values/culture/hiring philosophy) — this is the default
uv run tailor.py                       # company research is ON by default
uv run tailor.py --no-company-research  # role-only tailoring, no web research
uv run tailor.py --refresh-company-research  # ignore the cached brief and re-research
```

### Company research

By default, `tailor.py` doesn't only tailor to the *role* — it also researches the *employer*. It identifies the hiring company from the job description and uses Claude's web search to build a brief on the company's stated values, culture, how it evaluates candidates (e.g. Arm's "10x mindset"), and any company-specific CV/application guidance. That brief is fed into CV generation and the recruiter/ATS review pass so the CV speaks to how *that* company hires — echoing the company's genuine priorities and vocabulary, but only where your real experience supports them (it is never a licence to fabricate).

The brief is cached under `company_research/<company>.md` (so a company you apply to repeatedly is only researched once) and a copy is saved next to each output PDF. Turn it off with `--no-company-research`, or force a fresh web pass with `--refresh-company-research`.

### Flags

| Flag | Effect |
| --- | --- |
| `--cover-letter` | Generate a cover letter instead of a CV |
| `--write` | Save the full variant directory; without this, only the final PDF + `.tex` are kept |
| `--model MODEL` | Anthropic model for generation (default `claude-sonnet-4-6`) |
| `--max-pages N` | Hard page cap for the CV; compressed to fit (default `2`, `0` disables) |
| `--no-review` | Skip the recruiter/ATS review-and-revise pass |
| `--company-research` / `--no-company-research` | Research the target company and tailor to its values/culture (on by default) |
| `--refresh-company-research` | Re-run company research even if a cached brief exists |

Saved variants are reused as few-shot examples (selected by TF-IDF similarity), so the output improves over time when you run with `--write`.

---

*LaTeX template forked from [knyazer/cv](https://github.com/knyazer/cv) by Roman Knyazhitskiy.*
