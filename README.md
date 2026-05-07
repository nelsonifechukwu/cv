# Elijah Nelson — CV in LaTeX

[Click here to view the pdf](./cv.pdf)

Self-contained, short, blue-ish.

* Not using fancy fonts

* Not using fancy images

* Not using fancy anything

Just a CV. Just some words.

```bash
sudo apt install texlive-full
sudo tlmgr init-usertree # on debian-based systems
sudo tlmgr install fontawesome5
pdflatex cv.tex && bibtex cv && pdflatex cv.tex
```

Or, to build it quickly you can use `just`.

*Forked from [knyazer/cv](https://github.com/knyazer/cv) — original template by Roman Knyazhitskiy.*

## How to use the tailoring script

Approximately:

- put your Anthropic API key in `.env` as `ANTHROPIC_API_KEY`
- write your own CV in `/cv.tex`
- `rm -rf variants`

then you can run `uv run tailor.py` or `uv run tailor.py --write`. The first one generates just the pdf in `/pdfs`; the second one dumps the full variant into `/variants` *and* links the pdf into `/pdfs`.
