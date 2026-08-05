# MyCV — modular master CV

Single source of truth for Elijah Nelson's CV, split into sections and assembled by `main.tex`.

## Structure

```
MyCV/
├── main.tex              # Assembles everything (compile this)
├── pub.bib -> ../pub.bib # Symlink to the shared bibliography
├── config/
│   ├── packages.tex      # \usepackage imports
│   ├── formatting.tex    # Geometry, spacing, black links, list defaults
│   └── commands.tex      # \entrytitle, \entrytitlend, \sectiontitle, details list
└── content/
    ├── personal-info.tex # Header (name, contact)
    ├── summary.tex
    ├── education.tex
    ├── research-interests.tex
    ├── research-teaching.tex
    ├── publications.tex  # \bibliography{pub} + \nocite{*}
    ├── experience.tex    # Professional Experience
    ├── projects.tex
    ├── skills.tex
    ├── certifications.tex
    ├── leadership.tex
    ├── volunteering.tex
    ├── awards.tex
    ├── media.tex
    └── interests.tex
```

## Editing

Edit the relevant `content/*.tex` file. To change section order, reorder the
`\input` lines in `main.tex`. Styling changes go in `config/`.

## Build

```sh
cd MyCV
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Tailoring

`../tailor.py` reads this folder as the master CV. It flattens all `\input`s
in-memory (via `read_master_cv()`), so the model sees the full text while the
tailored output stays a single self-contained `cv.tex` per variant. The legacy
single-file `../cv.tex` is only a fallback if this folder is absent.
