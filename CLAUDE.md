# CV & Résumé Rulebook

This repository is Elijah Nelson's CV system. This file is the **single source of truth for how every CV and résumé here is written and tailored**. `tailor.py` reads this file and injects it into the generation prompt, so every job-tailored variant must obey these rules automatically.

## How the system works

- **Master source:** the modular LaTeX in `MyCV/` (`main.tex` assembles `config/*` styling + `content/*.tex` sections; `pub.bib` is symlinked). This is the comprehensive master — it holds everything; it is a reference, not something sent to employers.
- **Tailoring:** `tailor.py` flattens `MyCV/main.tex`, checks suitability against a job description, researches the target company (values, culture, hiring philosophy, CV guidance) via web search, generates a tailored single-file `cv.tex` variant, runs a recruiter/ATS review-and-revise pass, and compiles a PDF.
- **Company research (on by default):** the target employer is researched from the live web and a brief on its values, culture, evaluation philosophy (e.g. Arm's "10x mindset"), and CV/application guidance is injected into generation and review. It is cached under `company_research/`. Disable with `--no-company-research`.
- **Legacy:** root `cv.tex` is a fallback only; edit `MyCV/content/*` instead.

## Core principles

1. **Stop the scroll (6-second scan).** A recruiter scans for ~6 seconds along a fixed path: current title → company → tenure → first line of the most recent role → skills. Front-load so that scan lands the right message. The strongest, most relevant, most quantified content goes first — first section, first role, first bullet.
2. **ATS first.** ~77% of automated rejections are *content* (missing keywords, weak bullets); ~23% are formatting. Both must pass: real keywords from the job description in high-weight positions, and machine-readable formatting.
3. **Show, don't tell.** A quantified result proves a competency better than naming it. Never label bullets with abstract soft skills.
4. **Honesty is absolute.** Never invent experience, skills, metrics, dates, or scope. Reframe and select what is true; never fabricate. Education and Publications are never altered.

## Writing rules (bullets)

- **XYZ formula:** *Accomplished **X**, as measured by **Y**, by doing **Z**.* Lead with the outcome, prove it with a number, then state the method/tools. Use it for high-impact bullets; keep simpler bullets simple rather than forcing a fake metric.
- **Lead bullet = strongest, keyword-rich result.** ATS and humans weight the first bullet under each role most heavily.
- **One claim per sentence.** Precise academic verbs (ICLR/ICML/NeurIPS register). No clichés ("workhorse of", "passionate about", "results-driven", "responsible for"), no filler. If a clause does not advance the claim, cut it.
- **Quantify with scale, selectivity, and prestige**, not just percentages: units deployed, users served, assets covered, dataset size, latency, rank ("1 of 1,175"), competitive selection ("69 of 4,218"), recognised scope ("WMO-compliant"). Numbers that convey magnitude beat vague adjectives.
- **No em dashes (`---`) in prose.** Use the comma, colon, semicolon, or parentheses the sentence actually needs. En-dash separators in award/media lists and proper names (e.g. Gauss--Newton) are fine.
- **No soft-skill bullet labels** (no "Leadership skills:", "Execution history:"). Bold lead-ins are allowed only when they name a concrete deliverable/theme (e.g. "**MIRI firmware:**").
- **British English** spelling (optimise, modelling, characterise).
- **No first-person pronouns** in CV bullets; start with the verb.

## ATS / formatting rules

- Single-column, reverse-chronological, text-selectable PDF; standard font 10–12pt; dates right-aligned as MM/YYYY or "Mon YYYY".
- Standard section headings (Experience, Education, Skills, Projects, Publications).
- No tables, multi-column layouts, text boxes, images, logos, skill-rating bars, or contact details inside headers/footers. Simple horizontal rules are fine.
- Never use hidden/white/zero-opacity text — modern ATS auto-rejects it as fraud.
- Mirror the job description's exact terminology (only for skills the candidate genuinely has) in the summary, the skills section, and the first bullet of relevant roles.
- Keep ATS-extraction hardening in the template (`\pdfgentounicode=1`, T1 fontenc, lmodern).

## Tailoring rules (applied automatically per application)

1. **Positioning.** Tailor the headline under the name and the Summary to the specific role; tune the wording of job titles to match the target vocabulary (without misrepresenting the actual role).
2. **Company-values alignment.** When a company-research brief is supplied, tailor not only to the role but to *how that employer hires*: mirror the company's genuine values, culture, and evaluation philosophy (e.g. Arm's "10x mindset") in the Summary emphasis, section ordering, and the framing/vocabulary of bullets — but only where the candidate's real experience already evidences it. This governs framing, emphasis, ordering, tone, and word choice; it is never a licence to invent, exaggerate, or relabel skills. Never name the target company in the CV body.
3. **Recruiter match-score pass.** Before finalising, act as a senior recruiter *for that exact company*: give a match score out of 100, list the top missing keywords from the job description, and name the red flags a hiring manager would catch in under 10 seconds. Then revise to raise the score and remove the red flags — using only true content.
4. **ATS-filter + hiring-manager pass.** Re-read as an ATS parser and as a manager scanning 200 résumés: identify which sections would be skipped, and rewrite them to stop the scroll.
5. **Reorder sections by role type:**
   - *Industry ML/AI engineer:* Experience → Skills → Projects → Education → Certifications.
   - *Embedded/systems engineer:* Experience → Projects → Skills → Education.
   - *Research / PhD / academic:* Education → Research → Publications → Teaching → Skills.
6. **Select and exclude by relevance; compress hard.** Drop or condense weakly relevant entries; expand the directly relevant ones. Target ~2 pages for industry, 1.5–2 for academic. `tailor.py` enforces a hard page cap via `--max-pages` (default 2): it counts the compiled PDF's pages and compresses the CV until it fits.
7. **Red flags to remove:** unexplained gaps, duties listed without impact, dated/irrelevant tech, bullets over two lines, buzzword soup, inconsistent dates or verb tense, first-person voice.

## Terminology bank (use the precise current term — only when true)

- **LLM / GenAI:** RAG, vector search / similarity indexing (FAISS), fine-tuning (LoRA/QLoRA, PEFT, TRL, SFT), instruction-tuned models, prompt engineering tied to pipelines, multi-agent / tool-calling, evaluation harnesses, ablations, benchmarking.
- **ML methods:** graph neural networks (GNNs), reinforcement learning (PPO/A2C/TD3, reward design, Gymnasium environments), supervised / few-shot learning, distribution shift, drift detection, feature engineering.
- **MLOps / infra:** Docker, Kubernetes, REST APIs / microservices, Ray / DeepSpeed / CUDA, Triton / TorchServe, MLflow / Weights & Biases, Prometheus / Grafana, HPC / SLURM / A100 GPUs, AWS / GCP, CI/CD, reproducible pipelines.
- **Embedded / systems (Elijah's core domain):** RTOS, bare-metal firmware, non-blocking state machines, OTA updates, BLE / LoRa / GSM/GPRS, I2C / SPI / UART, sensor fusion, low-power / sleep scheduling, NOR-flash drivers, ROS, Gazebo, KiCad.
- **Strong verbs:** Built, Designed, Architected, Implemented, Developed, Shipped, Deployed, Optimised, Reduced, Cut, Led, Trained, Benchmarked, Hardened, Ported.

## Hard constraints for generated output

- Output valid, compilable LaTeX only — no markdown, no code fences, no commentary.
- Do not change structural commands or formatting macros (`\entrytitle`, `\sectiontitle`, geometry); change content, not presentation.
- Do not modify the Education section or the Publications/`\bibliography`/`\nocite{*}` block.
- Do not mention the target company name in the CV body.
- **Never introduce a skill, tool, language, framework, technology, or certification absent from the master CV — even to match a job-description keyword.** Never relabel work to borrow the job's vocabulary (embedded firmware is not "RTL design", "digital hardware design", or "hardware verification"; functional testing/simulation is not hardware "verification"). A missing keyword is fine; a misrepresented one is disqualifying and gets caught in interview.

## Honesty & gaps

- When the role wants a skill the candidate genuinely lacks, **leave it out of the CV and report it to the candidate as a recommendation instead.** `tailor.py` prints and saves a gap analysis per run (match score, missing keywords, concrete things to learn or buried strengths to surface). Fabrication is replaced by advice.
- A skill the candidate is genuinely acquiring may appear marked **"(currently learning)"** or **"(in progress)"** — only if that is true.
