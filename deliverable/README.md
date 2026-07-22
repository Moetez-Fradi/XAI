# EDBT 2027 short research paper (XQdrant)

**Track:** Research short paper (≤6 pages body + unlimited Artifacts/references).  
**Title prefix:** `[Short Paper]` (required by CFP).  
**Template:** official EDBT A4 / ACM `sigconf` (`edbt-macros.tex`, `balance=false`).

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Auxiliary TeX files (`*.aux`, `*.log`, `*.cut`, …) are gitignored; `main.pdf` is kept for convenience.

## Layout

| Path | Role |
|------|------|
| `main.tex` / `main.bib` | Paper source |
| `edbt-macros.tex` | Official EDBT proceedings macros |
| `figures/` | Canonical plots (M1/M2/M3/K1/C1/V1 labels) |
| `artifact-stub/` | README to push to the artifact GitHub stub |

## Before CMT submit

1. Confirm author / affiliation / ORCID on the PDF.
2. Populate the artifact repo linked in §Artifacts (or attach a CMT zip).
3. Verify body content ends by page 6 (Artifacts + refs may continue).

Research harness (paper → folders): [`../DB_systems/README.md`](../DB_systems/README.md).  
Mechanism design notes: [`../xqdrant_docs/`](../xqdrant_docs/).  
API / milestone scratch: [`../docs/`](../docs/).
