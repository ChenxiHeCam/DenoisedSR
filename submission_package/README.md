# Communications Physics submission package
DenoisedSR: Learned variable-support priors accelerate symbolic regression of physical laws

## Files for submission

| File | Description | Notes |
|---|---|---|
| `main.pdf` | Manuscript (19 pages, formatted) | Compiled from `main.tex` |
| `main.tex` | Manuscript LaTeX source | Uses `references.bib`, `figures/*.{pdf,png}` |
| `supplementary.pdf` | Supplementary Information (4 pages) | 7 SI tables + 1 SI figure |
| `supplementary.tex` | SI LaTeX source | |
| `references.bib` | Bibliography (40+ entries) | `naturemag.bst` style |
| `figures/*.{pdf,png}` | All figures (Okabe-Ito colour-blind-safe, 300 dpi) | 12 main + 1 SI figure |
| `cover_letter.pdf` | Cover letter (2 pages) | Includes suggested reviewers; **author should edit reviewer list before submission** |
| `cover_letter.tex` | Cover letter source | |
| `reporting_summary_answers.md` | Pre-filled Nature Reporting Summary | Copy into the official PDF: https://www.nature.com/documents/nr-reporting-summary.pdf |
| `editorial_policy_checklist.md` | Pre-filled Editorial Policy Checklist | Copy into the official PDF: https://www.nature.com/documents/nr-editorial-policy-checklist.pdf |

## Submission system

https://mts-commsphys.nature.com/cgi-bin/main.plex
- Author needs to register/log in with ORCID or email
- Upload: main.pdf + supplementary.pdf + cover_letter.pdf + reporting_summary.pdf + editorial_policy_checklist.pdf
- Indicate suggested reviewers from cover letter

## Public release

- Code + result artifacts: https://github.com/ChenxiHeCam/DenoisedSR
- Persistent archive: https://doi.org/10.5281/zenodo.20584889

## Before submitting

1. **Edit suggested reviewers** in `cover_letter.tex` (current list is a placeholder of well-known SR researchers)
2. **Re-compile cover letter** if you change anything: `pdflatex cover_letter.tex`
3. **Transcribe** `reporting_summary_answers.md` into the official Nature Reporting Summary PDF
4. **Transcribe** `editorial_policy_checklist.md` into the official Editorial Policy Checklist PDF
5. Done.
