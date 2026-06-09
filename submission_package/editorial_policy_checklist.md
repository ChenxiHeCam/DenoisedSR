# Nature Portfolio Editorial Policy Checklist — pre-filled answers

> Download the official PDF template at
> https://www.nature.com/documents/nr-editorial-policy-checklist.pdf
> and transcribe / paste these answers.

---

## Corresponding author: Chenxi He
## Last updated: 2026-06-10

---

# Editorial Policy Checklist (Nature Portfolio)

## Reporting

| Item | Status |
|---|---|
| Data deposition in a public repository compliant with Nature Portfolio policy | ✓ Yes — code and result artifacts on GitHub (https://github.com/ChenxiHeCam/DenoisedSR) and Zenodo (https://doi.org/10.5281/zenodo.20584889). |
| Code availability statement in manuscript | ✓ Yes |
| Data availability statement in manuscript | ✓ Yes |
| Persistent identifier (DOI) for code | ✓ Yes — 10.5281/zenodo.20584889 |
| Reporting Summary submitted with manuscript | ✓ Yes (separate document) |

## Statistics and reproducibility

| Item | Status |
|---|---|
| All results reproducible from released code | ✓ Yes |
| Random seeds reported | ✓ Yes — three random seeds per multi-seed experiment, listed explicitly in Methods |
| Sample sizes given | ✓ Yes — n stated for every figure and table |
| Sample sizes pre-specified | ✓ Yes — fixed at the standard benchmark sizes (e.g., AI-Feynman 118, SRSD 120) |
| Data exclusions reported | ✓ Yes — n_features filter ($2 \le n \le 9$) for AI-Feynman tasks documented in Methods |
| Tests not pre-specified are exploratory | Not applicable (no hypothesis tests) |

## Human/animal subjects, biological materials, clinical data

All sections of the Editorial Policy Checklist on human subjects, animal subjects, biological materials, clinical data, dual use research of concern, and field samples / specimens are **Not applicable** for this manuscript. This is a methodological machine-learning paper using publicly available physics benchmarks; no human or animal subjects, biological samples, or clinical data are involved.

## Competing interests

✓ Statement included in manuscript: "The authors declare no competing interests."

## Inclusion & ethics in global research

Not applicable. No fieldwork, no community/Indigenous data collaboration, no global-research ethics review required.

## Pre-registration

Not applicable (computational benchmark study; no clinical or biological trial pre-registration).

## Author contributions

✓ Statement included in manuscript identifying who designed the study, implemented code, performed experiments, analyzed data, supervised the project, and wrote / revised the manuscript.

---

# Summary

This manuscript reports a methodological contribution in machine learning for
symbolic regression of physical laws. It uses only publicly available
benchmarks (AI-Feynman, PMLB/SRBench, SRSD-Feynman, Strogatz, Nguyen). All
custom code, trained model weights, recorded result artifacts, and the
manuscript source are released on GitHub and archived on Zenodo with a
persistent DOI. There are no human or animal subjects, no biological
materials, no clinical data, and no dual-use considerations.
