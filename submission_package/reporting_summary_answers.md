# Nature Portfolio Reporting Summary — pre-filled answers

> Download the official PDF template at
> https://www.nature.com/documents/nr-reporting-summary.pdf
> and transcribe / paste these answers. Sections below match the PDF's section headings exactly.

---

## Corresponding author(s): Chenxi He
## Last updated by author(s): 2026-06-10

---

# Statistics

For all statistical analyses, confirm that the following items are present in the figure legend, table legend, main text, or Methods section.

- **The exact sample size (n) for each experimental group/condition, given as a discrete number and unit of measurement**
  ✓ Confirmed. n is given in every figure caption, body sentence, and Supplementary Table (e.g., n=118 Feynman, n=120 SRSD, n=30 in PySR headline, n=3 seeds, etc.).

- **A statement on whether measurements were taken from distinct samples or whether the same sample was measured repeatedly**
  ✓ Distinct samples. Each random seed defines a fresh sampling of (X, y) and distractor columns; seeds are independent.

- **The statistical test(s) used AND whether they are one- or two-sided**
  Not applicable. We report descriptive statistics (mean ± s.d. over random seeds) and exact-recovery rates; no hypothesis tests with p-values are performed.

- **A description of all covariates tested**
  Not applicable.

- **A description of any assumptions or corrections, such as tests of normality and adjustment for multiple comparisons**
  Not applicable.

- **A full description of the statistical parameters including central tendency (e.g. means) or other basic estimates (e.g. regression coefficient) AND variation (e.g. standard deviation) or associated estimates of uncertainty (e.g. confidence intervals)**
  Mean and sample standard deviation (Bessel-corrected, ddof=1) across random seeds, reported as mean ± s.d. throughout.

- **For null hypothesis testing, the test statistic (e.g. F, t, r) with confidence intervals, effect sizes, degrees of freedom and P value noted. Give P values as exact values whenever suitable.**
  Not applicable.

- **For Bayesian analysis, information on the choice of priors and Markov chain Monte Carlo settings**
  Not applicable.

- **For hierarchical and complex designs, identification of the appropriate level for tests and full reporting of outcomes**
  Not applicable.

- **Estimates of effect sizes (e.g. Cohen's d, Pearson's r), indicating how they were calculated**
  Reported as percentage-point differences in exact recovery and as Pearson r where applicable (e.g., r=-0.69 between SRSD dummy count and DenoisedSR precision).

*Our web collection on statistics for biologists contains articles on many of the points above.*

---

# Software and code

## Policy information about availability of computer code

- **Data collection**
  No proprietary or special-purpose data-collection software was used. External benchmark datasets are obtained from their authors' public releases:
  - AI-Feynman: https://space.mit.edu/home/tegmark/aifeynman.html
  - PMLB / SRBench: https://github.com/EpistasisLab/pmlb and https://github.com/cavalab/srbench
  - SRSD-Feynman: https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_easy (and `_medium`, `_hard` variants)
  - Strogatz / Nguyen: implemented from textbook references (formulas listed in `frontend/eval/eval_external_3way.py`).

- **Data analysis**
  All analyses use Python 3.11 with the following major packages:
  - `numpy >= 1.24`, `scipy >= 1.11`, `scikit-learn >= 1.3`, `sympy >= 1.12`, `pandas >= 2.0`, `matplotlib >= 3.7`, `joblib >= 1.3`
  - Deep-learning: `torch >= 2.0` (with CUDA 12.8 wheel), `torch_geometric >= 2.4`
  - Symbolic-regression backends: `pysr >= 0.18` (Julia backend via juliacall), `gplearn >= 0.4.2`, `psrn` (PSE; Ruan et al. 2026)
  - Benchmark loaders: `pmlb >= 1.0`, `datasets >= 2.18`, `huggingface_hub >= 0.20`

  All custom analysis code is publicly released at
  https://github.com/ChenxiHeCam/DenoisedSR (commit f75b29f) and archived at
  https://doi.org/10.5281/zenodo.20584889. The Zenodo archive bundles the two
  large random-forest weight files that exceed GitHub's per-file size limit.

For manuscripts utilizing custom algorithms or software that are central to the research but not yet described in published literature, software must be made available to editors and reviewers. We strongly encourage code deposition in a community repository (e.g. GitHub). See the Nature Portfolio guidelines for submitting code & software for further information.

---

# Data

## Policy information about availability of data

All manuscripts must include a **data availability statement**. This statement should provide the following information, where applicable:
- Accession codes, unique identifiers, or web links for publicly available datasets
- A description of any restrictions on data availability
- For clinical datasets or third party data, please ensure that the statement adheres to our policy

- **External benchmarks are public, with sources as listed above** (AI-Feynman, PMLB, SRBench, SRSD-Feynman dummy variants on Hugging Face).
- **Internally-curated physical-law records** used for training are materialised from public formula and range information; we do not redistribute the raw records because they include intermediate filtering artifacts unrelated to the published claims.
- **Result artifacts** (every JSON file backing every figure and table in the paper) are released alongside code at the Zenodo archive above.

---

# Field-specific reporting

Please select the one below that is the best fit for your research. If you are not sure, read the appropriate sections before making your selection.

- [ ] Life sciences
- [ ] Behavioural & social sciences
- [x] **Ecological, evolutionary & environmental sciences ☐** (none of the above; this is methodological machine learning for physics)

*Note: select "Ecological, evolutionary & environmental sciences" only if applicable. Otherwise pick the closest match. For an ML-for-physics manuscript, none of the three Life / Behavioural / Ecological categories is the natural fit — see Nature Portfolio guidance for "physical sciences and engineering" submissions, where Sections 1–3 above suffice.*

---

# Reporting for specific materials, systems and methods

Not applicable (no biological materials, antibodies, eukaryotic cells, palaeontology, animals, human subjects, or clinical data are involved). All "Materials & experimental systems" and "Methods" sub-sections of the Reporting Summary template should be marked **Not applicable** for this manuscript.
