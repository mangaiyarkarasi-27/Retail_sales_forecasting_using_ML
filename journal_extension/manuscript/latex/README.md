# Journal Extension Manuscript ,  LaTeX Package

This directory contains the complete, production-grade LaTeX manuscript for the journal study:

> **Bridging Multi-Horizon Retail Demand Forecasting and Inventory Decisions via Rolling Forecast-Conditional Conformal Uncertainty Calibration**

This manuscript represents a **substantial methodological and empirical extension** of our preliminary IEEE conference baseline study (\emph{E-IDSS} \cite{e_idss_ieee}).

, 

## 1. Directory Structure

```
journal_extension/manuscript/latex/
├── main.tex                           # Master LaTeX manuscript document
├── references.bib                     # Complete BibTeX bibliography (29 verified citations)
├── tables.tex                         # Structured LaTeX tables (Tables 1–11)
├── figures.tex                        # Publication-quality figure specifications (Figures 1–9)
├── conference_extension_statement.tex # Formal declaration of extension from IEEE conference study
├── supplementary_material.tex         # Technical appendices, notation, feature table, & simulation algorithm
├── FINAL_MANUSCRIPT_AUDIT.md          # Comprehensive scientific and numerical audit report
└── README.md                          # Compilation guide and package documentation
```

, 

## 2. Compilation Instructions

### Standard Compilation via `latexmk` (Recommended)
```bash
latexmk -pdf main.tex
```

### Manual Two-Pass Compilation via `pdflatex` and `bibtex`
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Overleaf Import
1. Create a new blank project in [Overleaf](https://www.overleaf.com/).
2. Upload all files from `journal_extension/manuscript/latex/` (`main.tex`, `references.bib`, `tables.tex`, `figures.tex`, `conference_extension_statement.tex`, `supplementary_material.tex`).
3. Set the compiler to **pdfLaTeX** and set `main.tex` as the root document.
4. Click **Recompile**.

, 

## 3. Key Methodological & Numerical Certifications

- **100% Traceability:** Every numerical value matches frozen experiments **EXP-011 through EXP-020** and the **JOURNAL-005A** reconciliation.
- **Strict Disambiguation:** Differentiates **Portfolio Demand-Weighted Fill Rate** ($+10.64\text{ pp}$ at $L=1$) from **Mean Series-Level Paired Fill Rate** ($+10.36\text{ pp}$, 95% bootstrap CI $[10.19, 10.55]\text{ pp}$).
- **Distribution Shift Transparency:** Diagnoses nonstationary residual standard deviation expansion ($3.55\times$ to $4.30\times$, KS $p < 10^{-42}$) and reports empirical coverage breakdown ($17.90\%–31.98\%$) without over-claiming.
- **Leakage-Free Temporal Protocol:** Enforces decision-time minimum lag $\tau \ge 7$ days, passing 10/10 automated tests.
- **Economic Trade-Offs:** Documents cost savings at $L=1$ ($-10.83\%$, $-\$790,834$, $p = 6.65 \times 10^{-55}$) and maps the multi-day cost-preference threshold ($p \ge \$2.00$).
- **Zero Unsupported Claims:** 0 Category D claims across the entire manuscript.

, 

## 4. Target Journal Template Migration

The manuscript is formatted in standard, modular LaTeX (`\documentclass[11pt]{article}`), making it trivial to migrate to any journal-specific template:
- **Elsevier (`elsarticle.cls`):** Replace document class with `\documentclass[preprint,12pt]{elsarticle}`.
- **Springer Nature (`sn-jnl.cls` / `svjour3.cls`):** Migrate author/affiliation blocks to Springer format.
- **IEEE Transactions (`IEEEtran.cls`):** Replace document class with `\documentclass[journal]{IEEEtran}`.
- **INFORMS / Management Science:** Replace document class with `\documentclass[mnsc,blindrev]{informs3}`.
