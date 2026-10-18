"""
create_intro_citation_audit.py
------------------------------
Generates INTRODUCTION_CITATION_AUDIT.csv for JOURNAL-002.
"""

import pandas as pd
from pathlib import Path

citation_records = [
    {
        "Paragraph": "Para 1 (Context & Motivation)",
        "Claim": "Dynamic retail environment and demand forecasting importance",
        "Citation": "Tiwari & Prasad (2025)",
        "Source Type": "Conference Proceedings (AIMV 2025)",
        "Verified?": "YES (In Conference Bib)",
        "Citation Issue": "Verify proceeding page range"
    },
    {
        "Paragraph": "Para 1 (Context & Motivation)",
        "Claim": "Store-item demand heterogeneity, seasonality, and multi-channel patterns",
        "Citation": "Mitra et al. (2022); Venkiteela (2025)",
        "Source Type": "Peer-Reviewed Journals (OR Forum; IJCESEN)",
        "Verified?": "YES",
        "Citation Issue": "None"
    },
    {
        "Paragraph": "Para 1 (Context & Motivation)",
        "Claim": "Sales and Operations Planning (S&OP) dependence on forecasting",
        "Citation": "Mustapha & Sithole (2025)",
        "Source Type": "Peer-Reviewed Journal (AJSAS)",
        "Verified?": "YES",
        "Citation Issue": "None"
    },
    {
        "Paragraph": "Para 1 (Context & Motivation)",
        "Claim": "Operational failures from forecast errors: stockouts vs overstocking",
        "Citation": "Spanos et al. (2022); Malik et al. (2024)",
        "Source Type": "Peer-Reviewed Journals (Algorithms; JCBI)",
        "Verified?": "YES",
        "Citation Issue": "None"
    },
    {
        "Paragraph": "Para 2 (Point Forecasts & Decision Risk)",
        "Claim": "Point forecasts do not communicate dispersion or tail risk; proper scoring rules",
        "Citation": "Gneiting & Raftery (2007)",
        "Source Type": "Peer-Reviewed Journal (JASA)",
        "Verified?": "YES",
        "Citation Issue": "None"
    },
    {
        "Paragraph": "Para 2 (Point Forecasts & Decision Risk)",
        "Claim": "Asymmetry of operational costs in retail inventory replenishment",
        "Citation": "Ban & Rudin (2019); Bertsimas & Kallus (2020)",
        "Source Type": "Peer-Reviewed Journals (Operations Research)",
        "Verified?": "YES",
        "Citation Issue": "None"
    },
    {
        "Paragraph": "Para 3 (Machine Learning in Retail)",
        "Claim": "Evolution from univariate linear statistical models (ARIMA, ETS) to ML",
        "Citation": "Mustapha & Sithole (2025); Yeesin & Kajornkasirat (2024)",
        "Source Type": "Peer-Reviewed Journal / Conference (AJSAS; IIPEM)",
        "Verified?": "YES",
        "Citation Issue": "None"
    },
    {
        "Paragraph": "Para 3 (Machine Learning in Retail)",
        "Claim": "GBDT (XGBoost/LightGBM) superiority on structured tabular demand series",
        "Citation": "Makridakis et al. (2022); Chen & Guestrin (2016); Hobor et al. (2026)",
        "Source Type": "Peer-Reviewed Journals / Conferences (IJF; ACM KDD; CROERR)",
        "Verified?": "YES",
        "Citation Issue": "Verify Hobor et al. volume/page details"
    },
    {
        "Paragraph": "Para 3 (Machine Learning in Retail)",
        "Claim": "Global feature pooling across store-item categories",
        "Citation": "Al Ghurabli et al. (2025)",
        "Source Type": "Conference Proceedings (ICBITI 2025)",
        "Verified?": "YES",
        "Citation Issue": "None"
    },
    {
        "Paragraph": "Para 3 (Machine Learning in Retail)",
        "Claim": "Multi-step recursive error compounding and look-ahead feature leakage risk",
        "Citation": "Tiwari et al. (2026)",
        "Source Type": "Peer-Reviewed Journal (Mathematics)",
        "Verified?": "YES",
        "Citation Issue": "None"
    },
    {
        "Paragraph": "Para 4 (Explainability & Conference Baseline)",
        "Claim": "Explainable Intelligent Decision Support System baseline framework",
        "Citation": "IEEE E-IDSS Conference Paper (2026)",
        "Source Type": "Conference Baseline (Project Base)",
        "Verified?": "YES (Base Study)",
        "Citation Issue": "Prior conference paper by authors"
    },
    {
        "Paragraph": "Para 4 (Explainability & Conference Baseline)",
        "Claim": "TreeSHAP game-theoretic local and global feature attribution",
        "Citation": "Lundberg & Lee (2017)",
        "Source Type": "Peer-Reviewed Conference (NeurIPS 2017)",
        "Verified?": "YES",
        "Citation Issue": "None"
    },
    {
        "Paragraph": "Para 4 (Explainability & Conference Baseline)",
        "Claim": "XAI resolving the explanation gap and reducing human override rates",
        "Citation": "Lin, Lau, & Hu (2026); Narlawar (2025); Chang & Bau (2025)",
        "Source Type": "Peer-Reviewed Journals (AI Review; WJARR; JEBDE)",
        "Verified?": "YES",
        "Citation Issue": "Verify Lin et al. page numbers"
    },
    {
        "Paragraph": "Para 5 (Probabilistic Forecasting & Conformal UQ)",
        "Claim": "Distribution-free conformal prediction framework and finite-sample guarantees",
        "Citation": "Vovk, Gammerman, & Shafer (2005); Angelopoulos & Bates (2021)",
        "Source Type": "Foundational Book / Peer-Reviewed Review",
        "Verified?": "YES",
        "Citation Issue": "None"
    },
    {
        "Paragraph": "Para 5 (Probabilistic Forecasting & Conformal UQ)",
        "Claim": "Parametric Gaussian safety stock assumptions and limitations in retail",
        "Citation": "Silver, Pyke, & Peterson (1998)",
        "Source Type": "Classic Textbook (John Wiley & Sons)",
        "Verified?": "YES",
        "Citation Issue": "None"
    },
    {
        "Paragraph": "Para 5 (Probabilistic Forecasting & Conformal UQ)",
        "Claim": "Sequential time-series conformal prediction and rolling residual pooling",
        "Citation": "Stankeviciute, Alaa, & van der Schaar (2021); Huber et al. (2019)",
        "Source Type": "Peer-Reviewed Conferences / Journals (ICLR 2021; EJOR 2019)",
        "Verified?": "YES",
        "Citation Issue": "None"
    },
    {
        "Paragraph": "Para 5 & 6 (Distribution Shift)",
        "Claim": "Conformal coverage degradation when exchangeability is violated under shift",
        "Citation": "Barber, Candes, Ramdas, & Tibshirani (2023); Tibshirani et al. (2019)",
        "Source Type": "Peer-Reviewed Journals / Conferences (Annals of Statistics; NeurIPS)",
        "Verified?": "YES",
        "Citation Issue": "None"
    },
    {
        "Paragraph": "Para 7 (Inventory Decision Problem)",
        "Claim": "Supply chain integration of forecasting and continuous-review ROP control",
        "Citation": "Syntetos et al. (2016); Axsäter (2015); Snyder & Shen (2019)",
        "Source Type": "Peer-Reviewed Journals / Textbooks (IJPR; Springer; Wiley)",
        "Verified?": "YES",
        "Citation Issue": "None"
    }
]

df = pd.DataFrame(citation_records)
output_file = Path("journal_extension/manuscript/drafts/INTRODUCTION_CITATION_AUDIT.csv")
df.to_csv(output_file, index=False)
print(f"Successfully wrote {output_file} with {len(df)} citation verification records.")
