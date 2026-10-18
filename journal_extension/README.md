# Journal Extension: Decision-Aware Adaptive Multi-Model Forecasting Framework

This directory houses the codebase, configurations, experiments, and documentation for the journal version of the retail sales forecasting project.

## Key Research Components:
1. **Demand Profiling & Regime Detection**: Classifies products/stores into distinct demand types (smooth, erratic, intermittent, lumpy) and detects state shifts.
2. **Forecasting Experts**: Integrates existing XGBoost and baseline experts with a modern foundation model forecasting expert.
3. **Decision-Aware Adaptive Model Selection**: Routes predictions dynamically to the best expert based on demand regimes and decision-aware costs.
4. **Uncertainty Quantification**: Implements conformal prediction and prediction intervals for risk assessment.
5. **Risk-Aware Inventory Policy Simulation**: Evaluates safety stock, reorder points, and order quantities via simulation under various policies.
