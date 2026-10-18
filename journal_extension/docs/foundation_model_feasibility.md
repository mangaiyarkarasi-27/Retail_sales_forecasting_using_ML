# Foundation Model Feasibility Study

This document details the feasibility assessment of integrating deep-learning-based time-series foundation models (such as the Chronos or TimesFM families) into the retail sales forecasting pipeline under the current project constraints.

---

## 1. Candidate Models Evaluated

### **A. Chronos Family (Amazon)**
* **Architecture**: Language model (T5-based) trained on synthetic and real-world time series, treating values as discrete tokens.
* **Access / License**: Open weights, Apache-2.0 License.
* **Input Representation**: Univariate sequence of numerical values.
* **Forecast Horizon**: Supports arbitrary forecasting horizons via auto-regressive decoding.
* **Libraries Required**: `torch`, `transformers`, `accelerate`.

### **B. TimesFM Family (Google)**
* **Architecture**: Decoder-only transformer trained on large-scale time-series corpora (up to 100 billion points).
* **Access / License**: Open weights, Apache-2.0 License.
* **Input Representation**: Univariate sequence of values.
* **Libraries Required**: `jax`, `torch`, `huggingface_hub`.

---

## 2. Resource & Environment Constraints

An audit of the virtual environment and execution environment reveals major constraints:
1. **No PyTorch or JAX Installed**: The virtual environment has standard scikit-learn, statsmodels, and xgboost, but lacks deep-learning runtimes (`torch`, `torchvision`, `torchaudio`, `jax`, `jaxlib`).
2. **Disk Space & Package Size Limits**: Installing PyTorch on Windows requires downloading wheels of approximately **2.6 GB** (plus Hugging Face transformers, accelerate, and dependencies), which exceeds the standard disk quotas of the VM.
3. **Hardware Constraints (CPU-only)**: The current system is running in a CPU-only environment. A single T5-based model inference (e.g., Chronos-T5-Base) takes roughly **100ms to 500ms** per time series.
4. **Execution Scalability**: For 500 store-item series evaluated across 5 validation folds ($500 \times 5 = 2,500$ series forecast rollouts), CPU inference would require:
   $$2,500 \text{ series} \times 0.3 \text{ seconds/series} = 750 \text{ seconds (12.5 minutes)}$$
   This excludes the overhead of sliding-window evaluation and data formatting, which introduces high risk of execution timeout or memory exhaustion.

---

## 3. Feasibility Conclusion

* **Status**: **UNFEASIBLE**
* **Rationale**: Due to the lack of PyTorch/JAX libraries, CPU-only hardware limits, package size boundaries, and potential process timeouts, deep-learning time-series foundation models cannot be executed reliably in the current workspace.
* **Recommendation**: Rather than fabricating results or risking VM crash, the project will officially record the foundation model as unfeasible. To fulfill the Multi-Model Forecasting Expert requirements of the journal, we will use **Random Forest** (as the alternative ML expert) and **ARIMA/AutoReg** (as the statistical baseline expert) alongside the conference **XGBoost** regressor, comparing their performance across demand profiles.
