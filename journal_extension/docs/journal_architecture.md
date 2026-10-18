# Journal Extension Architecture

This document maps out the system components, data flows, and chronological dependencies of the journal extension's leakage-free uncertainty-aware inventory replenishment framework, highlighting the isolation of the original conference baseline.

---

## 1. System Architecture Flow

The flowchart below traces the path of information from historical demand observations through point-forecasting, conformal calibration, and rolling inventory simulation.

```mermaid
graph TD
    %% Demand Data
    subgraph Input Data Layer
        A["Historical Sales (y_k, k ≤ t-1)"] --> B["Decision-Time Feature Generator"]
    end

    %% Forecasting Layer
    subgraph Leakage-Free Forecasting Layer
        B --> C["7 Direct Horizon XGBoost Models (M_1...M_7)"]
        C --> D["7-Day Forecast Vector (ŷ_{t+1}...ŷ_{t+7})"]
    end

    %% Calibration Layer
    subgraph Uncertainty Quantification Layer
        A & D --> E["Out-of-Sample Calibration Residuals (E_{r,L}, r ≤ t-L-1)"]
        E --> F["Rolling-Origin Conformal Calibration (W=100)"]
        F --> G["Conformal Quantile (q_{t,L})"]
    end

    %% Decision Layer
    subgraph Replenishment Decision Layer
        D --> H["Cumulative Lead-Time Forecast (D̂_{t,L})"]
        G --> I["Safety Stock (SS_t)"]
        H & I --> J["Reorder Point (s_t = D̂_{t,L} + q_{t,L})"]
        K["Inventory On-Hand (OH_t)"] & L["Inventory On-Order (OO_t)"] --> M["Inventory Position (IP_t = OH_t + OO_t)"]
        M & J --> N["Replenishment Decision (IP_t < s_t?)"]
    end

    %% Simulation Layer
    subgraph Inventory Simulator
        N --> O["Place Order of size Q (EOQ)"]
        O --> P["Arrives at t+L (Arriving Deliveries)"]
        Q["Realized Today's Demand (y_t)"] --> R["Demand Fulfillment (Sales & Shortage)"]
        R --> S["Calculate carrying/shortage/ordering costs"]
        S --> T["Update Physical On-Hand (OH_{t+1})"]
    end

    %% Styling
    classDef input fill:#f9f,stroke:#333,stroke-width:2px;
    classDef fore fill:#bbf,stroke:#333,stroke-width:2px;
    classDef cal fill:#bfb,stroke:#333,stroke-width:2px;
    classDef dec fill:#ffb,stroke:#333,stroke-width:2px;
    classDef sim fill:#fbf,stroke:#333,stroke-width:2px;
    class A,Q input;
    class C,D,H fore;
    class E,F,G cal;
    class J,M,N dec;
    class O,P,R,S,T sim;
```

---

## 2. Chronological Pipeline Isolation

To preserve scientific integrity, the conference pipeline and the new journal pipelines are completely decoupled:

### **Original Conference Pipeline (Frozen)**
```
Raw Data → Lag/Rolling Feature Engine (Shift-1) → Single Global XGBoost → Pre-computed Point Forecasts
```
* **Status**: Locked. No parameters, models, or data inputs are modified.

### **Journal Extension Pipeline (Proposed)**
```
Raw Data → Direct Multi-Horizon Feature Engine (Shift-h) → 7 Decoupled XGBoost Models (M_1...M_7) 
         → Real-Time Point Forecast Vector (ŷ_{t+1}...ŷ_{t+L})
         → Rolling Conformal Quantiles (q_{t,L}, r ≤ t-L-1)
         → Inventory Position s_t
         → Daily (s, Q) Replenishment Simulator
```
* **Status**: Methodology audit completed. No data leakage or temporal indexing errors permitted.
