# Automated Workflow for Li-Ion Cell Modeling 🔋

**Author:** NamoFans | IIT Kharagpur  
**Course:** ES60208 - Rechargeable Battery Performance Modelling  

## Overview
This repository contains a comprehensive computational workflow designed to estimate lithium-ion battery performance. The pipeline automatically extracts the thermodynamic Open-Circuit Voltage (OCV) versus State of Charge (SOC) relationship, identifies time-varying Equivalent-Circuit Model (ECM) parameters, and deploys an Extended Kalman Filter (EKF) for real-time state estimation. 

[cite_start]The primary objective is to maintain a rigorous SOC estimation error of $\le 5\%$ across highly dynamic operating profiles[cite: 656].

## Repository Structure
\`\`\`text
battery-soc-modelling/
├── data/
│   ├── raw/                  # Store raw .mpr and NASA datasets here
│   └── processed/            # Automated .parquet columnar outputs
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py     # Feature engineering and dataset parsing
│   ├── ocv_extractor.py      # IC-GITT and pseudo-OCV mathematical alignment
│   ├── ecm_fitter.py         # Boundary-constrained 1-RC parameter optimization 
│   └── soc_estimator.py      # Extended Kalman Filter and Evaluation Metrics
├── notebooks/                # Jupyter notebooks for EDA and validation
├── requirements.txt          # Python dependencies
├── run_pipeline.py           # Master execution script
└── README.md
\`\`\`

## Setup & Installation
This project requires Python 3.9+. To ensure a clean workspace and avoid system package conflicts (especially if you are running this on Arch Linux or similar distributions), it is highly recommended to use a virtual environment.

1. **Clone the repository:**
   
   git clone [https://github.com/Saurx9611/Battery_Soc.git](https://github.com/Saurx9611/Battery_Soc)
   cd battery-soc-modelling
   

2. **Create and activate a virtual environment:**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   \`\`\`

3. **Install dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
   *(Required libraries: `pandas`, `numpy`, `scipy`, `lmfit`, `prog_models`, `galvani`, `pyarrow`)*

## Data Management
Because battery cycle data is massive, raw data is **not** tracked in this repository. 

1. [cite_start]Download the **Randomized Battery Dataset** from the NASA repository[cite: 659]. (Note: `src/data_ingestion.py` can fetch this automatically via API).
3. Do not manually convert the data. The pipeline will automatically process these files, apply a 500-second rolling window and 5-point stencil derivative, and save them as highly optimized `.parquet` files in `data/processed/` to prevent I/O bottlenecks.

## Execution
To run the end-to-end pipeline—from simulated data ingestion to final SOC error metric evaluation—simply execute the master script:

\`\`\`bash
python run_pipeline.py
\`\`\`

The console will output the identified $R_0$, $R_1$, and $C_1$ parameters, the deployable OCV polynomial coefficients, and the final RMSE and MAE tracking metrics for the Extended Kalman Filter. 

## Future Extensions
While the current EKF meets the $\le 5\%$ error threshold utilizing the strictly bounded `lmfit` optimization, future iterations of this codebase could integrate an XGBoost regression model to dynamically update the physical ECM parameters in real-time as the battery ages, offering even greater robustness under extreme thermal variations.
