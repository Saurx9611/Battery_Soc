# Rechargeable Battery Performance Modelling

## Project Overview
This project provides an automated workflow to estimate the open-circuit voltage (OCV) versus the state of charge (SOC) of Lithium-ion cells. It identifies equivalent-circuit model (ECM) parameters from relaxation data and implements a robust real-time SOC estimator. The primary goal is to achieve an SOC estimation error of 5% or less across various operating profiles by processing raw charge, discharge, and impedance laboratory data.

## Setup Details
1. Clone the repository and navigate to the project directory.
2. It is recommended to create and activate a Python virtual environment.
3. Install the required dependencies using pip.

```bash
pip install -r requirements.txt
```

## Requirements
* Python 3.12+
* pandas
* numpy
* scipy
* matplotlib
* scikit-learn
* pybamm

## Metadata
* **Course/Assignment:** ES60208
* **Input Data:** Experimental charge/discharge tests (e.g., 0.1C discharge, GITT 0.5C discharge data)
* **Deliverables:** OCV-SOC models, ECM parameter identification scripts, real-time EKF SOC estimator.

<details>
<summary>Detailed Solution and Methodology</summary>

### 1. Data Ingestion and Processing
The raw data comes in CSV format from laboratory battery tests containing time, voltage, current, charge capacity, and temperature measurements. I processed the data by renaming the columns to standardized constants and converting units where necessary (e.g., mA to A). I calculated the initial true State of Charge (SOC) through Coulomb counting, which tracks the integral of the current over time against the total measured cell capacity.

### 2. OCV-SOC Curve Extraction
I derived the Open Circuit Voltage (OCV) versus State of Charge (SOC) curve using two primary approaches:
* **Reference Data via PyBaMM:** I utilized the `pybamm` library alongside the "Chen2020" parameter set to extract the theoretical Open Circuit Potentials (OCP) for both the positive (NMC811) and negative (Graphite/Silicon composite) electrodes.
* **Experimental Data (Pseudo-OCV):** I processed a low-rate 0.1C discharge test. Because the applied current is very small, the measured terminal voltage closely approximates the true OCV. The capacity was mapped to an SOC ranging from 100% to 0%, providing a continuous lookup curve. I also extracted static OCV points dynamically from the relaxation periods of the Galvanostatic Intermittent Titration Technique (GITT) dataset.

### 3. ECM Parameter Identification
I modeled the battery using a 1RC Equivalent Circuit Model (ECM). The parameters — ohmic resistance (R0), polarization resistance (R1), and polarization capacitance (C1) — were identified dynamically from the GITT 0.5C discharge data:
* The algorithm parses the dataset to pinpoint relaxation periods where the current abruptly drops to zero.
* **R0 Calculation:** The instantaneous voltage drop immediately following the current cut-off was divided by the current magnitude to calculate the immediate ohmic resistance (R0).
* **R1 and C1 Calculation:** I applied a non-linear exponential curve fit using `scipy.optimize.curve_fit` to the voltage recovery curve during the long relaxation phases. By fitting the equation `V(t) = V_inf - b * exp(-t/tau)`, I extracted the time constant (`tau`). This time constant was then used alongside the voltage asymptote to compute R1 and C1.

### 4. Real-time SOC Estimator (Extended Kalman Filter)
To estimate the SOC accurately in real-time, I implemented an Extended Kalman Filter (EKF) class:
* **State Vector:** The EKF tracks the internal state of the battery, which includes the SOC and the polarization voltage (V1).
* **Prediction Step (Time Update):** The algorithm uses Coulomb counting and the discrete-time 1RC ECM to project the next SOC and V1 based on the measured current and the time elapsed.
* **Correction Step (Measurement Update):** The filter estimates the expected terminal voltage using the OCV-SOC lookup interpolator and the internal state. It then compares this estimated voltage against the actual sensor-measured voltage. The residual difference is multiplied by the dynamically computed Kalman Gain to correct the predicted SOC.
* **Evaluation:** When tested against the true Coulomb-counted SOC, the final EKF model demonstrated excellent tracking. It yielded an SOC Mean Absolute Error (MAE) of 0.53% and a Voltage Root Mean Square Error (RMSE) of 0.0041 V, successfully meeting the project's target accuracy of ≤ 5%.

</details>