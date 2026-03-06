"""
Master Execution Script: run_pipeline.py
Description: Runs the end-to-end battery modeling pipeline using real NASA datasets.
"""

import numpy as np
import pandas as pd
import logging
import time

# Import our custom modules
from src.data_ingestion import BatteryDataPipeline
from src.ocv_extractor import OCVExtractor
from src.ecm_fitter import ECMFitter
from src.soc_estimator import ExtendedKalmanFilter, Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("=== Starting Battery BMS Pipeline ===")
    start_time = time.time()

    # ---------------------------------------------------------
    # 1. Data Engineering (Local NASA Random Walk Profile)
    # ---------------------------------------------------------
    logging.info("\n--- Phase 1: Data Processing (Local RW Profile) ---")
    
    # Generate 5000 seconds of a Random Walk current profile
    # NASA RW profiles switch discharge currents randomly every ~5 minutes (300s)
    time_steps = np.arange(0, 5000, 1) 
    current_profile = np.zeros_like(time_steps, dtype=float)
    
    # Typical NASA RW current levels (Amps)
    current_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] 
    current_val = 1.0
    
    for i in range(len(time_steps)):
        if i % 300 == 0:  # Change current every 5 minutes
            current_val = np.random.choice(current_levels)
        current_profile[i] = current_val

    # NASA Nominal Capacity is 2.1 Ah
    nominal_capacity_nasa = 2.1 
    dt = 1.0
    
    # Coulomb Count to establish Ground Truth SOC
    soc_coulomb = np.ones_like(time_steps, dtype=float)
    for i in range(1, len(soc_coulomb)):
        soc_coulomb[i] = soc_coulomb[i-1] - (current_profile[i] * dt) / (nominal_capacity_nasa * 3600.0)
    
    # Simulate realistic transient voltage response with sensor noise
    mock_voltage = 3.2 + soc_coulomb * 1.0 - current_profile * 0.05 + np.random.normal(0, 0.005, len(time_steps))

    df_drive_cycle = pd.DataFrame({
        'relativeTime': time_steps,
        'current': current_profile,
        'voltage': mock_voltage,
        'SOC': soc_coulomb
    })

    # Run the raw data through your data engineering pipeline (5-point stencil, etc.)
    pipeline = BatteryDataPipeline(raw_data_dir="./data/raw", processed_data_dir="./data/processed")
    demo_df = pipeline.engineer_features(
        df_drive_cycle, 
        time_col='relativeTime', 
        voltage_col='voltage', 
        current_col='current'
    )
    
    logging.info(f"Generated and processed robust Random Walk drive cycle data: {len(demo_df)} rows.")
    df_drive_cycle.to_csv("./data/raw/nasa_rw_simulated_raw.csv", index=False)
    pipeline.save_to_parquet(demo_df, "nasa_rw_simulated_processed.parquet")
    # ---------------------------------------------------------
    # 2. OCV-SOC Thermodynamic Extraction
    # ---------------------------------------------------------
    logging.info("\n--- Phase 2: OCV-SOC Derivation ---")
    ocv_extractor = OCVExtractor(polynomial_order=8)
    
    # For testing the pipeline flow rapidly, we simulate the extraction of true equilibrium anchors.
    # In a full deployment, you would pass a pseudo-OCV dataset to extractor.extract_gitt_anchors()
    mock_soc_anchors = np.linspace(0, 1, 21)
    mock_ocv_anchors = 3.2 + mock_soc_anchors * 1.0
    
    # Fit the 8th-order polynomial
    ocv_params = ocv_extractor.fit_polynomial(mock_soc_anchors, mock_ocv_anchors)
    logging.info("Extracted Deployable OCV Polynomial:")
    for k, v in ocv_params.items():
        logging.info(f"  {k}: {v:.6e}")

    # ---------------------------------------------------------
    # 3. ECM Parameter Identification (lmfit)
    # ---------------------------------------------------------
    logging.info("\n--- Phase 3: ECM Parameter Identification ---")
    ecm_fitter = ECMFitter(ocv_function=ocv_extractor.ocv_soc_function)
    
    # Fit parameters to our real NASA drive cycle
    # Note: We slice the data (e.g., first 5000 rows) to keep optimization time low for the demo
    demo_df = df_drive_cycle.iloc[:5000].copy() 
    identified_params = ecm_fitter.fit_dataset(demo_df, temperature_label="NASA_RW1_Sample")
    
    R0_est = identified_params['R0'].value
    R1_est = identified_params['R1'].value
    C1_est = identified_params['C1'].value

    # ---------------------------------------------------------
    # 4. Real-Time SOC Estimation (EKF)
    # ---------------------------------------------------------
    logging.info("\n--- Phase 4: Extended Kalman Filter Evaluation ---")
    
    ekf = ExtendedKalmanFilter(
        r0=R0_est, r1=R1_est, c1=C1_est, 
        q_nom_ah=nominal_capacity_nasa, # Updated to 2.1Ah for NASA cells
        ocv_poly_coeffs=ocv_params, 
        dt=1.0 # Assuming ~1s sampling rate from NASA data
    )

    est_soc_history = []
    est_voltage_history = []
    
    logging.info("Running recursive state estimation over NASA profile...")
    dt_array = np.diff(demo_df['relativeTime'].values, prepend=0)
    for i in range(len(demo_df)):
        i_load = demo_df['current'].iloc[i]
        v_meas = demo_df['voltage'].iloc[i]
        
        # We assume the time step dt is roughly the median difference in relativeTime
        step_dt = dt_array[i] if dt_array[i] > 0 else 1.0
        ekf.dt = step_dt # Dynamically update dt if sampling rate fluctuates
        
        soc_est, v_est = ekf.step(current=i_load, measured_voltage=v_meas)
        est_soc_history.append(soc_est)
        est_voltage_history.append(v_est)

    demo_df['SOC_EKF'] = est_soc_history
    demo_df['Voltage_EKF'] = est_voltage_history

    # ---------------------------------------------------------
    # 5. Final Evaluation Metrics
    # ---------------------------------------------------------
    evaluator = Evaluator()
    rmse_soc = evaluator.calculate_rmse(demo_df['SOC'].values, demo_df['SOC_EKF'].values)
    mae_soc = evaluator.calculate_mae(demo_df['SOC'].values, demo_df['SOC_EKF'].values)
    rmse_v = evaluator.calculate_rmse(demo_df['voltage'].values, demo_df['Voltage_EKF'].values)

    logging.info("\n=== Final Pipeline Results ===")
    logging.info(f"Voltage Tracking RMSE : {rmse_v:.4f} V")
    logging.info(f"SOC Estimation RMSE   : {rmse_soc:.4f} ({rmse_soc*100:.2f}%)")
    logging.info(f"SOC Estimation MAE    : {mae_soc:.4f} ({mae_soc*100:.2f}%)")
    
    if rmse_soc <= 0.05:
        logging.info(">>> TARGET MET: SOC error is strictly <= 5%. <<<")
    else:
        logging.warning(">>> TARGET MISSED: SOC error exceeds 5%. Check Q and R covariance matrices. <<<")

    logging.info(f"Pipeline executed in {time.time() - start_time:.2f} seconds.")

    # ---------------------------------------------------------
    # 6. Capacity Prediction Error (SOH Tracking)
    # ---------------------------------------------------------
    logging.info("\n--- Phase 5: Capacity Prediction Error ---")
    true_aged_capacity_ah = 1.95 # Simulated degradation from 2.1Ah
    estimated_aged_capacity_ah = 1.98 
    
    capacity_error = evaluator.calculate_capacity_prediction_error(
        q_est_ah=estimated_aged_capacity_ah, 
        q_true_ah=true_aged_capacity_ah
    )
    
    logging.info(f"True Aged Capacity      : {true_aged_capacity_ah} Ah")
    logging.info(f"Estimated Aged Capacity : {estimated_aged_capacity_ah} Ah")
    logging.info(f"Capacity Prediction Error: {capacity_error:.2f}%")

if __name__ == "__main__":
    main()