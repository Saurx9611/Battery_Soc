"""
Master Execution Script: run_pipeline.py
Description: Runs the end-to-end battery modeling pipeline.
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
    # 1. Data Engineering (Simulated for immediate testing)
    # ---------------------------------------------------------
    logging.info("\n--- Phase 1: Data Processing ---")
    # In a real run, you would use: pipeline.ingest_zenodo_mpr("Expt4_file.mpr")
    # Here, we generate a synthetic drive cycle to test the architecture
    time_steps = np.arange(0, 1000, 1)
    mock_current = np.where(time_steps % 100 < 50, 2.0, -0.5) # Dynamic discharge/charge
    mock_soc = np.linspace(1.0, 0.8, len(time_steps))
    mock_voltage = 3.2 + mock_soc * 1.0 - mock_current * 0.05 + np.random.normal(0, 0.005, len(time_steps))

    df_drive_cycle = pd.DataFrame({
        'relativeTime': time_steps,
        'current': mock_current,
        'voltage': mock_voltage,
        'SOC': mock_soc
    })
    logging.info(f"Loaded drive cycle data: {len(df_drive_cycle)} rows.")

    # ---------------------------------------------------------
    # 2. OCV-SOC Thermodynamic Extraction
    # ---------------------------------------------------------
    logging.info("\n--- Phase 2: OCV-SOC Derivation ---")
    ocv_extractor = OCVExtractor(polynomial_order=8)
    
    # Simulating the extraction of true equilibrium anchors
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
    
    # Fit parameters to our drive cycle
    identified_params = ecm_fitter.fit_dataset(df_drive_cycle, temperature_label="Simulated_WLTP")
    
    R0_est = identified_params['R0'].value
    R1_est = identified_params['R1'].value
    C1_est = identified_params['C1'].value

    # ---------------------------------------------------------
    # 4. Real-Time SOC Estimation (EKF)
    # ---------------------------------------------------------
    logging.info("\n--- Phase 4: Extended Kalman Filter Evaluation ---")
    
    ekf = ExtendedKalmanFilter(
        r0=R0_est, r1=R1_est, c1=C1_est, 
        q_nom_ah=4.8, # LG M50T nominal capacity
        ocv_poly_coeffs=ocv_params, 
        dt=1.0
    )

    est_soc_history = []
    est_voltage_history = []
    
    logging.info("Running recursive state estimation...")
    for i in range(len(df_drive_cycle)):
        i_load = df_drive_cycle['current'].iloc[i]
        v_meas = df_drive_cycle['voltage'].iloc[i]
        
        soc_est, v_est = ekf.step(current=i_load, measured_voltage=v_meas)
        est_soc_history.append(soc_est)
        est_voltage_history.append(v_est)

    df_drive_cycle['SOC_EKF'] = est_soc_history
    df_drive_cycle['Voltage_EKF'] = est_voltage_history

    # ---------------------------------------------------------
    # 5. Final Evaluation Metrics
    # ---------------------------------------------------------
    evaluator = Evaluator()
    rmse_soc = evaluator.calculate_rmse(df_drive_cycle['SOC'].values, df_drive_cycle['SOC_EKF'].values)
    mae_soc = evaluator.calculate_mae(df_drive_cycle['SOC'].values, df_drive_cycle['SOC_EKF'].values)
    rmse_v = evaluator.calculate_rmse(df_drive_cycle['voltage'].values, df_drive_cycle['Voltage_EKF'].values)

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
    
    # Example: A fresh LG M50T cell has 4.8 Ah. 
    # Let's assume after 500 cycles (from the Zenodo data), the true measured capacity is 4.5 Ah.
    true_aged_capacity_ah = 4.50 
    
    # Assume your machine learning parameter updater (or Coulomb counting integration) 
    # estimated the remaining capacity to be 4.54 Ah.
    estimated_aged_capacity_ah = 4.54 
    
    capacity_error = evaluator.calculate_capacity_prediction_error(
        q_est_ah=estimated_aged_capacity_ah, 
        q_true_ah=true_aged_capacity_ah
    )
    
    logging.info(f"True Aged Capacity      : {true_aged_capacity_ah} Ah")
    logging.info(f"Estimated Aged Capacity : {estimated_aged_capacity_ah} Ah")
    logging.info(f"Capacity Prediction Error: {capacity_error:.2f}%")

if __name__ == "__main__":
    main()