"""
Module: soc_estimator.py
Description: Implements an Extended Kalman Filter (EKF) for real-time State of Charge (SOC) 
estimation using a 1-RC Equivalent-Circuit Model (ECM). Includes rigorous evaluation 
metrics (RMSE, MAE, and Capacity Prediction Error) to validate performance against 
the <=5% error mandate.

Mathematical Formulation:
State Vector (x): [SOC, V1]^T
Input (u): Current (I) - defined as positive during discharge.
"""

import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExtendedKalmanFilter:
    def __init__(self, r0: float, r1: float, c1: float, q_nom_ah: float, 
                 ocv_poly_coeffs: dict, dt: float):
        """
        Initializes the Extended Kalman Filter with ECM parameters and noise covariance matrices.
        
        Args:
            r0 (float): Ohmic resistance (Ohms).
            r1 (float): Polarization resistance (Ohms).
            c1 (float): Polarization capacitance (Farads).
            q_nom_ah (float): Nominal battery capacity in Ampere-hours.
            ocv_poly_coeffs (dict): Dictionary of 8th-order polynomial coefficients (k8 to k0).
            dt (float): Sampling time step in seconds.
        """
        self.R0 = r0
        self.R1 = r1
        self.C1 = c1
        self.Q_nom_As = q_nom_ah * 3600.0  # Convert Ah to Ampere-seconds (Coulombs)
        self.dt = dt
        
        # OCV Polynomial and its mathematical derivative (Jacobian H_k)
        # Assuming dict is formatted as {'k8': val, 'k7': val, ... 'k0': val}
        coeffs_list = [ocv_poly_coeffs[f"k{i}"] for i in range(8, -1, -1)]
        self.ocv_poly = np.poly1d(coeffs_list)
        self.ocv_poly_der = np.polyder(self.ocv_poly)  # Computes d(OCV)/d(SOC)

        # ---------------------------------------------------------
        # EKF Initialization
        # ---------------------------------------------------------
        # State Vector: x = [SOC, V1]^T
        self.x = np.array([[1.0],   # Assume 100% SOC initially (can be dynamically updated)
                           [0.0]])  # Assume zero initial polarization
                           
        # State Covariance Matrix (P)
        self.P = np.array([[1e-4, 0.0],
                           [0.0, 1e-4]])
                           
        # Process Noise Covariance (Q) - Tuning parameters
        # Represents uncertainty in Coulomb counting and ECM drift
        self.Q = np.array([[1e-6, 0.0],
                           [0.0, 1e-5]])
                           
        # Measurement Noise Covariance (R) - Tuning parameter
        # Represents voltage sensor noise variance
        self.R = np.array([[1e-2]]) 

    def step(self, current: float, measured_voltage: float) -> tuple:
        """
        Executes one recursive step of the EKF (Predict and Update).
        Optimized for embedded C translation (basic matrix algebra).
        
        Args:
            current (float): Instantaneous applied current (A). Positive = discharge.
            measured_voltage (float): Instantaneous terminal voltage (V).
            
        Returns:
            tuple: (Estimated SOC, Estimated Terminal Voltage)
        """
        # ==========================================
        # 1. PREDICT PHASE (Time Update)
        # ==========================================
        # Extract previous states
        soc_prev = self.x[0, 0]
        v1_prev = self.x[1, 0]
        
        # State Transition Equations (f)
        soc_pred = soc_prev - (current * self.dt) / self.Q_nom_As
        exp_term = np.exp(-self.dt / (self.R1 * self.C1))
        v1_pred = v1_prev * exp_term + current * self.R1 * (1 - exp_term)
        
        self.x = np.array([[soc_pred], 
                           [v1_pred]])
        
        # State Transition Jacobian (F_k = df/dx)
        F_k = np.array([[1.0, 0.0],
                        [0.0, exp_term]])
                        
        # Predict State Covariance: P_k = F * P_{k-1} * F^T + Q
        self.P = F_k @ self.P @ F_k.T + self.Q

        # ==========================================
        # 2. UPDATE PHASE (Measurement Update)
        # ==========================================
        # Calculate theoretical terminal voltage using the measurement model (h)
        ocv_pred = self.ocv_poly(soc_pred)
        voltage_pred = ocv_pred - current * self.R0 - v1_pred
        
        # Innovation (Residual error between measured and predicted voltage)
        innovation = measured_voltage - voltage_pred
        
        # Measurement Jacobian (H_k = dh/dx)
        # H_k = [ d(OCV)/d(SOC),  -1.0 ]
        d_ocv_d_soc = self.ocv_poly_der(soc_pred)
        H_k = np.array([[d_ocv_d_soc, -1.0]])
        
        # Innovation Covariance (S_k)
        S_k = H_k @ self.P @ H_k.T + self.R
        
        # Kalman Gain (K_k)
        K_k = self.P @ H_k.T @ np.linalg.inv(S_k)
        
        # Update State Estimate (a posteriori)
        self.x = self.x + K_k * innovation
        
        # Update State Covariance (a posteriori)
        I = np.eye(2)
        self.P = (I - K_k @ H_k) @ self.P
        
        # Constrain SOC mathematically between bounds
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)
        
        # Recalculate terminal voltage estimate based on updated state for reporting
        final_voltage_est = self.ocv_poly(self.x[0, 0]) - current * self.R0 - self.x[1, 0]
        
        return float(self.x[0, 0]), float(final_voltage_est)

class Evaluator:
    """
    Computes rigorous mathematical evaluation metrics to validate pipeline accuracy.
    """
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Square Error. Heavily penalizes transient spikes in estimation error.
        """
        return np.sqrt(np.mean((y_true - y_pred)**2))

    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error. Provides a direct linear penalty for average estimation drift.
        """
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def calculate_capacity_prediction_error(q_est_ah: float, q_true_ah: float) -> float:
        """
        Evaluates long-term State of Health (SOH) tracking by measuring 
        the relative percentage error of the estimated cell capacity.
        """
        error_pct = np.abs((q_est_ah - q_true_ah) / q_true_ah) * 100.0
        return float(error_pct)

# --- Example Execution Flow ---
if __name__ == "__main__":
    logging.info("Initializing embedded EKF simulation...")
    
    # 1. Define hardware constants and identified parameters from previous modules
    DT = 1.0  # 1 second sampling time
    NOMINAL_CAPACITY_AH = 4.8  # e.g., for an LG M50T cell
    
    # Mock parameters from ecm_fitter.py
    identified_r0 = 0.035
    identified_r1 = 0.018
    identified_c1 = 1200.0
    
    # Mock parameters from ocv_extractor.py (k8 to k0)
    mock_poly_coeffs = {
        'k8': 0.0, 'k7': 0.0, 'k6': 0.0, 'k5': 0.0, 
        'k4': 0.0, 'k3': 0.0, 'k2': -0.5, 'k1': 1.2, 'k0': 3.0
    }
    
    # 2. Instantiate the Filter
    ekf = ExtendedKalmanFilter(
        r0=identified_r0, r1=identified_r1, c1=identified_c1, 
        q_nom_ah=NOMINAL_CAPACITY_AH, ocv_poly_coeffs=mock_poly_coeffs, dt=DT
    )
    
    # 3. Simulate a short discharging cycle (e.g., 500 seconds at 2A)
    time_steps = 500
    true_soc_array = np.zeros(time_steps)
    est_soc_array = np.zeros(time_steps)
    true_voltage_array = np.zeros(time_steps)
    est_voltage_array = np.zeros(time_steps)
    
    # Initial condition setup
    current_load = 2.0  # Constant 2A discharge
    true_soc = 0.95     # Start simulation slightly off from EKF's 1.0 assumption to test convergence
    
    logging.info("Executing recursive EKF loop over load profile...")
    for k in range(time_steps):
        # Generate Ground Truth
        true_soc = true_soc - (current_load * DT) / (NOMINAL_CAPACITY_AH * 3600.0)
        true_soc_array[k] = true_soc
        
        # Simulate physical voltage (adding Gaussian sensor noise)
        true_v = (3.0 + 1.2*true_soc - 0.5*true_soc**2) - current_load*identified_r0 - 0.01
        noisy_v = true_v + np.random.normal(0, 0.01)
        true_voltage_array[k] = true_v
        
        # Step the EKF
        est_soc, est_voltage = ekf.step(current=current_load, measured_voltage=noisy_v)
        est_soc_array[k] = est_soc
        est_voltage_array[k] = est_voltage

    # 4. Evaluate Performance
    logging.info("--- Simulation Complete. Evaluating Metrics ---")
    
    evaluator = Evaluator()
    soc_rmse = evaluator.calculate_rmse(true_soc_array, est_soc_array)
    soc_mae = evaluator.calculate_mae(true_soc_array, est_soc_array)
    
    voltage_rmse = evaluator.calculate_rmse(true_voltage_array, est_voltage_array)
    voltage_mae = evaluator.calculate_mae(true_voltage_array, est_voltage_array)
    
    logging.info(f"SOC Estimation - RMSE: {soc_rmse:.4f} ({soc_rmse*100:.2f}%), MAE: {soc_mae:.4f} ({soc_mae*100:.2f}%)")
    logging.info(f"Voltage Tracking - RMSE: {voltage_rmse:.4f} V, MAE: {voltage_mae:.4f} V")
    
    if soc_rmse <= 0.05:
         logging.info("SUCCESS: SOC Error is strictly bounded within the <= 5% mandate.")
    else:
         logging.warning("FAIL: SOC Error exceeds 5%. Tune covariance matrices (Q, R).")