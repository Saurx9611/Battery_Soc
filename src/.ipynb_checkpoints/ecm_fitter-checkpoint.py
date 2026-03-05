"""
Module: ecm_fitter.py
Description: Identifies the dynamic Equivalent-Circuit Model (ECM) parameters 
(R0, R1, C1) for a 1-RC Thevenin network using non-linear least squares optimization.
"""

import pandas as pd
import numpy as np
import lmfit
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ECMFitter:
    def __init__(self, ocv_function):
        """
        Initializes the fitter with the established thermodynamic baseline.
        
        Args:
            ocv_function (callable): A function that takes an array of SOC values 
                                     and returns the corresponding OCV values.
        """
        self.ocv_function = ocv_function

    def simulate_1rc_voltage(self, params: lmfit.Parameters, current: np.ndarray, 
                             soc: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """
        Simulates the terminal voltage of the 1-RC ECM using discrete state-space equations.
        Note: Current is defined as positive during discharge.
        """
        R0 = params['R0'].value
        R1 = params['R1'].value
        C1 = params['C1'].value
        
        n_steps = len(current)
        V1 = np.zeros(n_steps)
        Vt = np.zeros(n_steps)
        
        # Calculate OCV vector based on the provided SOC array
        V_ocv = self.ocv_function(soc)
        
        # Initial state assumption (assuming fully relaxed at t=0)
        V1[0] = 0.0
        Vt[0] = V_ocv[0] - current[0] * R0 - V1[0]
        
        # Time-stepping simulation using the discrete state-space equations
        for k in range(n_steps - 1):
            # Calculate the time constant exponent
            exp_term = np.exp(-dt[k] / (R1 * C1))
            
            # Update the polarization voltage state
            V1[k+1] = V1[k] * exp_term + current[k] * R1 * (1 - exp_term)
            
            # Update the measurable terminal voltage
            Vt[k+1] = V_ocv[k+1] - current[k+1] * R0 - V1[k+1]
            
        return Vt

    def objective_function(self, params: lmfit.Parameters, current: np.ndarray, 
                           soc: np.ndarray, dt: np.ndarray, v_exp: np.ndarray) -> np.ndarray:
        """
        Calculates the residual errors between the experimental voltage and the simulated ECM voltage.
        lmfit automatically squares and sums these residuals during optimization.
        """
        v_sim = self.simulate_1rc_voltage(params, current, soc, dt)
        
        # Return the raw residual array (v_exp - v_sim); lmfit handles the SSE calculation
        return v_exp - v_sim

    def fit_dataset(self, df: pd.DataFrame, temperature_label: str, 
                    time_col='relativeTime', current_col='current', 
                    voltage_col='voltage', soc_col='SOC') -> lmfit.Parameters:
        """
        Executes the optimization routine to find the optimal R0, R1, C1 parameters 
        for a specific temperature dataset.
        """
        logging.info(f"Starting parameter identification for dataset: {temperature_label}")
        
        # Ensure data is sorted temporally
        df = df.sort_values(by=time_col).reset_index(drop=True)
        
        # Extract numpy arrays for speed
        t_arr = df[time_col].values
        dt_arr = np.diff(t_arr, prepend=t_arr[0]) # Calculate delta t
        
        # Fix the first dt to be small but non-zero if it happens to be 0
        dt_arr[0] = dt_arr[1] if len(dt_arr) > 1 and dt_arr[1] > 0 else 1.0 
        
        i_arr = df[current_col].values
        v_exp_arr = df[voltage_col].values
        soc_arr = df[soc_col].values

        # Initialize the lmfit Parameters dictionary with strict physical boundaries
        params = lmfit.Parameters()
        
        # R0: Instantaneous ohmic resistance (must be positive, typically in milliohms)
        params.add('R0', value=0.05, min=0.001, max=0.5)
        
        # R1: Charge-transfer & diffusion resistance (must be strictly positive)
        params.add('R1', value=0.02, min=0.0001, max=0.5)
        
        # C1: Polarization capacitance (must be significantly greater than 0)
        params.add('C1', value=1000.0, min=100.0, max=50000.0)

        logging.info("Executing Levenberg-Marquardt minimization...")
        
        # Run the optimizer
        result = lmfit.minimize(
            self.objective_function, 
            params, 
            args=(i_arr, soc_arr, dt_arr, v_exp_arr),
            method='leastsq' # Levenberg-Marquardt
        )
        
        # Log results
        logging.info(f"Optimization successful: {result.success}")
        logging.info(f"Final SSE (Chi-square): {result.chisqr:.4f}")
        logging.info(f"Identified R0: {result.params['R0'].value:.5f} Ohms")
        logging.info(f"Identified R1: {result.params['R1'].value:.5f} Ohms")
        logging.info(f"Identified C1: {result.params['C1'].value:.2f} Farads")
        
        return result.params

# --- Example Execution Flow ---
if __name__ == "__main__":
    # 1. Define a mock OCV function (In reality, this comes from ocv_extractor.py)
    def mock_ocv_func(soc_array):
        return 3.2 + soc_array * 1.0
        
    fitter = ECMFitter(ocv_function=mock_ocv_func)
    
    # 2. Generate a simulated dataset (e.g., a discharge pulse)
    time_steps = np.arange(0, 100, 1) # 100 seconds
    current_profile = np.zeros_like(time_steps, dtype=float)
    current_profile[10:90] = 2.0 # Apply a 2A discharge pulse
    
    # Mock SOC dropping slightly during the pulse
    soc_profile = np.linspace(0.80, 0.79, len(time_steps)) 
    
    # Create the theoretical perfect voltage (using R0=0.03, R1=0.015, C1=1500)
    mock_params = lmfit.Parameters()
    mock_params.add('R0', value=0.03)
    mock_params.add('R1', value=0.015)
    mock_params.add('C1', value=1500)
    
    dt_profile = np.ones_like(time_steps)
    perfect_voltage = fitter.simulate_1rc_voltage(mock_params, current_profile, soc_profile, dt_profile)
    
    # Add some random sensor noise to test the optimizer
    noisy_voltage = perfect_voltage + np.random.normal(0, 0.005, len(perfect_voltage))
    
    df_test = pd.DataFrame({
        'relativeTime': time_steps,
        'current': current_profile,
        'voltage': noisy_voltage,
        'SOC': soc_profile
    })
    
    # 3. Fit the parameters
    identified_params = fitter.fit_dataset(df_test, temperature_label="25C_Test")