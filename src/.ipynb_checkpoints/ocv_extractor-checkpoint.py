"""
Module: ocv_extractor.py
Description: Extracts the thermodynamic Open-Circuit Voltage (OCV) vs. State of Charge (SOC) 
relationship from raw cycler data stored in Parquet format.

Proposed Laboratory Test Protocol for Robust Estimation:
To ensure the SOC estimation error remains <= 5%, the laboratory protocol must isolate 
the true thermodynamic equilibrium from kinetic polarization. 
1. Thermal Stabilization: Soak the cell in an environmental chamber set to the target temperature.
2. Pseudo-OCV Sweeps: Execute an uninterrupted C/30 charge and discharge sweep to capture 
   high-resolution phase-transition plateaus.
3. GITT Anchoring: Apply a Galvanostatic Intermittent Titration Technique (GITT) protocol 
   with discrete pulses at 5% SOC intervals.
4. Relaxation: Enforce a strict 2-to-4 hour relaxation period after each pulse until 
   internal concentration gradients homogenize and the voltage derivative approaches zero.
5. Mathematical Alignment: Post-process the continuous pseudo-OCV curves to intersect 
   the fully relaxed GITT anchor points, canceling out asymmetric kinetic overpotentials.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import logging
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OCVExtractor:
    def __init__(self, polynomial_order: int = 8):
        """
        Initializes the OCV Extractor. 
        Uses an 8th-order polynomial regression to prevent mathematical oscillations
        while minimizing embedded memory footprint.
        """
        self.poly_order = polynomial_order
        self.poly_coefficients = None
        self.ocv_soc_function = None

    def load_parquet_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads the pre-processed time-series data from a Parquet file.
        """
        logging.info(f"Loading Parquet data from {filepath}...")
        df = pd.read_parquet(filepath)
        return df

    def extract_gitt_anchors(self, df_gitt: pd.DataFrame, voltage_col='voltage', current_col='current', soc_col='SOC') -> pd.DataFrame:
        """
        Identifies the fully relaxed voltage points at the absolute end of each GITT rest period.
        Assumes the rest period is characterized by current ~ 0 A and stabilized voltage.
        """
        logging.info("Extracting GITT thermodynamic anchor points...")
        
        # Mask for rest periods (current essentially zero)
        rest_mask = np.abs(df_gitt[current_col]) < 0.001
        df_rest = df_gitt[rest_mask].copy()
        
        # In a robust pipeline, we group by the specific step index or rounded SOC 
        # to capture the final asymptotic voltage of that specific rest phase.
        # Here we group by rounded SOC to find the terminal relaxation point.
        anchors = df_rest.groupby(np.round(df_rest[soc_col], 2))[voltage_col].last().reset_index()
        anchors.columns = ['SOC', 'OCV_Anchor']
        
        logging.info(f"Extracted {len(anchors)} GITT anchors.")
        return anchors

    def align_pseudo_ocv(self, df_pseudo: pd.DataFrame, anchors: pd.DataFrame, voltage_col='voltage', soc_col='SOC') -> pd.DataFrame:
        """
        Aligns the continuous C/30 pseudo-OCV sweep to the rigid GITT anchors.
        Calculates the overpotential offset and shifts the curve to thermodynamic truth.
        """
        logging.info("Aligning pseudo-OCV curve to GITT anchors to remove kinetic overpotential...")
        
        # Create an interpolation function from the pure GITT anchors
        gitt_interp = interp1d(anchors['SOC'], anchors['OCV_Anchor'], kind='linear', fill_value="extrapolate")
        
        # Calculate the dynamic offset between the continuous sweep and the true equilibrium anchors
        df_pseudo['GITT_Expected'] = gitt_interp(df_pseudo[soc_col])
        df_pseudo['Overpotential_Offset'] = df_pseudo[voltage_col] - df_pseudo['GITT_Expected']
        
        # Shift the continuous curve by the calculated transient offset
        df_pseudo['True_OCV'] = df_pseudo[voltage_col] - df_pseudo['Overpotential_Offset']
        
        return df_pseudo[['SOC', 'True_OCV']]

    def fit_polynomial(self, soc_array: np.ndarray, ocv_array: np.ndarray) -> dict:
        """
        Fits an 8th-order polynomial to the True OCV-SOC curve.
        Returns a dictionary of scalar coefficients for deployment in a BMS microcontroller.
        """
        logging.info(f"Fitting {self.poly_order}th-order polynomial to OCV-SOC data...")
        
        # Fit polynomial: OCV(SOC) = k8*SOC^8 + k7*SOC^7 + ... + k0
        # Note: np.polyfit returns highest power first
        coeffs = np.polyfit(soc_array, ocv_array, self.poly_order)
        self.poly_coefficients = coeffs
        
        # Create a callable function for internal evaluation and validation
        self.ocv_soc_function = np.poly1d(coeffs)
        
        # Calculate fitting error (RMSE)
        ocv_pred = self.ocv_soc_function(soc_array)
        rmse = np.sqrt(np.mean((ocv_array - ocv_pred)**2))
        logging.info(f"Polynomial fitting complete. RMSE: {rmse:.4f} V")
        
        # Format for embedded C/C++ deployment
        param_dict = {f"k{self.poly_order - i}": float(coeff) for i, coeff in enumerate(coeffs)}
        return param_dict

    def generate_lookup_table(self, resolution: float = 0.01) -> pd.DataFrame:
        """
        Generates a 1D Look-Up Table (LUT) from the parameterized model.
        Useful for environments where polynomial evaluation is too computationally heavy.
        """
        if self.ocv_soc_function is None:
            raise ValueError("Polynomial must be fitted before generating LUT.")
            
        logging.info(f"Generating OCV-SOC Lookup Table with {resolution} resolution...")
        soc_query = np.arange(0.0, 1.0 + resolution, resolution)
        ocv_query = self.ocv_soc_function(soc_query)
        
        lut = pd.DataFrame({'SOC': soc_query, 'OCV': ocv_query})
        return lut

# --- Example Execution Flow ---
if __name__ == "__main__":
    extractor = OCVExtractor(polynomial_order=8)
    
    # In a real run, you would point this to your saved Parquet files from Phase 1
    # Example: 
    # df_gitt = extractor.load_parquet_data("data/processed/gitt_test_data.parquet")
    # df_pseudo = extractor.load_parquet_data("data/processed/pseudo_ocv_data.parquet")
    
    # ---------------------------------------------------------
    # Mocking data for standalone execution demonstration
    # ---------------------------------------------------------
    mock_soc = np.linspace(0, 1, 100)
    mock_ocv_gitt = 3.2 + mock_soc * 1.0 + 0.05 * np.sin(mock_soc * np.pi) 
    
    # 1. Extract Anchors (mocking the output directly)
    anchors = pd.DataFrame({'SOC': np.arange(0, 1.05, 0.05), 
                            'OCV_Anchor': 3.2 + np.arange(0, 1.05, 0.05) * 1.0})
    
    # 2. Align Pseudo-OCV (mocking a continuous dataframe)
    df_pseudo = pd.DataFrame({'SOC': mock_soc, 'voltage': mock_ocv_gitt + 0.02}) # Added artificial 20mV overpotential
    aligned_ocv_df = extractor.align_pseudo_ocv(df_pseudo, anchors)
    
    # 3. Fit Polynomial
    parameters = extractor.fit_polynomial(aligned_ocv_df['SOC'].values, aligned_ocv_df['True_OCV'].values)
    
    print("\n--- Deployable BMS Polynomial Coefficients ---")
    for key, val in parameters.items():
        print(f"  {key}: {val:.6e}")
        
    # 4. Generate LUT
    lut = extractor.generate_lookup_table()
    print(f"\nGenerated LUT with {len(lut)} standard points.")