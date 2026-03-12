import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Constants
TIME = 'Time (s)'
VOLTAGE = 'Voltage (V)'
CURRENT = 'Current (mA)'
CHARGE = 'Charge (mA.h)'

class BatteryEKF:
    def __init__(self, q_capacity_ah, ocv_soc_curve, r0, r1, c1):
        self.Q_ah = q_capacity_ah
        self.R0 = r0
        self.R1 = r1
        self.C1 = c1

        self.get_ocv = interp1d(ocv_soc_curve[0], ocv_soc_curve[1], fill_value="extrapolate")
        
        soc_pts = ocv_soc_curve[0]
        ocv_pts = ocv_soc_curve[1]
        docv_dsoc = np.gradient(ocv_pts, soc_pts)
        self.get_docv_dsoc = interp1d(soc_pts, docv_dsoc, fill_value="extrapolate")

        self.x = np.array([[1.0], [0.0]])
        self.P = np.array([[1e-4, 0], [0, 1e-4]])
        self.Q = np.array([[1e-6, 0], [0, 1e-5]])
        self.R_noise = 1e-2

    def step(self, current_A, voltage_V, dt):
        soc_prev = self.x[0, 0]
        v1_prev = self.x[1, 0]

        # Prediction Step
        tau = self.R1 * self.C1 if self.R1 * self.C1 > 0 else 1e-6
        exp_val = np.exp(-dt / tau)

        soc_pred = soc_prev + (current_A * dt) / (3600.0 * self.Q_ah)
        v1_pred = exp_val * v1_prev + self.R1 * (1 - exp_val) * current_A

        x_pred = np.array([[soc_pred], [v1_pred]])

        A = np.array([[1.0, 0.0], [0.0, exp_val]])
        P_pred = A @ self.P @ A.T + self.Q

        # Correction Step
        ocv_pred = self.get_ocv(soc_pred)
        v_est = ocv_pred + self.R0 * current_A + v1_pred

        docv = self.get_docv_dsoc(soc_pred)
        C = np.array([[docv, 1.0]])

        S = C @ P_pred @ C.T + self.R_noise
        K = (P_pred @ C.T) / S

        y_residual = voltage_V - v_est

        self.x = x_pred + K * y_residual
        self.P = (np.eye(2) - K @ C) @ P_pred
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)

        return self.x[0, 0], self.x[1, 0], float(v_est)

def extract_ocv_curve(df, capacity_ah):
    """Extracts OCV-SOC points dynamically from GITT relaxation data."""
    ocv_points, soc_points = [], []
    
    # Add initial full-charge point based on voltage right before discharge starts
    start_discharge_idx = df[df[CURRENT] < -5].index[0]
    ocv_points.append(df[VOLTAGE].iloc[start_discharge_idx - 1])
    soc_points.append(1.0)

    relax_start_indices = df[(df[CURRENT] == 0) & (df[CURRENT].shift(1) < -5)].index
    for row in relax_start_indices:
        future_current = df.loc[row + 1:, CURRENT]
        relax_end_idx = future_current[future_current != 0].index[0] - 1 if len(future_current[future_current != 0]) > 0 else df.index[-1]
            
        ocv_points.append(df[VOLTAGE].iloc[relax_end_idx])
        discharged_ah = df[CHARGE].iloc[row - 1] / 1000.0
        soc_points.append(max(0.0, 1.0 - (discharged_ah / capacity_ah)))

    # Sort strictly for interpolation
    sorted_indices = np.argsort(soc_points)
    soc_arr = np.array(soc_points)[sorted_indices]
    ocv_arr = np.array(ocv_points)[sorted_indices]
    
    # Ensure no duplicates
    for i in range(1, len(soc_arr)):
        if soc_arr[i] <= soc_arr[i-1]: soc_arr[i] = soc_arr[i-1] + 1e-5
            
    return soc_arr, ocv_arr

# ----------------- EXECUTION SCRIPT ----------------- #
# Make sure the file is in your working directory
file_name = "Expt 5 - cell A - RPT0 - 5-pulse GITT 0.5C discharge data.csv"
df = pd.read_csv(file_name)

# 1. Setup Parameters
# The max discharged capacity in the file is ~4.79Ah
capacity_ah = df['Charge (mA.h)'].max() / 1000.0 
soc_arr, ocv_arr = extract_ocv_curve(df, capacity_ah)
ocv_curve_tuple = (soc_arr, ocv_arr)

# 2. Prepare Data (Start from when discharge actually begins)
start_idx = df[df[CURRENT] < -5].index[0]
test_df = df.iloc[start_idx:].copy().reset_index(drop=True)

# Correct the Time so it starts at 0 for plotting
test_df[TIME] = test_df[TIME] - test_df[TIME].iloc[0]

# Generate True SOC based on raw Coulomb Counting for evaluation
test_df['True SOC (%)'] = (1.0 - (test_df['Charge (mA.h)'] / (capacity_ah * 1000.0))) * 100.0

# 3. Run EKF Estimator
# Using average ECM values calculated from this dataset previously
R0_base, R1_base, C1_base = 0.028, 0.028, 17000.0

ekf = BatteryEKF(capacity_ah, ocv_curve_tuple, R0_base, R1_base, C1_base)

estimated_soc, estimated_voltage = [], []

for i in range(len(test_df)):
    dt = 1.0 if i == 0 else test_df[TIME].iloc[i] - test_df[TIME].iloc[i-1]
    if dt <= 0: dt = 1.0

    current_A = test_df[CURRENT].iloc[i] / 1000.0
    voltage_V = test_df[VOLTAGE].iloc[i]

    soc_est, v1_est, v_est = ekf.step(current_A, voltage_V, dt)
    
    estimated_soc.append(soc_est * 100.0)
    estimated_voltage.append(v_est)

test_df['Estimated SOC (%)'] = estimated_soc
test_df['Predicted Voltage (V)'] = estimated_voltage

# 4. Evaluation Metrics
rmse_v = np.sqrt(mean_squared_error(test_df['Voltage (V)'], test_df['Predicted Voltage (V)']))
mae_v = mean_absolute_error(test_df['Voltage (V)'], test_df['Predicted Voltage (V)'])
rmse_soc = np.sqrt(mean_squared_error(test_df['True SOC (%)'], test_df['Estimated SOC (%)']))
mae_soc = mean_absolute_error(test_df['True SOC (%)'], test_df['Estimated SOC (%)'])
cap_err_ah = abs((test_df['True SOC (%)'].iloc[-1] - test_df['Estimated SOC (%)'].iloc[-1]) / 100.0 * capacity_ah)

print(f"--- EKF Evaluation Metrics ---")
print(f"Voltage RMSE: {rmse_v:.4f} V")
print(f"Voltage MAE:  {mae_v:.4f} V")
print(f"SOC RMSE:     {rmse_soc:.2f} %")
print(f"SOC MAE:      {mae_soc:.2f} %")
print(f"Cap Error:    {cap_err_ah:.4f} Ah")

# ----------------- PLOTTING CODE ----------------- #
plt.figure(figsize=(10, 8))

# Subplot 1: Voltage Tracking
plt.subplot(2, 1, 1)
plt.plot(test_df['Time (s)'], test_df['Voltage (V)'], label='Measured Voltage (True)', color='green')
plt.plot(test_df['Time (s)'], test_df['Predicted Voltage (V)'], label='EKF Predicted Voltage', color='red', linestyle='--')
plt.title('EKF Voltage Tracking', fontsize=14)
plt.ylabel('Voltage (V)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: SOC Estimation
plt.subplot(2, 1, 2)
plt.plot(test_df['Time (s)'], test_df['True SOC (%)'], label='Coulomb Counting SOC (True)', color='blue')
plt.plot(test_df['Time (s)'], test_df['Estimated SOC (%)'], label='EKF Estimated SOC', color='orange', linestyle='--')
plt.title('EKF Real-Time SOC Estimation', fontsize=14)
plt.ylabel('SOC (%)', fontsize=12)
plt.xlabel('Time (s)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()