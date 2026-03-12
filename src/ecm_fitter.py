import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants for column mapping
TIME = 'Time (s)'
VOLTAGE = 'Voltage (V)'
CURRENT = 'Current (mA)'
CHARGE = 'Charge (mA.h)' # Using Charge as Q

def relaxation_equation(t, v_inf, b, tau):
    """Exponential function to fit the relaxation data."""
    return v_inf - b * np.exp(-t / tau)

def get_ecm_parameters_and_plot(file_path):
    df = pd.read_csv(file_path)
    
    q_vals, r0_vals, r1_vals, c1_vals, tau_vals = [], [], [], [], []
    
    # Identify relaxation periods
    relax_start_indices = df[(df[CURRENT] == 0) & (df[CURRENT].shift(1) < -5)].index
    
    for row in relax_start_indices:
        pulse_end_idx = row - 1
        relax_start_idx = row
        
        # Find end of relaxation
        future_current = df.loc[relax_start_idx + 1:, CURRENT]
        if len(future_current[future_current != 0]) > 0:
            relax_end_idx = future_current[future_current != 0].index[0] - 1
        else:
            relax_end_idx = df.index[-1]
            
        # Calculate Delta I (A)
        delta_i = abs(df[CURRENT].iloc[pulse_end_idx] - df[CURRENT].iloc[relax_start_idx]) / 1000
        if delta_i == 0:
            continue
            
        # 1. Calculate R0 (Instantaneous voltage drop right when current stops)
        v_pulse_end = df[VOLTAGE].iloc[pulse_end_idx]
        v_relax_start = df[VOLTAGE].iloc[relax_start_idx]
        delta_v0 = abs(v_relax_start - v_pulse_end)
        r0 = delta_v0 / delta_i
            
        # Curve Fitting
        relax_df = df.loc[relax_start_idx:relax_end_idx].copy()
        t_data = relax_df[TIME] - relax_df[TIME].iloc[0]
        v_data = relax_df[VOLTAGE]
        
        v_inf_guess = v_data.iloc[-1]
        b_guess = v_inf_guess - v_data.iloc[0]
        tau_guess = 1000.0 
        
        try:
            popt, _ = curve_fit(
                relaxation_equation, 
                t_data, 
                v_data, 
                p0=[v_inf_guess, b_guess, tau_guess],
                bounds=(0, np.inf)
            )
            _, b_fit, tau_fit = popt
            
            # ECM Math
            r1 = b_fit / delta_i
            c1 = tau_fit / r1 if r1 > 0 else 0
            
            # Store values
            q_vals.append(df[CHARGE].iloc[pulse_end_idx])
            r0_vals.append(r0)
            r1_vals.append(r1)
            c1_vals.append(c1)
            tau_vals.append(tau_fit)
            
        except RuntimeError:
            continue

    # Create the DataFrame
    output_data = [[q, r_0, r_1, c_1, t] for q, r_0, r_1, c_1, t in zip(q_vals, r0_vals, r1_vals, c1_vals, tau_vals)]
    ecm_output = pd.DataFrame(output_data, columns=['Q (mAh)', 'R0 (Ohms)', 'R1 (Ohms)', 'C1 (Farads)', 'Tau (s)'])
    
    # ------------------ Plotting Code ------------------
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Plot R0 (Green) and R1 (Blue) on the Left Axis
    l1 = ax1.plot(ecm_output['Q (mAh)'], ecm_output['R0 (Ohms)'], marker='s', color='green', label='R0')
    l2 = ax1.plot(ecm_output['Q (mAh)'], ecm_output['R1 (Ohms)'], marker='o', color='blue', label='R1')
    
    ax1.set_xlabel('Discharged Capacity Q (mAh)', fontsize=12)
    ax1.set_ylabel('Resistance (Ohms)', color='black', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot C1 (Orange) on the Right Axis
    ax2 = ax1.twinx()
    l3 = ax2.plot(ecm_output['Q (mAh)'], ecm_output['C1 (Farads)'], marker='^', color='orange', label='C1')
    ax2.set_ylabel('Capacitance C1 (Farads)', color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='orange')

    # Create a single legend for all 3 lines
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center left')

    plt.title('R0, R1, and C1 vs Discharge Capacity (Q)', fontsize=14)
    fig.tight_layout()
    plt.show()

    return ecm_output

# Run the function
file_name = "Expt 5 - cell A - RPT0 - 5-pulse GITT 0.5C discharge data.csv"
results = get_ecm_parameters_and_plot(file_name)
print(results)