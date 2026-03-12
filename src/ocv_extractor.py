import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
file_path = 'Expt 5 - cell A - RPT0 - 0.1C discharge data.csv'
df = pd.read_csv(file_path)

# 2. Rename columns to match your constants.py definitions
# Your CSV has: 'Time (s)', 'Voltage (V)', 'Current (mA)', 'Charge (mA.h)', 'Temperature (degC)'
df = df.rename(columns={
    'Time (s)': 'Time (s)',
    'Voltage (V)': 'Voltage (V)',
    'Current (mA)': 'Current (mA)',
    'Charge (mA.h)': 'Discharge (mA.h)', # Renaming because current is negative, so this is discharged capacity
    'Temperature (degC)': 'Temperature (degC)'
})

# 3. Calculate SOC (%)
# Since it's a discharge test, SOC starts at 100% and goes down as capacity is drawn.
max_discharge_capacity = df['Discharge (mA.h)'].max()

# Formula: 100 * (1 - (current_discharge / max_discharge))
df['SOC (%)'] = 100 * (1 - (df['Discharge (mA.h)'] / max_discharge_capacity))

# 4. Generate the OCV vs SOC Curve (Pseudo-OCV from 0.1C Discharge)
plt.figure(figsize=(10, 6))
plt.plot(df['SOC (%)'], df['Voltage (V)'], label='0.1C Discharge (Pseudo-OCV)', color='blue')

# Formatting the plot
plt.title('OCV vs SOC Curve (Cell A - RPT0)')
plt.xlabel('State of Charge (SOC) [%]')
plt.ylabel('Voltage [V]')
plt.xlim(0, 100) # SOC ranges from 0 to 100
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Display the plot
plt.show()

# Optional: If you want to pass this dataframe into your custom cell_data.py or analysis_functions.py
# The dataframe 'df' now contains the exact columns ['Voltage (V)', 'Current (mA)', 'SOC (%)'] 
# required by functions like those in your 'analysis_functions.py' module.