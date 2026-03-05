import pandas as pd
import numpy as np
from pathlib import Path
from prog_models.datasets import nasa_battery
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BatteryDataPipeline:
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        self.raw_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_data_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def ingest_nasa_data(self, batt_id: str) -> pd.DataFrame:
        """
        Fetches the NASA Randomized Battery dataset using the prog_models API.
        """
        logging.info(f"Fetching NASA dataset for battery: {batt_id}")
        try:
            # The API returns a tuple; we extract the list of DataFrames
            data_tuple = nasa_battery.load_data(batt_id)
            # Assuming the second element contains the actual time-series DataFrames
            df_list = data_tuple[1] 
            
            # Concatenate all operational profiles into a single DataFrame
            df = pd.concat(df_list, ignore_index=True)
            logging.info(f"Successfully loaded {len(df)} rows for {batt_id}")
            return df
        except Exception as e:
            logging.error(f"Failed to load NASA data for {batt_id}: {e}")
            raise

    def ingest_zenodo_mpr(self, file_name: str) -> pd.DataFrame:
        """
        Parses multi-gigabyte .mpr files from the Zenodo LG M50T dataset using galvani.
        """
        file_path = self.raw_dir / file_name
        logging.info(f"Parsing Biologic .mpr file: {file_path}")
        try:
            mpr_file = galvani_mpr.MPRfile(str(file_path))
            df = pd.DataFrame(mpr_file.data)
            logging.info(f"Successfully parsed {len(df)} rows from {file_name}")
            return df
        except Exception as e:
            logging.error(f"Failed to parse .mpr file {file_name}: {e}")
            raise

    def engineer_features(self, df: pd.DataFrame, time_col: str, voltage_col: str, current_col: str) -> pd.DataFrame:
        """
        Applies a 500-second rolling window and calculates the voltage derivative 
        using a 5-point stencil method for noise reduction.
        """
        logging.info("Engineering temporal features...")
        df = df.copy()
        
        # Ensure data is sorted by time
        df = df.sort_values(by=time_col).reset_index(drop=True)
        
        # Calculate time step (dt) in seconds assuming uniform sampling
        dt = df[time_col].diff().median()
        
        # 1. Rolling Averages (500-second window)
        # We calculate how many rows make up 500 seconds
        window_size = max(1, int(500 / dt)) if dt > 0 else 500
        df['voltage_rolling_avg'] = df[voltage_col].rolling(window=window_size, min_periods=1).mean()
        df['current_rolling_avg'] = df[current_col].rolling(window=window_size, min_periods=1).mean()

        # 2. 5-Point Stencil Derivative for Voltage (dV/dt)
        # Formula: f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
        v = df[voltage_col].values
        dvdt = np.zeros_like(v)
        
        # Apply stencil where we have enough boundaries, otherwise leave as 0 or use simple diff
        for i in range(2, len(v) - 2):
            dvdt[i] = (-v[i+2] + 8*v[i+1] - 8*v[i-1] + v[i-2]) / (12 * dt)
            
        df['dV_dt_stencil'] = dvdt

        logging.info("Feature engineering complete.")
        return df

    def save_to_parquet(self, df: pd.DataFrame, output_filename: str):
        """
        Saves the processed DataFrame to a highly optimized Apache Parquet format.
        """
        output_path = self.processed_dir / output_filename
        logging.info(f"Saving data to Parquet: {output_path}")
        
        # Save to parquet with Snappy compression (good balance of speed and size)
        df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        logging.info("Save complete.")

# --- Example Execution ---
if __name__ == "__main__":
    pipeline = BatteryDataPipeline(raw_data_dir="./data/raw", processed_data_dir="./data/processed")
    
    # Example: Processing NASA Data
    # nasa_df = pipeline.ingest_nasa_data("RW1")
    # if 'relativeTime' in nasa_df.columns:
    #     nasa_df = pipeline.engineer_features(nasa_df, time_col='relativeTime', voltage_col='voltage', current_col='current')
    #     pipeline.save_to_parquet(nasa_df, "nasa_RW1_processed.parquet")