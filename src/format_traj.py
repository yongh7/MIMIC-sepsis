import argparse
import json
import numpy as np
import pandas as pd
import pyprind
import os
from scipy.interpolate import interp1d
from fancyimpute import KNN
import math
import warnings 

warnings.filterwarnings("ignore", category=RuntimeWarning)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_raw", action='store_true', 
                       help="If specified, additionally save trajectories without normalized features")
    parser.add_argument("--output_dir", type=str, default="processed_files",
                       help="Directory to save processed files")
    parser.add_argument("--missing_threshold", type=float, default=0.8,
                       help="Threshold for dropping columns with missing values (default: 0.7)")
    parser.add_argument("--low_missing_threshold", type=float, default=0.05,
                       help="Threshold for using linear interpolation instead of KNN (default: 0.05)")
    parser.add_argument("--knn_neighbors", type=int, default=1,
                       help="Number of neighbors to use for KNN imputation (default: 1)")
    parser.add_argument("--knn_chunk_size", type=int, default=9999,
                       help="Chunk size for KNN imputation processing (default: 9999)")
    parser.add_argument("--fluid_window", type=int, default=12,
                       help="Window in hours for fluid calculation in septic shock detection (default: 12)")
    # TODO: double check the unit to be ml
    parser.add_argument("--min_fluid_threshold", type=float, default=2000,
                       help="Minimum fluid threshold in mL for septic shock detection (default: 2000)")
    parser.add_argument("--map_threshold", type=float, default=65,
                       help="MAP threshold for septic shock detection (default: 65)")
    parser.add_argument("--lactate_threshold", type=float, default=2,
                       help="Lactate threshold for septic shock detection (default: 2)")
    parser.add_argument("--timestep", type=int, default=4,
                       help="Size of timestep in hours (default: 4)")
    parser.add_argument("--window_before", type=int, default = 24,
                       help="Hours to include before onset time (default: 24)")
    parser.add_argument("--window_after", type=int, default= 72,
                       help="Hours to include after onset time (default: 72)")
    parser.add_argument("--notes_dir", type=str, default="processed_files",
                       help="Directory containing processed notes files")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Number of subjects to sample for testing (default: None, use all subjects)")
    return parser.parse_args()

def load_processed_files():
    print('Loading processed files created from database using "preprocess.py"')
    files = {
        'stay': 'icustays.csv',
        'abx': 'abx_processed.csv', 
        'bacterio': 'bacterio_processed.csv',
        'demog': 'demog_processed.csv',
        'ce': 'chartevents.csv',
        'MV': 'mechvent.csv',
        'fluid': 'fluid.csv',
        'vaso': 'vaso.csv',
        'UO': 'uo.csv',
        'labU': 'labu.csv',
        'onset': 'onset.csv'
    }
    
    data = {}
    for key, filename in files.items():
        data[key] = pd.read_csv(f'processed_files/{filename}', sep='|')
        
    return data

def load_measurement_mappings():
    """Load the measurement mappings from JSON file"""
    print('Loading measurement mappings')
    with open("ReferenceFiles/measurement_mappings.json", "r") as f:
        measurements = json.load(f)
    
    # Create reverse mapping (code to concept)
    code_to_concept = {}
    for concept, info in measurements.items():
        for code in info['codes']:
            code_to_concept[code] = concept
    
    # Create hold times mapping, used for sample and hold
    hold_times = {}
    for concept, info in measurements.items():
        if 'hold_time' in info:
            hold_times[concept] = info['hold_time']
            
    return measurements, code_to_concept, hold_times

def process_patient_measurements(data, measurements, code_to_concept, icustayid, onset_time, winb4=24, winaft=72):
    """Process data for a single patient"""
    # Get relevant data for this patient
    temp = data['ce'][data['ce']['stay_id'] == icustayid].copy()
    temp2 = data['labU'][data['labU']['stay_id'] == icustayid].copy()
    temp3 = data['MV'][data['MV']['stay_id'] == icustayid].copy()
    
    # Filter for time window
    time_window = lambda df: (df['charttime'] >= onset_time - winb4*3600) & \
                           (df['charttime'] < onset_time + winaft*3600)
    
    temp = temp[time_window(temp)]
    temp2 = temp2[time_window(temp2)]
    temp3 = temp3[time_window(temp3)]
    
    # Get unique timestamps
    t = np.unique(pd.concat([temp['charttime'], 
                           temp2['charttime'], 
                           temp3['charttime']], ignore_index=True).values)
    
    if len(t) == 0:
        return None
        
    patient_data = []
    
    # Process each timestamp
    for timestamp in t:
        row_data = {
            'stay_id': icustayid,
            'charttime': timestamp
        }
        
        # Process chartevents
        mask = temp['charttime'] == timestamp
        for _, row in temp[mask].iterrows():
            item_id = str(int(row['itemid']))
            if item_id in code_to_concept:
                concept = code_to_concept[item_id]
                row_data[concept] = row['valuenum']
        
        # Process lab values
        mask = temp2['charttime'] == timestamp
        for _, row in temp2[mask].iterrows():
            item_id = str(int(row['itemid']))
            if item_id in code_to_concept:
                concept = code_to_concept[item_id]
                row_data[concept] = row['valuenum']
        
        # Process mechanical ventilation
        mask = temp3['charttime'] == timestamp
        if mask.any():
            row_data['mechvent'] = temp3.loc[mask, 'mechvent'].values[0]
        
        patient_data.append(row_data)
    
    return pd.DataFrame(patient_data)


def handle_outliers(df):
    """Handle outliers in the patient timeseries data based on clinical thresholds
    """
    print('Handling outliers in patient timeseries data')
    
    # Weight
    df.loc[df['weight_kg'] > 300, 'weight_kg'] = np.nan
    df.loc[df['weight_lb'] > 660, 'weight_lb'] = np.nan
    
    # Heart Rate
    df.loc[df['heart_rate'] > 250, 'heart_rate'] = np.nan
    
    # Blood Pressure
    df.loc[df['sbp_arterial'] > 300, 'sbp_arterial'] = np.nan
    df.loc[df['map'] < 0, 'map'] = np.nan
    df.loc[df['map'] > 200, 'map'] = np.nan
    df.loc[df['dbp_arterial'] < 0, 'dbp_arterial'] = np.nan
    df.loc[df['dbp_arterial'] > 200, 'dbp_arterial'] = np.nan
    
    # Respiratory Rate
    df.loc[df['respiratory_rate'] > 80, 'respiratory_rate'] = np.nan
    
    # SpO2
    df.loc[df['spo2'] > 150, 'spo2'] = np.nan
    df.loc[df['spo2'] > 100, 'spo2'] = 100
    
    # Temperature
    mask = (df['temp_C'] > 90) & (df['temp_F'].isna())
    df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C']
    df.loc[df['temp_C'] > 90, 'temp_C'] = np.nan
    
    # FiO2
    df.loc[df['fio2'] > 100, 'fio2'] = np.nan
    df.loc[df['fio2'] < 1, 'fio2'] *= 100
    df.loc[df['fio2'] < 20, 'fio2'] = np.nan
    
    # O2 Flow
    df.loc[df['oxygen_flow'] > 70, 'oxygen_flow'] = np.nan
    
    # PEEP
    df.loc[df['peep'] < 0, 'peep'] = np.nan
    df.loc[df['peep'] > 40, 'peep'] = np.nan
    
    # Tidal Volume
    df.loc[df['tidal_volume'] > 1800, 'tidal_volume'] = np.nan
    
    # Minute Ventilation
    df.loc[df['minute_volume'] > 50, 'minute_volume'] = np.nan
    
    # Lab Values
    df.loc[df['potassium'] < 1, 'potassium'] = np.nan
    df.loc[df['potassium'] > 15, 'potassium'] = np.nan
    
    df.loc[df['sodium'] < 95, 'sodium'] = np.nan
    df.loc[df['sodium'] > 178, 'sodium'] = np.nan
    
    df.loc[df['chloride'] < 70, 'chloride'] = np.nan
    df.loc[df['chloride'] > 150, 'chloride'] = np.nan
    
    df.loc[df['glucose'] < 1, 'glucose'] = np.nan
    df.loc[df['glucose'] > 1000, 'glucose'] = np.nan
    
    df.loc[df['creatinine'] > 150, 'creatinine'] = np.nan
    df.loc[df['magnesium'] > 10, 'magnesium'] = np.nan
    df.loc[df['calcium_total'] > 20, 'calcium_total'] = np.nan
    df.loc[df['calcium_ionized'] > 5, 'calcium_ionized'] = np.nan

    df.loc[df['total_co2'] > 120, 'total_co2'] = np.nan
    
    df.loc[df['ast'] > 10000, 'ast'] = np.nan
    df.loc[df['alt'] > 10000, 'alt'] = np.nan
    
    df.loc[df['hemoglobin'] > 20, 'hemoglobin'] = np.nan
    df.loc[df['hematocrit'] > 65, 'hematocrit'] = np.nan
    df.loc[df['wbc'] > 500, 'wbc'] = np.nan
    df.loc[df['platelets'] > 2000, 'platelets'] = np.nan
    
    df.loc[df['inr'] > 20, 'inr'] = np.nan
    
    df.loc[df['ph_arterial'] < 6.7, 'ph_arterial'] = np.nan
    df.loc[df['ph_arterial'] > 8, 'ph_arterial'] = np.nan
    
    df.loc[df['arterial_o2_pressure'] > 700, 'arterial_o2_pressure'] = np.nan
    df.loc[df['arterial_co2_pressure'] > 200, 'arterial_co2_pressure'] = np.nan
    df.loc[df['arterial_base_excess'] < -50, 'arterial_base_excess'] = np.nan
    df.loc[df['lactic_acid'] > 30, 'lactic_acid'] = np.nan
    
    df.loc[df['bilirubin_total'] > 30, 'bilirubin_total'] = np.nan
    
    return df

def estimate_gcs_from_rass(df):
    """
    Estimate Glasgow Coma Scale (GCS) from Richmond Agitation-Sedation Scale (RASS)
    Based on data from Wesley JAMA 2003
    
    If GCS column doesn't exist, creates it and initializes with NaN values.
    """
    # Create GCS column if it doesn't exist
    if 'gcs' not in df.columns:
        df['gcs'] = np.nan
    
    # RASS +4 (Combative) -> GCS = 15
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == 4), 'gcs'] = 15
    
    # RASS +3 (Pulls tubes) -> GCS = 15
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == 3), 'gcs'] = 15
    
    # RASS +2 (Fights ventilator) -> GCS = 15
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == 2), 'gcs'] = 15
    
    # RASS +1 (Anxious) -> GCS = 15
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == 1), 'gcs'] = 15
    
    # RASS 0 (Alert and calm) -> GCS = 15
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == 0), 'gcs'] = 15
    
    # RASS -1 (Awakens to voice >10s) -> GCS = 14
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == -1), 'gcs'] = 14
    
    # RASS -2 (Light sedation, awakens <10s) -> GCS = 12
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == -2), 'gcs'] = 12
    
    # RASS -3 (Moderate sedation) -> GCS = 11
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == -3), 'gcs'] = 11
    
    # RASS -4 (Deep sedation) -> GCS = 6
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == -4), 'gcs'] = 6
    
    # RASS -5 (Unarousable) -> GCS = 3
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == -5), 'gcs'] = 3
    
    return df

def estimate_fio2(df):
    """
    Estimate FiO2 values based on O2 flow rate. FiO2 is a critical measurement needed to
    Calculate the P/F ratio (PaO2/FiO2) which is used in SOFA scoring
    
    Args:
        df: pandas DataFrame containing patient data with columns:
            - oxygen_flow_device: type of oxygen delivery device (numeric codes)
            - oxygen_flow, oxygen_flow_cannula_rate, oxygen_flow_rate: different flow measurements
            - fio2: Fraction of inspired oxygen
    Returns:
        DataFrame with estimated fio2 values
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Combine all oxygen flow measurements into a single column
    # Taking the first non-null value from any of the flow measurements, (horizontal fill)
    flow_columns = ['oxygen_flow', 'oxygen_flow_cannula_rate', 'oxygen_flow_rate']
    df['combined_o2_flow'] = df[flow_columns].bfill(axis=1).iloc[:, 0]
    
    # Helper function to set FiO2 based on O2 flow thresholds
    def set_fio2_by_flow(mask, flow_thresholds, fio2_values):
        df_subset = df[mask].copy()
        df_subset['fio2'] = None 
        for threshold, fio2 in zip(flow_thresholds, fio2_values):
            flow_mask = df_subset['combined_o2_flow'] <= threshold
            df_subset.loc[flow_mask, 'fio2'] = fio2
        df.loc[mask, 'fio2'] = df_subset['fio2']

    # Case 1: No FiO2, Yes O2 flow, No interface or nasal cannula
    mask = (df['fio2'].isna()) & \
           (df['combined_o2_flow'].notna()) & \
           (df['oxygen_flow_device'].isin(['0', '2']))  # None or Nasal cannula
    
    if mask.any():
        flow_thresholds = [15, 12, 10, 8, 6, 5, 4, 3, 2, 1]
        fio2_values = [70, 62, 55, 50, 44, 40, 36, 32, 28, 24]
        set_fio2_by_flow(mask, flow_thresholds, fio2_values)

    # Case 2: No FiO2, No O2 flow, No interface or nasal cannula
    mask = (df['fio2'].isna()) & \
           (df['combined_o2_flow'].isna()) & \
           (df['oxygen_flow_device'].isin(['0', '2']))  # None or Nasal cannula
    df.loc[mask, 'fio2'] = 21  # Room air

    # Case 3: No FiO2, Yes O2 flow, Face mask or similar devices
    face_mask_types = ['3', '4', '5', '6', '8', '9', '10', '11', '12']  # Face tent through T-piece
    mask = (df['fio2'].isna()) & \
           (df['combined_o2_flow'].notna()) & \
           (df['oxygen_flow_device'].isin(face_mask_types))
    
    if mask.any():
        flow_thresholds = [15, 12, 10, 8, 6, 4]
        fio2_values = [75, 69, 66, 58, 40, 36]
        set_fio2_by_flow(mask, flow_thresholds, fio2_values)

    # Case 4: No FiO2, Yes O2 flow, Non-rebreather mask
    mask = (df['fio2'].isna()) & \
           (df['combined_o2_flow'].notna()) & \
           (df['oxygen_flow_device'] == '7')  # Non-rebreather
    
    if mask.any():
        df_subset = df[mask].copy()
        flow = df_subset['combined_o2_flow']
        
        df_subset.loc[flow >= 15, 'fio2'] = 100
        df_subset.loc[(flow >= 10) & (flow < 15), 'fio2'] = 90
        df_subset.loc[(flow < 10) & (flow > 8), 'fio2'] = 80
        df_subset.loc[(flow <= 8) & (flow > 6), 'fio2'] = 70
        df_subset.loc[flow <= 6, 'fio2'] = 60
        
        df.loc[mask, 'fio2'] = df_subset['fio2']

    # Case 5: No FiO2, Yes O2 flow, CPAP/BiPAP mask
    mask = (df['fio2'].isna()) & \
           (df['combined_o2_flow'].notna()) & \
           (df['oxygen_flow_device'] == '13')  # CPAP/BiPAP mask
    
    if mask.any():
        df_subset = df[mask].copy()
        flow = df_subset['combined_o2_flow']
        
        df_subset.loc[flow >= 15, 'fio2'] = 100
        df_subset.loc[(flow >= 10) & (flow < 15), 'fio2'] = 80
        df_subset.loc[flow < 10, 'fio2'] = 60
        
        df.loc[mask, 'fio2'] = df_subset['fio2']

    # Case 6: No FiO2, Yes O2 flow, Oxymizer
    mask = (df['fio2'].isna()) & \
           (df['combined_o2_flow'].notna()) & \
           (df['oxygen_flow_device'] == '14')  # Oxymizer
    
    if mask.any():
        df_subset = df[mask].copy()
        flow = df_subset['combined_o2_flow']
        
        df_subset.loc[flow >= 10, 'fio2'] = 80
        df_subset.loc[(flow >= 5) & (flow < 10), 'fio2'] = 60
        df_subset.loc[flow < 5, 'fio2'] = 40
        
        df.loc[mask, 'fio2'] = df_subset['fio2']

    # Clean up temporary column
    df = df.drop('combined_o2_flow', axis=1)

    
    
    return df

def handle_unit_conversions(df):
    """
    Handle various unit conversions and fix incorrectly recorded values:
    - Temperature (Celsius/Fahrenheit)
    - Hemoglobin/Hematocrit
    - Bilirubin (Total/Direct)
    """
    # Some values recorded in wrong column
    mask = (df['temp_F'] > 25) & (df['temp_F'] < 45)  # tempF close to 37deg
    if mask.any():
        df.loc[mask, 'temp_C'] = df.loc[mask, 'temp_F']
        df.loc[mask, 'temp_F'] = None

    # Values likely recorded in Fahrenheit but in Celsius column
    mask = df['temp_C'] > 70
    if mask.any():
        df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C']
        df.loc[mask, 'temp_C'] = None

    # Convert Celsius to Fahrenheit where missing
    mask = (~df['temp_C'].isna()) & (df['temp_F'].isna())
    if mask.any():
        df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C'] * 1.8 + 32

    # Convert Fahrenheit to Celsius where missing
    mask = (~df['temp_F'].isna()) & (df['temp_C'].isna())
    if mask.any():
        df.loc[mask, 'temp_C'] = (df.loc[mask, 'temp_F'] - 32) / 1.8

    # Handle Hemoglobin/Hematocrit conversions
    mask = (~df['hemoglobin'].isna()) & (df['hematocrit'].isna())
    if mask.any():
        df.loc[mask, 'hematocrit'] = (df.loc[mask, 'hemoglobin'] * 2.862) + 1.216

    mask = (~df['hematocrit'].isna()) & (df['hemoglobin'].isna())
    if mask.any():
        df.loc[mask, 'hemoglobin'] = (df.loc[mask, 'hematocrit'] - 1.216) / 2.862

    # Handle Bilirubin conversions
    mask = (~df['bilirubin_total'].isna()) & (df['bilirubin_direct'].isna())
    if mask.any():
        df.loc[mask, 'bilirubin_direct'] = (df.loc[mask, 'bilirubin_total'] * 0.6934) - 0.1752

    mask = (~df['bilirubin_direct'].isna()) & (df['bilirubin_total'].isna())
    if mask.any():
        df.loc[mask, 'bilirubin_total'] = (df.loc[mask, 'bilirubin_direct'] + 0.1752) / 0.6934

    return df

def sample_and_hold(df, vitalslab_hold):
    print('Performing sample and hold interpolation')
    temp = df.copy()
    
    # Process only columns that have hold times and are numeric
    cols_to_process = [col for col in vitalslab_hold if col in temp.columns]
    
    # Convert DataFrame to numpy array for faster processing
    data_array = temp.values
    col_indices = {col: temp.columns.get_loc(col) for col in cols_to_process}
    stay_id_idx = temp.columns.get_loc('stay_id')
    charttime_idx = temp.columns.get_loc('charttime')
    
    # Initialize progress bar for all columns
    bar = pyprind.ProgBar(len(cols_to_process), title='Processing columns')
    for col in cols_to_process:
        # Skip non-numeric columns
        if not np.issubdtype(temp[col].dtype, np.number):
            print(f"Skipping non-numeric column: {col}")
            continue
            
        col_idx = col_indices[col]
        hold_period = vitalslab_hold[col] * 3600
        
        last_charttime = 0
        last_value = np.nan  # Initialize with NaN instead of 0
        current_stay_id = data_array[0, stay_id_idx]
        
        for i in range(len(data_array)):
            # Reset trackers if stay_id changes
            if data_array[i, stay_id_idx] != current_stay_id:
                last_charttime = 0
                last_value = np.nan  # Reset to NaN instead of 0
                current_stay_id = data_array[i, stay_id_idx]
            
            # Update last known value if we have a valid measurement
            if not np.isnan(data_array[i, col_idx]):
                last_charttime = data_array[i, charttime_idx]
                last_value = data_array[i, col_idx]
            
            # Fill missing value if within hold period AND we have a valid last value
            elif (data_array[i, charttime_idx] - last_charttime) <= hold_period and not np.isnan(last_value):
                data_array[i, col_idx] = last_value
        bar.update()
    # Convert back to DataFrame
    result = pd.DataFrame(data_array, columns=temp.columns)
    # Preserve original dtypes
    for col in temp.columns:
        result[col] = result[col].astype(temp[col].dtype)
    
    return result

def combine_patient_data(patient_data, timestep=4, window_before=24, window_after=72):
    """
    Combines multiple measurement sources into a unified time series format with fixed timesteps.
    
    Args:
        patient_data: Dictionary containing patient measurements and metadata
        timestep: Size of timestep in hours (default: 4)
        window_before: Hours to include before onset time (default: up to 24)
        window_after: Hours to include after onset time (default: up to 72)
    """
    def process_antibiotics_data(start_time, end_time, abx_data):
        """Process antibiotics data within a time window"""
        # Create an explicit copy to avoid SettingWithCopyWarning
        abx_data = abx_data.copy()
        
        # Now it's safe to modify
        abx_data['stay_id'] = abx_data['stay_id'].astype('int64')
        
        # Get first antibiotics time (looking at all data, not just window)
        first_abx_time = abx_data['starttime'].min()
        
        # Find antibiotics active in the window
        mask = (abx_data['starttime'] <= end_time) & (abx_data['stoptime'] >= start_time)
        window_abx = abx_data[mask]
        
        return {
            'abx_given': 1 if len(window_abx) > 0 else 0,
            'hours_since_first_abx': (end_time - first_abx_time) / 3600 if first_abx_time else None,
            'num_abx': len(window_abx['drug'].unique()) if len(window_abx) > 0 else 0
        }
    
    def process_fluid_data(start_time, end_time, fluid_data):
        """Calculate fluid intake between two timepoints
        """
        if fluid_data is None or len(fluid_data) == 0:
            return 0, 0
        
        # For step fluids: entries that overlap with the window
        step_mask = (fluid_data['starttime'] < end_time) & (fluid_data['endtime'] >= start_time)
        step_fluids = fluid_data[step_mask]['amount'].sum()
        
        # For total fluids: entries that completed before the window end
        total_mask = fluid_data['endtime'] < end_time
        total_fluids = fluid_data[total_mask]['amount'].sum()
        
        return total_fluids, step_fluids

    def process_vasopressor_data(start_time, end_time, vaso_data):
        """Calculate vasopressor doses between two timepoints"""
        if vaso_data is None or len(vaso_data) == 0:
            return 0, 0
        
        # Find vasopressor entries that overlap with the window
        mask = (vaso_data['starttime'] <= end_time) & (vaso_data['endtime'] >= start_time)
        window_data = vaso_data[mask]
        
        if len(window_data) == 0:
            return 0, 0
        
        return window_data['rate_std'].median(), window_data['rate_std'].max()

    def process_urine_output(start_time, end_time, uo_data):
        """Calculate urine output between two timepoints"""
        if uo_data is None or len(uo_data) == 0:
            return 0, 0
        mask = (uo_data['charttime'] >= start_time) & (uo_data['charttime'] < end_time)
        step_uo = uo_data[mask]['value'].sum()  # Changed from 'urineoutput' to 'value'
        total_uo = uo_data[uo_data['charttime'] < end_time]['value'].sum()  # Changed from 'urineoutput' to 'value'
        return total_uo, step_uo
    
    def get_window_measurements(measurements, start_time, end_time):
        """Get all measurements within a time window"""
        if measurements is None or len(measurements) == 0:
            # For empty measurements, return dictionary with NaN values for all columns
            # and window midpoint as charttime
            dummy_row = {col: np.nan for col in measurements.columns 
                         if col not in ['stay_id']}
            dummy_row['charttime'] = (start_time + end_time) / 2
            return dummy_row
        
        mask = (measurements['charttime'] >= start_time) & (measurements['charttime'] < end_time)
        window_data = measurements[mask]
        
        if len(window_data) == 0:
            # If no data in this window, return NaN for all columns except charttime
            dummy_row = {col: np.nan for col in measurements.columns 
                         if col not in ['stay_id']}
            dummy_row['charttime'] = (start_time + end_time) / 2
            return dummy_row
        
        # Use mean for aggregation, but don't fill missing values
        return window_data.mean(axis=0, skipna=True).to_dict()
    
    # Initialize output dataframe with required columns
    columns = [
        'timestep', 'stay_id', 'timestamp',
        # Demographics columns
        'gender', 'age', 'charlson_comorbidity_index', 're_admission', 'los',
        'morta_hosp', 'morta_90', 
        # Clinical measurements
        *[col for col in patient_data['measurements'].columns if col not in ['timestep', 'stay_id', 'timestamp']],
        # Fluid balance
        'fluid_total', 'fluid_step', 'uo_total', 'uo_step', 'balance',
        # Vasopressors
        'vaso_median', 'vaso_max',
        # Antibiotics
        'abx_given', 'hours_since_first_abx', 'num_abx'
    ]
    
    processed_data = []
    
    # Get actual data time range for this patient
    patient_times = sorted(list(set(
        patient_data['measurements']['charttime'].values
    )))
    
    if not patient_times:  # Skip if no data available
        return None
    
    # Get first and last timestamp with valid data
    first_time = max(patient_times[0], patient_data['start_time'] - window_before * 3600)
    last_time = min(patient_times[-1], patient_data['start_time'] + window_after * 3600)
    
    # Calculate number of complete timesteps
    total_seconds = last_time - first_time
    total_hours = total_seconds / 3600
    num_timesteps = math.ceil(total_hours / timestep)
    
    # Process each timestep
    for timestep_idx in range(num_timesteps):
        # Calculate window boundaries in epoch time
        window_start = first_time + (timestep_idx * timestep * 3600)
        window_end = window_start + (timestep * 3600)
        
        # Skip if window is outside available data range
        if window_end < first_time or window_start > last_time:
            continue
        
        # Get measurements within time window
        measurements = get_window_measurements(
            patient_data['measurements'],
            window_start,
            window_end
        )
        
        if measurements is not None:
            # Process each data type
            fluid_total, fluid_step = process_fluid_data(
                window_start, 
                window_end, 
                patient_data['fluid']
            )
            
            vaso_median, vaso_max = process_vasopressor_data(
                window_start,
                window_end,
                patient_data['vasopressors']
            )
            
            uo_total, uo_step = process_urine_output(
                window_start,
                window_end,
                patient_data['urine_output']
            )

            abx_info = process_antibiotics_data(
                window_start,
                window_end,
                patient_data['antibiotics']
            )
            
            # Combine all data for this timestep
            timestep_data = {
                'timestep': timestep_idx + 1,
                'stay_id': patient_data['stay_id'],
                'timestamp': window_start,  # Keep as epoch time
                # Add demographics
                **patient_data['demographics'],
                # Add measurements
                **measurements,
                # Add fluid balance
                'fluid_total': fluid_total,
                'fluid_step': fluid_step,
                'uo_total': uo_total,
                'uo_step': uo_step,
                'balance': fluid_total - uo_total,
                'vaso_median': vaso_median,
                'vaso_max': vaso_max,
                **abx_info
            }
            
            processed_data.append(timestep_data)
    
    # IMPORTANT: Create DataFrame and explicitly set NaNs for missing values
    df = pd.DataFrame(processed_data, columns=columns)
    
    return df

def standardize_patient_trajectories(init_traj, data_dict, timestep=4, window_before=24, window_after=72):
    print('Processing all patients with fixed time windows')
    all_patient_data = []
    
    
    # Group by stay_id to process each patient
    bar = pyprind.ProgBar(len(init_traj.groupby('stay_id')))
    for stay_id, patient_traj in init_traj.groupby('stay_id'):
        # Use onset time from onset data instead of first measurement
        start_time = data_dict['onset'].loc[
            data_dict['onset']['stay_id'] == stay_id, 
            'onset_time'
        ].iloc[0]
        
        # Collect all data for this patient
        patient_data = {
            'stay_id': stay_id,
            'start_time': start_time,  # Using onset time as start time
            'measurements': patient_traj,
            'demographics': data_dict['demog'].loc[
                data_dict['demog']['stay_id'] == stay_id
            ].iloc[0].to_dict(),
            'fluid': data_dict['fluid'][
                data_dict['fluid']['stay_id'] == stay_id
            ],
            'vasopressors': data_dict['vaso'][
                data_dict['vaso']['stay_id'] == stay_id
            ],
            'urine_output': data_dict['UO'][
                data_dict['UO']['stay_id'] == stay_id
            ],
            'antibiotics': data_dict['abx'][
                data_dict['abx']['stay_id'] == stay_id
            ]
        }
        
        # Process this patient's data using the existing combine_patient_measurements function
        processed_patient = combine_patient_data(
            patient_data, 
            timestep=timestep,
            window_before=window_before,
            window_after=window_after
        )
        all_patient_data.append(processed_patient)
        bar.update()
    
    # Combine all processed patient data
    return pd.concat(all_patient_data, ignore_index=True)


def fixgaps(x: np.ndarray) -> np.ndarray:
    """Linearly interpolates gaps (NaN values) in a time series.
    
    Interpolates over NaN values in the input array, ignoring leading and trailing NaN values.
    The interpolation is done linearly between the nearest non-NaN values.
    
    Args:
        x: Input array containing the time series data with NaN values
        
    Returns:
        Array with NaN values interpolated, except for leading/trailing NaN values
    """
    # Make a copy to avoid modifying input
    y = np.copy(x)
    
    # Find NaN and non-NaN indices
    nan_mask = np.isnan(x)
    valid_indices = np.arange(len(x))[~nan_mask]
    
    if len(valid_indices) == 0:
        return y
        
    # Ignore leading/trailing NaN values
    nan_mask[:valid_indices[0]] = False
    nan_mask[valid_indices[-1]+1:] = False
    
    # Interpolate NaN values using valid data points
    y[nan_mask] = interp1d(
        valid_indices,
        x[valid_indices]
    )(np.arange(len(x))[nan_mask])
    
    return y

def handle_missing_values(df, missing_threshold=0.8):
    """Handle missing values through interpolation and KNN imputation
    
    Args:
        df: DataFrame containing patient measurements
        missing_threshold: Threshold for dropping columns with missing values (default: 0.7)
        
    Returns:
        DataFrame with missing values handled
    """
    print('Handling missing values...')
    
    # Get columns that need imputation (exclude non-numeric/demographic columns)
    measurement_cols = [col for col in df.columns if col not in [
        'timestep', 'stay_id', 'timestamp', 'gender', 'age', 
        'charlson_comorbidity_index', 're_admission', 'los',
        'morta_hosp', 'morta_90', 'fluid_total', 'fluid_step',
        'uo_total', 'uo_step', 'balance', 'vaso_median', 'vaso_max',
        'abx_given', 'hours_since_first_abx', 'num_abx'
    ]]
    # Print missingness statistics before imputation for all columns
    print("\nMissingness statistics before imputation:")
    print("-" * 50)
    
    # Get all numeric columns, including those we'll exclude
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Split into measurement cols and excluded cols
    excluded_numeric_cols = [col for col in all_numeric_cols if col not in measurement_cols]
    
    # Calculate missingness for both sets
    miss_stats_meas = df[measurement_cols].isna().sum() / len(df)
    miss_stats_excl = df[excluded_numeric_cols].isna().sum() / len(df)
    
    # Sort both by missingness
    miss_stats_meas = miss_stats_meas.sort_values(ascending=False)
    miss_stats_excl = miss_stats_excl.sort_values(ascending=False)
    
    print("Measurement columns to be imputed:")
    print(f"{'Variable':<30} {'Missing %':>10}")
    print("-" * 50)
    for var, miss_pct in miss_stats_meas.items():
        print(f"{var:<30} {miss_pct:>10.1%}")
    
    print("\nExcluded numeric columns:")
    print(f"{'Variable':<30} {'Missing %':>10}")
    print("-" * 50)
    for var, miss_pct in miss_stats_excl.items():
        print(f"{var:<30} {miss_pct:>10.1%}")
    print("-" * 50)
    print()
    # Calculate missingness per column
    miss = df[measurement_cols].isna().sum() / len(df)
    
    # Drop columns with missing values above threshold
    cols_to_keep = miss[miss < missing_threshold].index
    df = df[df.columns[~df.columns.isin(measurement_cols)].tolist() + cols_to_keep.tolist()]
    
    # Linear interpolation for columns with <5% missing
    low_missing_cols = miss[(miss > 0) & (miss < 0.05)].index
    for col in low_missing_cols:
        df[col] = pd.Series(fixgaps(df[col].values))
        
    # KNN imputation for remaining missing values
    cols_for_knn = [c for c in cols_to_keep if c not in low_missing_cols]
    if cols_for_knn:
        ref = df[cols_for_knn].values
        
        # Process in chunks of 10K rows
        chunk_size = 9999
        bar = pyprind.ProgBar(len(range(0, len(df), chunk_size)))
        for i in range(0, len(df), chunk_size):
            chunk_end = min(i + chunk_size, len(df))
            ref[i:chunk_end,:] = KNN(k=1, verbose=0).fit_transform(ref[i:chunk_end,:])
            bar.update()
            
        df[cols_for_knn] = ref
        
    return df

def calculate_derived_variables(df):
    """Calculate derived variables like P/F ratio, Shock Index, SOFA, and SIRS scores
    
    Args:
        df: DataFrame containing patient measurements
    Returns:
        DataFrame with added derived variables
    """
    print('Computing derived variables: P/F ratio, Shock Index, SOFA, SIRS...')
    
    # Make a copy to avoid modifying input
    df = df.copy()
    
    # Fix demographic variables
    df['gender'] = df['gender'] - 1
    df.loc[df['age'] > 150, 'age'] = 91.4
    
    # Fix mechanical ventilation
    df['mechvent'] = df['mechvent'].fillna(0)
    df.loc[df['mechvent'] > 0, 'mechvent'] = 1
    
    # Fix Charlson Comorbidity Index
    df['charlson_comorbidity_index'] = df['charlson_comorbidity_index'].fillna(
        df['charlson_comorbidity_index'].median()
    )
    
    # Fix vasopressor doses
    df['vaso_median'] = df['vaso_median'].fillna(0)
    df['vaso_max'] = df['vaso_max'].fillna(0)
    
    # Calculate P/F ratio
    df['pf_ratio'] = df['arterial_o2_pressure'] / (df['fio2'] / 100)
    
    # Calculate Shock Index
    df['shock_index'] = df['heart_rate'] / df['sbp_arterial']
    df.loc[np.isinf(df['shock_index']), 'shock_index'] = np.nan
    df['shock_index'] = df['shock_index'].fillna(df['shock_index'].mean())
    
    # Calculate SOFA score components
    def calc_sofa_resp(row):
        pf = row['pf_ratio']
        if pd.isna(pf): return 0
        if pf >= 400: return 0
        if pf >= 300: return 1
        if pf >= 200: return 2
        if pf >= 100: return 3
        return 4
    
    def calc_sofa_coag(row):
        plt = row['platelets']
        if pd.isna(plt): return 0
        if plt >= 150: return 0
        if plt >= 100: return 1
        if plt >= 50: return 2
        if plt >= 20: return 3
        return 4
    
    def calc_sofa_liver(row):
        bili = row['bilirubin_total']
        if pd.isna(bili): return 0
        if bili < 1.2: return 0
        if bili < 2.0: return 1
        if bili < 6.0: return 2
        if bili < 12.0: return 3
        return 4
    
    def calc_sofa_cv(row):
        map_ = row['map']
        vaso = row['vaso_max']
        if pd.isna(map_) and pd.isna(vaso): return 0
        if not pd.isna(map_) and map_ >= 70: return 0
        if not pd.isna(map_) and map_ >= 65: return 1
        if not pd.isna(map_) and map_ < 65: return 2
        if not pd.isna(vaso) and vaso <= 0.1: return 3
        if not pd.isna(vaso) and vaso > 0.1: return 4
        return 0
    
    def calc_sofa_cns(row):
        gcs = row['gcs']
        if pd.isna(gcs): return 0
        if gcs > 14: return 0
        if gcs > 12: return 1
        if gcs > 9: return 2
        if gcs > 5: return 3
        return 4
    
    def calc_sofa_renal(row):
        cr = row['creatinine']
        uo = row['uo_step']
        if pd.isna(cr) and pd.isna(uo): return 0
        if not pd.isna(cr):
            if cr < 1.2: return 0
            if cr < 2.0: return 1
            if cr < 3.5: return 2
            if cr < 5.0: return 3
            return 4
        if not pd.isna(uo):
            if uo >= 84: return 0
            if uo >= 34: return 3
            return 4
        return 0
    
    # Calculate SOFA score
    df['sofa_resp'] = df.apply(calc_sofa_resp, axis=1)
    print("SOFA Respiratory score distribution:")
    print(df['sofa_resp'].value_counts().sort_index())
    
    df['sofa_coag'] = df.apply(calc_sofa_coag, axis=1)
    print("\nSOFA Coagulation score distribution:")
    print(df['sofa_coag'].value_counts().sort_index())
    
    df['sofa_liver'] = df.apply(calc_sofa_liver, axis=1)
    print("\nSOFA Liver score distribution:")
    print(df['sofa_liver'].value_counts().sort_index())
    
    df['sofa_cv'] = df.apply(calc_sofa_cv, axis=1)
    print("\nSOFA Cardiovascular score distribution:")
    print(df['sofa_cv'].value_counts().sort_index())
    
    df['sofa_cns'] = df.apply(calc_sofa_cns, axis=1)
    print("\nSOFA CNS score distribution:")
    print(df['sofa_cns'].value_counts().sort_index())
    
    df['sofa_renal'] = df.apply(calc_sofa_renal, axis=1)
    print("\nSOFA Renal score distribution:")
    print(df['sofa_renal'].value_counts().sort_index())
    
    df['sofa_score'] = (df['sofa_resp'] + df['sofa_coag'] + df['sofa_liver'] + 
                       df['sofa_cv'] + df['sofa_cns'] + df['sofa_renal'])
    
    # Calculate SIRS score
    def calc_sirs(row):
        score = 0
        # Temperature criterion
        if not pd.isna(row['temp_C']):
            if row['temp_C'] >= 38 or row['temp_C'] <= 36:
                score += 1
        # Heart rate criterion
        if not pd.isna(row['heart_rate']) and row['heart_rate'] > 90:
            score += 1
        # Respiratory criterion
        if (not pd.isna(row['respiratory_rate']) and row['respiratory_rate'] >= 20) or \
           (not pd.isna(row['arterial_co2_pressure']) and row['arterial_co2_pressure'] <= 32):
            score += 1
        # WBC criterion
        if not pd.isna(row['wbc']):
            if row['wbc'] >= 12 or row['wbc'] < 4:
                score += 1
        return score
    
    df['sirs_score'] = df.apply(calc_sirs, axis=1)
    
    return df

def apply_exclusion_criteria(df):
    """Apply exclusion criteria for the sepsis cohort
    
    Excludes patients based on:
    1. Extreme urine output (>12000)
    2. Extreme fluid intake (>10000)
    3. Early deaths from possible withdrawals (death within 24h of ICU admission)
    4. None-sepsis patients (SOFA score < 2)
    """
    print('Applying exclusion criteria')
    
    # Keep track of excluded patients for logging
    initial_patients = len(df['stay_id'].unique())
    excluded_counts = {}
    
    # Exclude patients with extreme UO
    mask = df['uo_step'] > 12000
    excluded_stays = df[mask]['stay_id'].unique()
    df = df[~df['stay_id'].isin(excluded_stays)]
    excluded_counts['extreme_uo'] = len(excluded_stays)
    
    # Exclude patients with extreme fluid intake
    mask = df['fluid_step'] > 10000
    excluded_stays = df[mask]['stay_id'].unique()
    df = df[~df['stay_id'].isin(excluded_stays)]
    excluded_counts['extreme_fluid'] = len(excluded_stays)
    
    # Exclude early deaths (within 24h of ICU admission)
    # Group by stay_id and get first timestamp for each patient
    patient_starts = df.groupby('stay_id')['timestamp'].min()
    
    early_deaths = []
    for stay_id in df['stay_id'].unique():
        patient_data = df[df['stay_id'] == stay_id].iloc[0]
        if patient_data['morta_hosp'] == 1:  # if patient died in hospital
            # Get first timestamp for this patient
            start_time = patient_starts[stay_id]
            # Get last timestamp for this patient
            end_time = df[df['stay_id'] == stay_id]['timestamp'].max()
            time_to_death = (end_time - start_time) / 3600  # convert to hours
            
            if time_to_death <= 24:  # if death within 24h
                early_deaths.append(stay_id)
    
    df = df[~df['stay_id'].isin(early_deaths)]
    excluded_counts['early_death'] = len(early_deaths)

    # Exclude non-sepsis patients (SOFA score < 2)
    non_sepsis_stays = []
    for stay_id in df['stay_id'].unique():
        patient_data = df[df['stay_id'] == stay_id]
        # Check if patient ever had SOFA >= 2 during their stay
        if not (patient_data['sofa_score'] >= 2).any():
            non_sepsis_stays.append(stay_id)
    
    df = df[~df['stay_id'].isin(non_sepsis_stays)]
    excluded_counts['non_sepsis'] = len(non_sepsis_stays)
    
    # Print exclusion statistics
    final_patients = len(df['stay_id'].unique())
    print("\nExclusion Statistics:")
    print("-" * 50)
    print(f"Initial patient count: {initial_patients}")
    for reason, count in excluded_counts.items():
        print(f"Excluded due to {reason}: {count}")
    print(f"Final patient count: {final_patients}")
    print(f"Total excluded: {initial_patients - final_patients}")
    print("-" * 50)
    
    return df

def add_sepsis_flag(df):
    """Add sepsis flag to patient trajectories.
    
    Sepsis criteria:
    1. Sepsis-3 definition:
        - Positive culture data or antibiotic administration
        - SOFA score ≥ 2, assuming all baseline SOFA is 0
    
    Flag values:
    0 = No sepsis
    1 = Sepsis onset identified
    2 = Censored (post first sepsis)
    
    Args:
        df: DataFrame containing patient measurements
    Returns:
        DataFrame with added sepsis column
    """
    print('Adding sepsis flags to trajectories')
    
    # Initialize sepsis column
    df['sepsis'] = 0
    
    # Process each patient
    for stay_id in df['stay_id'].unique():
        patient_df = df[df['stay_id'] == stay_id].copy()
        
        # Sort by timestamp to ensure chronological order
        patient_df = patient_df.sort_values('timestamp')
        
        # Check for SOFA ≥ 2
        sepsis_condition = patient_df['sofa_score'] >= 2
        
        # Find first occurrence of sepsis
        sepsis_idx = sepsis_condition.idxmax() if sepsis_condition.any() else None
        
        if sepsis_idx is not None:
            # Mark sepsis onset
            df.loc[sepsis_idx, 'sepsis'] = 1
            
            # Mark all subsequent timestamps for this patient as censored
            subsequent_mask = (df['stay_id'] == stay_id) & \
                            (df.index > sepsis_idx)
            df.loc[subsequent_mask, 'sepsis'] = 2
    
    # Print statistics
    total_patients = len(df['stay_id'].unique())
    sepsis_patients = len(df[df['sepsis'] == 1]['stay_id'].unique())
    
    print("\nSepsis Statistics:")
    print("-" * 50)
    print(f"Total patients: {total_patients}")
    print(f"Patients developing sepsis: {sepsis_patients} ({sepsis_patients/total_patients*100:.1f}%)")
    print(f"Timesteps with sepsis onset: {(df['sepsis'] == 1).sum()}")
    print(f"Censored timesteps: {(df['sepsis'] == 2).sum()}")
    print("-" * 50)
    
    return df

def add_septic_shock_flag(df):
    """Add septic shock flag to patient trajectories.
    
    Septic shock criteria:
    1. Adequate fluid resuscitation (defined as minimum fluid intake over previous 12 hours)
    2. After adequate fluids:
        - MAP < 65 mmHg AND
        - Requires vasopressors (indicated by hypotension despite fluids) AND
        - Lactate > 2 mmol/L
    
    Flag values:
    0 = No septic shock
    1 = Septic shock identified
    2 = Censored (post first shock)
    
    Args:
        df: DataFrame containing patient measurements
    Returns:
        DataFrame with added septic_shock column
    """
    print('Adding septic shock flags to trajectories')
    
    # Initialize septic shock column
    df['septic_shock'] = 0
    
    # Define thresholds
    FLUID_WINDOW = 12  # hours
    TIMESTEP_SIZE = 4  # hours per timestep
    WINDOW_STEPS = max(1, FLUID_WINDOW // TIMESTEP_SIZE)  # number of timesteps for 12 hours
    MIN_FLUID_THRESHOLD = 2000  # mL in 12 hours
    MAP_THRESHOLD = 65
    LACTATE_THRESHOLD = 2  # mmol/L
    
    print(f"Using rolling window of {WINDOW_STEPS} timesteps ({WINDOW_STEPS * TIMESTEP_SIZE} hours)")
    print(f"Minimum fluid threshold: {MIN_FLUID_THRESHOLD}mL over {FLUID_WINDOW} hours")
    
    # Process each patient
    for stay_id in df['stay_id'].unique():
        patient_df = df[df['stay_id'] == stay_id].copy()
        
        # Calculate rolling fluid sum for previous 12 hours
        # First sort by timestamp to ensure correct rolling calculation
        patient_df = patient_df.sort_values('timestamp')
        rolling_fluid = patient_df['fluid_step'].rolling(
            window=WINDOW_STEPS, 
            min_periods=1
        ).sum()
        
        # Check conditions for septic shock
        shock_conditions = (
            (rolling_fluid >= MIN_FLUID_THRESHOLD) &  # adequate fluids
            (patient_df['map'] < MAP_THRESHOLD) &     # hypotension
            (patient_df['lactic_acid'] > LACTATE_THRESHOLD)  # elevated lactate
        )
        
        # Find first occurrence of shock
        shock_idx = shock_conditions.idxmax() if shock_conditions.any() else None
        
        if shock_idx is not None:
            # Mark shock onset
            df.loc[shock_idx, 'septic_shock'] = 1
            
            # Mark all subsequent timestamps for this patient as censored
            subsequent_mask = (df['stay_id'] == stay_id) & \
                            (df.index > shock_idx)
            df.loc[subsequent_mask, 'septic_shock'] = 2
    
    # Print statistics
    total_patients = len(df['stay_id'].unique())
    shock_patients = len(df[df['septic_shock'] == 1]['stay_id'].unique())
    
    print("\nSeptic Shock Statistics:")
    print("-" * 50)
    print(f"Total patients: {total_patients}")
    print(f"Patients developing shock: {shock_patients} ({shock_patients/total_patients*100:.1f}%)")
    print(f"Timesteps with shock onset: {(df['septic_shock'] == 1).sum()}")
    print(f"Censored timesteps: {(df['septic_shock'] == 2).sum()}")
    print("-" * 50)
    
    return df

def main():
    args = parse_args()
    
    # Load all required data
    data = load_processed_files()
    measurements, code_to_concept, hold_times = load_measurement_mappings()
    
    # Load onset data 
    onset = data['onset']
    
    # Sample subjects if sample_size is specified
    if args.sample_size is not None:
        print(f'Sampling {args.sample_size} subjects for testing')
        onset = onset.sample(n=args.sample_size, random_state=42)
    
    # Process each patient
    print('Processing patient timeseries data')
    all_patient_data = []
    
    # Check if processed file already exists
    output_path = f"{args.output_dir}/patient_timeseries_v4.csv"
    
    bar = pyprind.ProgBar(len(onset))
    for _, row in onset.iterrows():
        icustayid = row['stay_id']
        onset_time = row['onset_time']
        if onset_time > 0:  # if we have a flag time
            patient_df = process_patient_measurements(
                data, measurements, code_to_concept,
                icustayid, onset_time,
                winb4=args.window_before,
                winaft=args.window_after
            )
            if patient_df is not None:
                all_patient_data.append(patient_df)
        bar.update()

    # Combine all patient data
    init_traj = pd.concat(all_patient_data, ignore_index=True)
    

    # Handle outliers
    init_traj = handle_outliers(init_traj)

    # Estimate GCS from RASS
    init_traj = estimate_gcs_from_rass(init_traj)
    
    # Estimate FiO2 from O2 flow
    init_traj = estimate_fio2(init_traj)

    # Handle unit conversions
    init_traj = handle_unit_conversions(init_traj)

    # Sample and hold (forward fill)
    init_traj = sample_and_hold(init_traj, hold_times) 

    

    # Combine all patients
    init_traj = standardize_patient_trajectories(
        init_traj, 
        data,
        timestep=args.timestep,
        window_before=args.window_before,
        window_after=args.window_after
    )

    
    
    # Handle missing values
    init_traj = handle_missing_values(init_traj, args.missing_threshold)

    # Final check before returning
    print(f"FiO2 zeros after handling missing values: {(init_traj['fio2'] == 0).sum()}")

    # Calculate derived variables (SOFA, SIRS, etc.)
    init_traj = calculate_derived_variables(init_traj)    
    #Apply exclusion criteria at the end
    init_traj = apply_exclusion_criteria(init_traj)

    # Add septic shock flags
    init_traj = add_septic_shock_flag(init_traj)

    # Add sepsis flags
    init_traj = add_sepsis_flag(init_traj)

    # Print missingness statistics
    missing_pct = (init_traj.isna().sum() / len(init_traj)) * 100
    print("\nMissing value percentages:")
    for col, pct in missing_pct.sort_values(ascending=False).items():
        if pct > 0:
            print(f"{col}: {pct:.1f}%")
    
    # Save processed data
    output_path = f"{args.output_dir}/patient_timeseries_v4.csv"
    init_traj.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    

if __name__ == "__main__":
    main()

