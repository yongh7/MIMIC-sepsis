import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import pyprind


class TimeSeriesDataProcessor:
    def __init__(self, 
                 features: List[str],
                 task: str,  # 'mortality', 'los', 'mechvent', 'septic_shock'
                 window_size: int = None,
                 prediction_horizon: int = None):  # Made optional with None default
        self.features = features
        self.task = task
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets based on the task type
        Returns:
            features: Array of feature values
            targets: Array of target values
        """
        if self.task == 'morta_hosp':
            return self._prepare_mortality_data(df)
        elif self.task == 'los':
            return self._prepare_los_data(df)
        elif self.task == 'mechvent':
            return self._prepare_mechvent_data(df)
        elif self.task == 'septic_shock':
            return self._prepare_septic_shock_data(df)
        elif self.task == 'sepsis':
            return self._prepare_sepsis_data(df)
        elif self.task == 'vasopressor':
            return self._prepare_vasopressor_data(df)
        else:
            raise ValueError(f"Unknown task type: {self.task}")
        
    def _prepare_sepsis_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For sepsis prediction:
        - Use sliding windows of fixed size
        - Predict sepsis onset within prediction horizon (in timesteps)
        - Handles irregular timesteps
        """
        if self.prediction_horizon is None:
            raise ValueError("prediction_horizon must be set for sepsis prediction")
        
        grouped = df.groupby('stay_id')
        features, targets = [], []
        
        print("Processing sepsis data...")
        bar = pyprind.ProgBar(len(grouped))
        for _, group in grouped:
            # Sort by timestep to ensure temporal order
            group = group.sort_values('timestep')
            
            # Create windows
            for i in range(len(group) - self.window_size - self.prediction_horizon + 1):
                window = group.iloc[i:i + self.window_size]
                if len(window) == self.window_size:
                    features.append(window[self.features].values)
                    
                    # Check if sepsis occurs within prediction horizon
                    future_window = group.iloc[i + self.window_size:
                                             i + self.window_size + self.prediction_horizon]
                    sepsis_occurs = future_window[self.task].max() > 0
                    targets.append(1 if sepsis_occurs else 0)
            bar.update()
        
        return np.array(features), np.array(targets)
    
    
    def _prepare_mortality_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For mortality prediction:
        - Use fixed time window from admission
        - Single binary target per stay
        """
        grouped = df.groupby('stay_id')
        features, targets = [], []
        
        print("Processing mortality data...")
        bar = pyprind.ProgBar(len(grouped))
        for _, group in grouped:
            window_data = group.head(self.window_size)[self.features].values
            if len(window_data) == self.window_size:  # Only use complete windows
                features.append(window_data)
                targets.append(group[self.task].iloc[-1])
            bar.update()
        
        return np.array(features), np.array(targets)
    
    def _prepare_los_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For length of stay prediction:
        - Use fixed observation window
        - Predict total LOS using data from observation window
        - Exclude cases where LOS < observation window
        """
        # Temporary fix: Remove stays with NaN 'los' values
        df = df.dropna(subset=[self.task])
        
        grouped = df.groupby('stay_id')
        features, targets = [], []
        
        print("Processing length of stay data...")
        bar = pyprind.ProgBar(len(grouped))
        for _, group in grouped:
            # Sort by timestep to ensure temporal order
            group = group.sort_values('timestep')
            
            # Only include if we have enough data for the observation window
            if len(group) >= self.window_size:
                window_data = group.head(self.window_size)[self.features].values
                if len(window_data) == self.window_size:
                    features.append(window_data)
                    # Use the LOS value from the data (assuming it's in the self.task column)
                    total_los = group[self.task].iloc[-1]  # Get LOS from the last row
                    targets.append(total_los)
            bar.update()
        
        return np.array(features), np.array(targets)
    
    def _prepare_mechvent_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For mechanical ventilation prediction:
        - Use sliding windows of fixed size
        - Predict mechanical ventilation onset within prediction horizon
        - Handles irregular timesteps
        """
        if self.prediction_horizon is None:
            raise ValueError("prediction_horizon must be set for mechanical ventilation prediction")
        
        grouped = df.groupby('stay_id')
        features, targets = [], []
        
        print("Processing mechanical ventilation data...")
        bar = pyprind.ProgBar(len(grouped))
        for _, group in grouped:
            # Sort by timestep to ensure temporal order
            group = group.sort_values('timestep')
            
            # Create windows
            for i in range(len(group) - self.window_size - self.prediction_horizon + 1):
                window = group.iloc[i:i + self.window_size]
                if len(window) == self.window_size:
                    features.append(window[self.features].values)
                    
                    # Check if mechanical ventilation occurs within prediction horizon
                    future_window = group.iloc[i + self.window_size:
                                             i + self.window_size + self.prediction_horizon]
                    vent_occurs = future_window[self.task].max() > 0
                    targets.append(1 if vent_occurs else 0)
            bar.update()
        
        return np.array(features), np.array(targets)
    
    def _prepare_septic_shock_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For septic shock prediction:
        - Use sliding windows of fixed size
        - Predict septic shock onset within prediction horizon
        - Handles irregular timesteps
        """
        if self.prediction_horizon is None:
            raise ValueError("prediction_horizon must be set for septic shock prediction")
        
        grouped = df.groupby('stay_id')
        features, targets = [], []
        
        print("Processing septic shock data...")
        bar = pyprind.ProgBar(len(grouped))
        for _, group in grouped:
            # Sort by timestep to ensure temporal order
            group = group.sort_values('timestep')
            
            # Create windows
            for i in range(len(group) - self.window_size - self.prediction_horizon + 1):
                window = group.iloc[i:i + self.window_size]
                if len(window) == self.window_size:
                    features.append(window[self.features].values)
                    
                    # Check if septic shock occurs within prediction horizon
                    future_window = group.iloc[i + self.window_size:
                                             i + self.window_size + self.prediction_horizon]
                    shock_occurs = future_window[self.task].max() > 0
                    targets.append(1 if shock_occurs else 0)
            bar.update()
        
        return np.array(features), np.array(targets)
    
    def _prepare_vasopressor_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For vasopressor requirement prediction:
        - Use sliding windows of fixed size
        - Predict vasopressor requirement within prediction horizon
        - Handles irregular timesteps
        - Creates binary target based on vaso_median or vaso_max > 0
        """
        if self.prediction_horizon is None:
            raise ValueError("prediction_horizon must be set for vasopressor prediction")
        
        grouped = df.groupby('stay_id')
        features, targets = [], []
        
        print("Processing vasopressor requirement data...")
        bar = pyprind.ProgBar(len(grouped))
        for _, group in grouped:
            # Sort by timestep to ensure temporal order
            group = group.sort_values('timestep')
            
            # Create windows
            for i in range(len(group) - self.window_size - self.prediction_horizon + 1):
                window = group.iloc[i:i + self.window_size]
                if len(window) == self.window_size:
                    features.append(window[self.features].values)
                    
                    # Check if vasopressor is required within prediction horizon
                    future_window = group.iloc[i + self.window_size:
                                             i + self.window_size + self.prediction_horizon]
                    
                    # Check if either vaso_median or vaso_max is > 0
                    vaso_required = (future_window['vaso_median'].max() > 0) or (future_window['vaso_max'].max() > 0)
                    targets.append(1 if vaso_required else 0)
            bar.update()
        
        return np.array(features), np.array(targets)
    
    def normalize_features(self, train_data: np.ndarray, val_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize features using training data statistics.
        All tasks now use the same normalization approach.
        """
        scaler = StandardScaler()
        
        # Reshape to 2D for scaling
        train_shape = train_data.shape
        val_shape = val_data.shape
        
        # Reshape to (n_samples * n_timesteps, n_features) if 3D
        if len(train_shape) == 3:
            train_reshaped = train_data.reshape(-1, train_shape[-1])
            val_reshaped = val_data.reshape(-1, val_shape[-1])
        else:
            train_reshaped = train_data
            val_reshaped = val_data
            
        # Fit on training data and transform both
        train_normalized = scaler.fit_transform(train_reshaped)
        val_normalized = scaler.transform(val_reshaped)
        
        # Reshape back to original shape if necessary
        if len(train_shape) == 3:
            train_normalized = train_normalized.reshape(train_shape)
            val_normalized = val_normalized.reshape(val_shape)
        
        self.scalers['features'] = scaler
        return train_normalized, val_normalized 
    

