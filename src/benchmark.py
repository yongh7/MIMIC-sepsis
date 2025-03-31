import argparse
import pandas as pd
import numpy as np
from data_processor import TimeSeriesDataProcessor
from linear_model import LinearTimeSeriesModel
from lstm_model import LSTMModel
from transformer_model import TimeSeriesTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import torch


# Set random seeds at the top of your file
def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across all libraries used"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    # Make TensorFlow deterministic
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set NumPy print options for consistent output
    np.set_printoptions(precision=3, suppress=True)
    
    print(f"Random seeds set to {seed} for reproducibility")



def load_data(data_path: str) -> pd.DataFrame:
    """Load the patient timeseries data"""
    print("Loading patient timeseries data...")
    return pd.read_csv(data_path)

def get_feature_columns(df: pd.DataFrame, target_col: str) -> list:
    """Get feature columns by excluding specific columns"""
    exclude_columns = [
        'morta_hosp',  # future information, exclude to avoid data leakage
        'morta_90',    # future information, exclude to avoid data leakage
        'timestep',   # temporal index
        'stay_id',    # identifier
        target_col,   # target variable
        'los',      # future information, exclude to avoid data leakage
        'mechvent' if target_col != 'mechvent' else None,
        'septic_shock' if target_col != 'septic_shock' else None,
        'vasopressor' if target_col != 'vasopressor' else None,
        'vaso_median' if target_col != 'vasopressor' else None,
        'vaso_max' if target_col != 'vasopressor' else None,
    ]
    return [col for col in df.columns if col not in exclude_columns]

def split_data(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and validation sets"""
    patient_ids = df['stay_id'].unique()
    train_size = int(len(patient_ids) * train_ratio)
    
    # Use np.random.RandomState with fixed seed for shuffling
    rs = np.random.RandomState(42)
    shuffled_ids = patient_ids.copy()
    rs.shuffle(shuffled_ids)
    
    train_ids = shuffled_ids[:train_size]
    val_ids = shuffled_ids[train_size:]
    
    train_df = df[df['stay_id'].isin(train_ids)]
    val_df = df[df['stay_id'].isin(val_ids)]
    
    print(f"\nTrain set: {len(train_ids)} patients")
    print(f"Val set: {len(val_ids)} patients")
    
    return train_df, val_df

def evaluate_model(targets: np.ndarray, predictions: np.ndarray, task_type: str) -> Dict[str, float]:
    """Calculate performance metrics based on task type"""
    if task_type == 'classification':
        # Convert probability predictions to binary predictions using 0.5 threshold
        binary_predictions = (predictions >= 0.5).astype(int)
        accuracy = np.mean(binary_predictions == targets)
        return {
            'auroc': roc_auc_score(targets, predictions),
            'auprc': average_precision_score(targets, predictions),
            'accuracy': accuracy
        }
    else:  # regression
        mse = np.mean((targets - predictions) ** 2)
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': np.mean(np.abs(targets - predictions))
        }

def get_baseline_metrics(train_targets: np.ndarray, val_targets: np.ndarray, task_type: str) -> Dict[str, Dict[str, float]]:
    """Calculate baseline performance based on task type"""
    if task_type == 'classification':
        majority_pred = train_targets.mean() > 0.5
        train_baseline = np.ones_like(train_targets) * majority_pred
        val_baseline = np.ones_like(val_targets) * majority_pred
    else:  # regression
        mean_pred = np.mean(train_targets)
        train_baseline = np.ones_like(train_targets) * mean_pred
        val_baseline = np.ones_like(val_targets) * mean_pred
    
    return {
        'train': evaluate_model(train_targets, train_baseline, task_type),
        'val': evaluate_model(val_targets, val_baseline, task_type)
    }

def print_results(model_metrics: Dict[str, Dict[str, float]], baseline_metrics: Dict[str, Dict[str, float]], task_type: str):
    """Print model and baseline performance metrics based on task type"""
    print("\nResults:")
    print("Model Performance:")
    if task_type == 'classification':
        print(f"Train Accuracy: {model_metrics['train']['accuracy']:.3f}")
        print(f"Train AUROC: {model_metrics['train']['auroc']:.3f}")
        print(f"Train AUPRC: {model_metrics['train']['auprc']:.3f}")
        print(f"Val Accuracy: {model_metrics['val']['accuracy']:.3f}")
        print(f"Val AUROC: {model_metrics['val']['auroc']:.3f}")
        print(f"Val AUPRC: {model_metrics['val']['auprc']:.3f}")
        
        print("\nBaseline (Majority Class) Performance:")
        print(f"Train Accuracy: {baseline_metrics['train']['accuracy']:.3f}")
        print(f"Train AUROC: {baseline_metrics['train']['auroc']:.3f}")
        print(f"Train AUPRC: {baseline_metrics['train']['auprc']:.3f}")
        print(f"Val Accuracy: {baseline_metrics['val']['accuracy']:.3f}")
        print(f"Val AUROC: {baseline_metrics['val']['auroc']:.3f}")
        print(f"Val AUPRC: {baseline_metrics['val']['auprc']:.3f}")
    else:  # regression
        print(f"Train RMSE: {model_metrics['train']['rmse']:.3f}")
        print(f"Train MAE: {model_metrics['train']['mae']:.3f}")
        print(f"Val RMSE: {model_metrics['val']['rmse']:.3f}")
        print(f"Val MAE: {model_metrics['val']['mae']:.3f}")
        
        print("\nBaseline (Mean Prediction) Performance:")
        print(f"Train RMSE: {baseline_metrics['train']['rmse']:.3f}")
        print(f"Train MAE: {baseline_metrics['train']['mae']:.3f}")
        print(f"Val RMSE: {baseline_metrics['val']['rmse']:.3f}")
        print(f"Val MAE: {baseline_metrics['val']['mae']:.3f}")

def run_benchmark(task: str, model_type: str, include_treatments: bool = True, 
                 prediction_horizon: int = None, random_state: int = 42,
                 regularization: str = 'ridge', alpha: float = 1.0):
    # Determine task type based on target column
    task_type = 'classification' if task in ['mechvent', 'morta_hosp', 'septic_shock', 'sepsis', 'vasopressor'] else 'regression'
    
    # Load and prepare data
    df = load_data("processed_files/patient_timeseries_v4.csv")
    features = get_feature_columns(df, task)
    
    # Filter out treatment variables if specified
    if not include_treatments:
        treatment_vars = ['mechvent', 'vaso_median', 'vaso_max', 'abx_given',
       'hours_since_first_abx', 'num_abx', 'fluid_total', 'fluid_step', 'peep', 'tidal_volume', 'minute_volume', 'peak_inspiratory_pressure', 'mean_airway_pressure']
        features = [f for f in features if f not in treatment_vars]
    
    # Initialize processor
    print("\nInitializing data processor...")
    processor = TimeSeriesDataProcessor(
        features=features,
        task=task,
        window_size=6,
        prediction_horizon=prediction_horizon if task in ['septic_shock', 'mechvent', 'sepsis', 'vasopressor'] else None
    )

    
    # Split data
    train_df, val_df = split_data(df)
    
    # Process and normalize data
    print("\nProcessing data...")
    train_features, train_targets = processor.prepare_data(train_df)
    val_features, val_targets = processor.prepare_data(val_df)
    train_features_norm, val_features_norm = processor.normalize_features(train_features, val_features)
    
    # Configure batch size based on model type
    batch_size = 32 if model_type in ['lstm', 'transformer'] else None
    
    # Train model
    print(f"\nTraining {model_type} model...")
    if model_type == 'linear':
        # Print regularization information for regression tasks
        if task_type == 'regression':
            if regularization == 'ridge':
                print(f"Using Ridge regression with alpha={alpha} (L2 regularization)")
            elif regularization == 'lasso':
                print(f"Using Lasso regression with alpha={alpha} (L1 regularization)")
            elif regularization == 'elasticnet':
                print(f"Using ElasticNet regression with alpha={alpha}, l1_ratio=0.5 (combined L1/L2 regularization)")
            else:
                print("Using standard Linear Regression (no regularization)")
        else:
            print("Using Logistic Regression for classification task")
            
        model = LinearTimeSeriesModel(
            task_type=task_type, 
            random_state=random_state,
            regularization=regularization if task_type == 'regression' else None,
            alpha=alpha
        )
    elif model_type == 'lstm':
        input_dim = train_features_norm.shape[2]
        model = LSTMModel(task_type=task_type, input_dim=input_dim)
    elif model_type == 'transformer':
        model = TimeSeriesTransformer(task_type=task_type)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    if batch_size:
        # For LSTM and Transformer models, use batched training
        model.fit(train_features_norm, train_targets, batch_size=batch_size)
        train_preds = model.predict(train_features_norm, batch_size=batch_size)
        val_preds = model.predict(val_features_norm, batch_size=batch_size)
    else:
        # For linear model, use regular training
        model.fit(train_features_norm, train_targets)
        train_preds = model.predict(train_features_norm)
        val_preds = model.predict(val_features_norm)
    
    # Calculate metrics
    model_metrics = {
        'train': evaluate_model(train_targets, train_preds, task_type),
        'val': evaluate_model(val_targets, val_preds, task_type)
    }
    
    baseline_metrics = get_baseline_metrics(train_targets, val_targets, task_type)
    
    # Print results
    print_results(model_metrics, baseline_metrics, task_type)
    
    # Return metrics for saving to CSV
    result = {
        'task': task,
        'model_type': model_type,
        'include_treatments': include_treatments,
        'prediction_horizon': prediction_horizon,
        'regularization': regularization,
        'alpha': alpha
    }
    
    # Add model metrics
    for split in ['train', 'val']:
        for metric, value in model_metrics[split].items():
            result[f'{split}_{metric}'] = value
    
    # Add baseline metrics
    for split in ['train', 'val']:
        for metric, value in baseline_metrics[split].items():
            result[f'{split}_baseline_{metric}'] = value
            
    return result

def run_all_experiments():
    """Run experiments with different configurations and save results to CSV"""
    # Define tasks and their types
    tasks = {
        'morta_hosp': 'static',  # Static outcome
        'los': 'static',         # Static outcome
        'septic_shock': 'temporal',  # Time-varying outcome
        #'mechvent': 'temporal',       # Time-varying outcome
        'vasopressor': 'temporal'       # Time-varying outcome
    }
    
    model_types = ['linear', 'lstm', 'transformer']
    treatment_options = [True, False]
    
    # Set fixed prediction horizon for temporal tasks
    fixed_prediction_horizon = 6  # Hours ahead to predict
    
    results = []
    
    # Calculate total experiments
    total_experiments = 0
    for task, task_type in tasks.items():
        if task_type == 'static':
            total_experiments += len(model_types) * len(treatment_options)
        else:  # temporal
            total_experiments += len(model_types) * len(treatment_options)
    
    experiment_count = 0
    
    # Run experiments for all tasks
    for task, task_type in tasks.items():
        for model_type in model_types:
            for include_treatments in treatment_options:
                if task_type == 'static':
                    # For static tasks, run once with no prediction horizon
                    experiment_count += 1
                    print(f"\n\n{'='*80}")
                    print(f"Experiment {experiment_count}/{total_experiments}")
                    print(f"Task: {task}, Model: {model_type}, Include Treatments: {include_treatments}")
                    print(f"{'='*80}\n")
                    
                    result = run_benchmark(
                        task=task,
                        model_type=model_type,
                        include_treatments=include_treatments,
                        prediction_horizon=None
                    )
                    results.append(result)
                    
                    # Save intermediate results after each experiment
                    results_df = pd.DataFrame(results)
                    results_df.to_csv("benchmark_results.csv", index=False)
                    print(f"Results saved to benchmark_results.csv")
                else:
                    # For temporal tasks, use fixed prediction horizon
                    experiment_count += 1
                    print(f"\n\n{'='*80}")
                    print(f"Experiment {experiment_count}/{total_experiments}")
                    print(f"Task: {task}, Model: {model_type}, Include Treatments: {include_treatments}")
                    print(f"Prediction Horizon: {fixed_prediction_horizon} hours")
                    print(f"{'='*80}\n")
                    
                    result = run_benchmark(
                        task=task,
                        model_type=model_type,
                        include_treatments=include_treatments,
                        prediction_horizon=fixed_prediction_horizon
                    )
                    results.append(result)
                    
                    # Save intermediate results after each experiment
                    results_df = pd.DataFrame(results)
                    results_df.to_csv("benchmark_results.csv", index=False)
                    print(f"Results saved to benchmark_results.csv")
    
    return results

def run_selected_experiments(task: str, include_treatments: bool = False):
    """Run experiments with all models for a specific task and treatment setting"""
    model_types = ['linear', 'lstm', 'transformer']
    
    # Determine if this is a temporal task
    temporal_tasks = ['septic_shock', 'mechvent', 'sepsis', 'vasopressor']
    is_temporal = task in temporal_tasks
    
    # Define prediction horizons for temporal tasks
    prediction_horizons = [1, 2, 3, 4, 5, 6] if is_temporal else [None]
    
    results = []
    
    total_experiments = len(model_types) * len(prediction_horizons)
    experiment_count = 0
    
    for model_type in model_types:
        for horizon in prediction_horizons:
            experiment_count += 1
            print(f"\n\n{'='*80}")
            print(f"Experiment {experiment_count}/{total_experiments}")
            print(f"Task: {task}, Model: {model_type}, Include Treatments: {include_treatments}")
            if is_temporal:
                print(f"Prediction Horizon: {horizon} hours")
            print(f"{'='*80}\n")
            
            result = run_benchmark(
                task=task,
                model_type=model_type,
                include_treatments=include_treatments,
                prediction_horizon=horizon
            )
            results.append(result)
            
            # Save intermediate results after each experiment
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"{task}_benchmark_results.csv", index=False)
            print(f"Results saved to {task}_benchmark_results.csv")
    
    return results

if __name__ == "__main__":
    # Set random seeds at the beginning
    set_random_seeds()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--run_all", action="store_true", help="Run all experiments")
    parser.add_argument("--run_selected", action="store_true", help="Run all models for a specific task")
    parser.add_argument("--task", type=str, default="mechvent", help="Target column name")
    parser.add_argument("--model_type", type=str, default="lstm", help="Model type")
    parser.add_argument("--include_treatments", type=bool, default=False, help="Whether to include treatment variables")
    parser.add_argument("--prediction_horizon", type=int, default=6, help="Prediction horizon for temporal tasks (hours)")
    parser.add_argument("--regularization", type=str, default="ridge", choices=["ridge", "lasso", "elasticnet", "none"], 
                        help="Regularization type for linear models")
    parser.add_argument("--alpha", type=float, default=1.0, help="Regularization strength")
    
    args = parser.parse_args()

    # Call this function at the beginning of your main function or script
    set_random_seeds(args.random_state)
        
    if args.run_all:
        run_all_experiments()
    elif args.run_selected:
        run_selected_experiments(args.task, args.include_treatments)
    else:
        # For single runs, use the specified prediction horizon for temporal tasks
        temporal_tasks = ['septic_shock', 'mechvent', 'sepsis', 'vasopressor']
        
        result = run_benchmark(
            args.task, 
            args.model_type,
            args.include_treatments,
            prediction_horizon=args.prediction_horizon if args.task in temporal_tasks else None,
            random_state=args.random_state,
            regularization=args.regularization,
            alpha=args.alpha
        )
        # Save single result to CSV
        pd.DataFrame([result]).to_csv("single_benchmark_result.csv", index=False)
        print(f"Result saved to single_benchmark_result.csv")
