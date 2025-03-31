import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from typing import Dict, Union

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = 'classification'
) -> Dict[str, float]:
    """Calculate task-specific evaluation metrics"""
    metrics = {}
    
    if task_type == 'classification':
        # Classification metrics
        metrics['auroc'] = roc_auc_score(y_true, y_pred)
        metrics['auprc'] = average_precision_score(y_true, y_pred)
        
        # Calculate optimal threshold using PR curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate metrics at optimal threshold
        y_pred_binary = (y_pred >= optimal_threshold).astype(int)
        metrics['sensitivity'] = np.sum((y_true == 1) & (y_pred_binary == 1)) / np.sum(y_true == 1)
        metrics['specificity'] = np.sum((y_true == 0) & (y_pred_binary == 0)) / np.sum(y_true == 0)
        metrics['ppv'] = np.sum((y_true == 1) & (y_pred_binary == 1)) / np.sum(y_pred_binary == 1)
        metrics['npv'] = np.sum((y_true == 0) & (y_pred_binary == 0)) / np.sum(y_pred_binary == 0)
        
    else:
        # Regression metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
    
    return metrics 