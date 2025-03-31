import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from typing import Dict, Union, Tuple, Optional, Literal

class LinearTimeSeriesModel:
    def __init__(self, 
                 task_type: str = 'classification', 
                 random_state: Optional[int] = None,
                 regularization: Optional[Literal['ridge', 'lasso', 'elasticnet']] = 'ridge',
                 alpha: float = 1.0):
        """
        Args:
            task_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
            regularization: Type of regularization to use ('ridge', 'lasso', 'elasticnet', or None)
            alpha: Regularization strength parameter
        """
        self.task_type = task_type
        
        if task_type == 'classification':
            self.model = LogisticRegression(random_state=random_state)
        else:  # regression
            if regularization == 'ridge':
                self.model = Ridge(alpha=alpha, random_state=random_state)
                print(f"Using Ridge regression with alpha={alpha}")
            elif regularization == 'lasso':
                self.model = Lasso(alpha=alpha, random_state=random_state)
                print(f"Using Lasso regression with alpha={alpha}")
            elif regularization == 'elasticnet':
                self.model = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=random_state)
                print(f"Using ElasticNet regression with alpha={alpha}, l1_ratio=0.5")
            else:
                self.model = LinearRegression()
                print("Using standard Linear Regression (no regularization)")
    
    def flatten_features(self, data: np.ndarray) -> np.ndarray:
        """Reshape (N, T, X) data to (N, T*X)"""
        N, T, X = data.shape
        return data.reshape(N, T * X)
    
    def fit(self, train_data: np.ndarray, train_targets: np.ndarray) -> None:
        X = self.flatten_features(train_data)
        self.model.fit(X, train_targets)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        X = self.flatten_features(data)
        if self.task_type == 'classification':
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)
    

