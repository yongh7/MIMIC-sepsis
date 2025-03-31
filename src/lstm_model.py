import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple

class LSTMModel(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 task_type: str = 'classification'):
        super().__init__()
        self.task_type = task_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # LSTM for processing input sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        if task_type == 'classification':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        output = self.output_layer(last_hidden)
        return self.output_activation(output).squeeze()
    
    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, epochs: int = 10, 
            learning_rate: float = 0.001, random_state: int = None):
        """Train the LSTM model"""
        self.to(self.device)
        
        # Set random seed for reproducibility if provided
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
                torch.cuda.manual_seed_all(random_state)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss() if self.task_type == 'classification' else nn.MSELoss()
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Generate predictions using the trained model"""
        self.to(self.device)
        self.eval()
        
        # Convert input to PyTorch tensor
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        with torch.no_grad():
            for batch_X, in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self(batch_X)
                
                # Ensure outputs are properly shaped before converting to numpy
                # If we have a single sample, make sure it's still a 1D array
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                    
                # Convert to numpy and append to predictions
                batch_preds = outputs.cpu().numpy()
                
                # Ensure batch_preds is always at least 1D
                if batch_preds.ndim == 0:
                    batch_preds = np.array([batch_preds.item()])
                    
                predictions.append(batch_preds)
        
        # Concatenate all batch predictions
        if predictions:
            return np.concatenate(predictions)
        else:
            return np.array([]) 