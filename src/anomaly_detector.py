"""
Time Series Anomaly Detection
Detects anomalies in time series data using statistical and ML methods
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesAnomalyDetector:
    """Detect anomalies in time series data"""
    
    def __init__(self, method: str = 'isolation_forest'):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        
    def detect_statistical(self, data: pd.Series, n_std: float = 3.0) -> pd.Series:
        """Detect anomalies using statistical methods (z-score)"""
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / std)
        return z_scores > n_std
    
    def detect_isolation_forest(self, data: pd.DataFrame, contamination: float = 0.1) -> np.ndarray:
        """Detect anomalies using Isolation Forest"""
        self.model = IsolationForest(contamination=contamination, random_state=42)
        predictions = self.model.fit_predict(data)
        return predictions == -1  # -1 indicates anomaly
    
    def detect_moving_average(self, data: pd.Series, window: int = 7, n_std: float = 2.0) -> pd.Series:
        """Detect anomalies using moving average"""
        ma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_bound = ma + (n_std * std)
        lower_bound = ma - (n_std * std)
        return (data > upper_bound) | (data < lower_bound)
    
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main detection method"""
        result = data.copy()
        
        if self.method == 'statistical':
            result['is_anomaly'] = self.detect_statistical(data.iloc[:, 0])
        elif self.method == 'isolation_forest':
            result['is_anomaly'] = self.detect_isolation_forest(data)
        elif self.method == 'moving_average':
            result['is_anomaly'] = self.detect_moving_average(data.iloc[:, 0])
        
        return result


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    values = np.random.normal(100, 10, 100)
    values[50] = 200  # Inject anomaly
    
    df = pd.DataFrame({'date': dates, 'value': values})
    
    detector = TimeSeriesAnomalyDetector(method='statistical')
    result = detector.detect(df[['value']])
    
    print(f"Detected {result['is_anomaly'].sum()} anomalies")
