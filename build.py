# build.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_model(n_estimators=100, max_depth=None, random_state=42):
    """
    Build and return a scikit-learn Pipeline for regression
    using a RandomForestRegressor.
    
    Parameters:
    -----------
    n_estimators : int
        Number of trees in the random forest.
    max_depth : int
        Maximum depth of each tree.
    random_state : int
        Random state for reproducibility.
    
    Returns:
    --------
    Pipeline
        A scikit-learn Pipeline object containing a scaler and random forest regressor.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        ))
    ])
    return pipeline

if __name__ == "__main__":
    # Example usage:
    model = build_model()
    print("Model pipeline built successfully:")
    print(model)
