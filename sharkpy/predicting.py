#sharkpy/predict.py

import pandas as pd
import numpy as np

def predict(self, X=None):
    """
    Make predictions using the trained model.
    
    Parameters:
        X: dict, DataFrame, or array-like, optional
           Input samples to predict. If None, predicts on training data.
           Can be:
           - dict: Single prediction scenario {'feature1': value1, 'feature2': value2}
           - list of dicts: Multiple scenarios [{'feature1': value1}, {'feature1': value2}]
           - DataFrame: Multiple samples with feature columns
           - array-like: Raw feature values
        
    Returns:
        float, str, or array: Predicted values
    """
    print("🦈 Sharky is analyzing the data!")

    # Validate model exists
    if not hasattr(self, 'model'):
        raise AttributeError("🦈 Oops! Sharky needs training first. Call learn() before predict().")

    # If no X provided, use training features
    if X is None:
        print("🦈 No data provided! Using training data for prediction...")
        X = self.features
    else:
        # Process input
        X = _validate_and_process_input(self, X)
    
    # Make prediction
    try:
        predictions = self.model.predict(X)
        
        # Decode categorical predictions if encoder exists
        if hasattr(self, 'target_encoder'):
            predictions = self.target_encoder.inverse_transform(predictions)
        
        # Format output based on number of predictions
        if len(predictions) == 1:
            print(f"🦈 Prediction: {predictions[0]}")
            return predictions[0]
        else:
            print(f"🦈 Made {len(predictions)} predictions!")
            return predictions
            
    except Exception as e:
        print(f"🦈 Uh-oh! Sharky encountered an error: {str(e)}")
        raise

def predict_baseline(self):
    """
    Make a baseline prediction with all features at minimum values.
    
    Returns:
        float or str: Baseline prediction
    """
    print("🦈 Making baseline prediction (all features at minimum values)...")
    
    # Create baseline sample with minimum values
    baseline_data = {}
    for col in self.features.columns:
        if col in self.encoders:  # Only use encoder for categorical features
            baseline_data[col] = list(self.encoders[col].values())[0]
        else:
            # For numerical features, use 0
            baseline_data[col] = 0 
    
    return predict(self, baseline_data)

def _validate_and_process_input(self, X):
    """Helper method to validate and process input data"""
    # Convert input to DataFrame
    if isinstance(X, dict):
        X = pd.DataFrame([X])
    elif isinstance(X, np.ndarray):
        if not hasattr(self, 'feature_names'):
            raise ValueError("🦈 Sharky doesn't know the feature names for numpy arrays!")
        X = pd.DataFrame(X, columns=self.feature_names)
    elif not isinstance(X, pd.DataFrame):
        raise ValueError("🦈 Input must be a dictionary, DataFrame, or numpy array!")

    # Validate features
    missing_features = set(self.feature_names) - set(X.columns)
    if missing_features:
        raise ValueError(f"🦈 Missing features: {missing_features}")

    # Process categorical features
    X_processed = X.copy()
    for col in X_processed.columns:
        if col in self.encoders:
            cat_to_code = {v: k for k, v in self.encoders[col].items()}
            codes = set(cat_to_code.values())
            mapped = X_processed[col].map(cat_to_code)
            # If values are already numeric codes, keep them
            already_codes_mask = X_processed[col].isin(codes)
            combined = mapped.where(~mapped.isnull(), X_processed[col])
            # Now flag only truly unseen values (neither label nor known code)
            unseen_mask = mapped.isnull() & ~already_codes_mask
            if unseen_mask.any():
                unseen_values = X[col][unseen_mask].unique()
                raise ValueError(f"🦈 Unseen categories in '{col}': {unseen_values}")
            # Ensure integer codes
            X_processed[col] = combined.astype(int)

    return X_processed