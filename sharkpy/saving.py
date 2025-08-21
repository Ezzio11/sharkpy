#sharkpy/save_model.py

import joblib
import os
from pathlib import Path

def save_model(self, model, name="shark_model", directory="models"):
    """
    Save the trained model to a .joblib file with enhanced error handling.
    
    Parameters
    ----------
    model : object
        Trained ML model object
    name : str, optional
        Filename without extension (default: "shark_model")
    directory : str, optional
        Folder where the model will be saved (default: "models")
        
    Returns
    -------
    str
        Path to the saved model if successful
        
    Raises
    ------
    ValueError
        If model is None or not trained
    OSError
        If directory creation or file writing fails
    """
    try:
        # Validate model
        if model is None:
            raise ValueError("🦈 No model to save! Train a model first.")
            
        # Validate and clean filename
        name = Path(name).stem  # Remove any extension if provided
        
        # Create directory with proper error handling
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            raise OSError(f"🦈 Cannot create directory {directory}: {str(e)}")

        # Construct full path
        filepath = os.path.join(directory, f"{name}.joblib")
        
        # Check if file already exists
        if os.path.exists(filepath):
            print(f"⚠️ Warning: File {filepath} already exists and will be overwritten")
            
        # Save model with progress indicator
        print(f"🦈 Saving model to {filepath}...")
        joblib.dump({
                'model': model,
                'feature_names': self.feature_names,
                'encoders': self.encoders
            }, filepath)
        print(f"✨ Model saved successfully!")
        
        # Verify the save
        if not os.path.exists(filepath):
            raise OSError("🦈 Model file not created after save attempt")
            
        return filepath
        
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        raise

def load_model(self, model_path):
    """
    Load a saved SharkPy model from a .joblib file with enhanced error handling.
    
    Parameters
    ----------
    model_path : str
        Path to the saved .joblib model file
        
    Returns
    -------
    object
        The loaded model object
        
    Raises
    ------
    FileNotFoundError
        If model file doesn't exist
    ValueError
        If file is not a valid model
    """
    try:
        # Validate file path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"🦈 Model file not found: {model_path}")
            
        # Check file extension
        if not model_path.endswith('.joblib'):
            print("⚠️ Warning: File doesn't have .joblib extension")
            
        # Load model with progress indicator
        print(f"🦈 Loading model from {model_path}...")
        data = joblib.load(model_path)
        self.model = data['model']
        self.feature_names = data.get('feature_names', None)
        self.encoders = data.get('encoders', {}) 
               
        # Validate loaded model
        if not hasattr(self.model, 'fit') or not hasattr(self.model, 'predict'):
            raise ValueError("🦈 Loaded object doesn't appear to be a valid model")
            
        print("✨ Model loaded successfully!")
        return self.model
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise