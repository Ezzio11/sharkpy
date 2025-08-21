# sharkpy/shapash_integration.py

import pandas as pd
try:
    from shapash import SmartExplainer
except ImportError:
    SmartExplainer = None

def explain_with_shapash(self, title_story=None, display=True):
    """
    Create an interactive dashboard using Shapash to explain the model's predictions.
    
    Parameters
    ----------
    title_story : str, optional
        Title for the Shapash dashboard
    display : bool, optional
        Whether to display the dashboard immediately (default: True)
        
    Returns
    -------
    app : shapash.webapp.smart_app.SmartApp
        The Shapash web application object
        
    Notes
    -----
    - Requires the `shapash` package to be installed.
    - Warning: The dashboard may expose sensitive data from your features and predictions. Use caution with untrusted environments.
    """
    if not hasattr(self, 'model'):
        raise AttributeError("🦈 Oops! No model found. Train a model first using learn()")
    
    if SmartExplainer is None:
        raise ImportError("🦈 Shapash is not installed. Please install it with `pip install shapash`.")
        
    # Get predictions in the correct format
    predictions = self.predict()
    predictions_series = pd.Series(
        predictions,
        index=self.features.index,
        name='predicted_values'
    )
    
    # Create Shapash explainer
    try:
        xpl = SmartExplainer(model=self.model)
    except Exception as e:
        print(f"🦈 Failed to initialize Shapash explainer: {str(e)}")
        raise
    
    try:
        xpl.compile(
            x=self.features,
            y_pred=predictions_series,
            y_target=self.target
        )
        
        # Set default title if none provided
        if title_story is None:
            title_story = f"Shark's {self.problem_type.title()} Analysis 🦈"
            
        # Generate dashboard
        app = xpl.run_app(title_story=title_story)
        
        print("🦈 Shapash dashboard created successfully!")
        if display:
            print("📊 View the interactive dashboard in your browser")
            print("⚠️ Warning: Dashboard may expose sensitive data. Ensure you trust the environment.")
        return app
        
    except Exception as e:
        print(f"🦈 Oops! Shapash encountered an error: {str(e)}")
        print("💡 Tip: Make sure your model and data are compatible with Shapash")
        raise