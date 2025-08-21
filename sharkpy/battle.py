# sharkpy/battle.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Dict, Optional
from .learning import learn

MODEL_DETAILS = {
    'linear_regression': {
        'name': 'Linear Regression',
        'strengths': 'Simple, interpretable, fast',
        'best_for': 'Linear relationships, baseline models',
        'data_size': 'Any size',
        'training_speed': 'Very fast'
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'strengths': 'Simple, probabilistic output',
        'best_for': 'Binary/multiclass classification, baseline models',
        'data_size': 'Any size',
        'training_speed': 'Very fast'
    },
    'random_forest': {
        'name': 'Random Forest',
        'strengths': 'Robust, handles non-linearity, feature importance',
        'best_for': 'Complex relationships, feature selection',
        'data_size': 'Medium to large',
        'training_speed': 'Fast'
    },
    'svm': {
        'name': 'Support Vector Machine',
        'strengths': 'Effective in high dimensions',
        'best_for': 'Non-linear relationships, high-dimensional data',
        'data_size': 'Small to medium',
        'training_speed': 'Slow for large datasets'
    },
    'ridge': {
        'name': 'Ridge Regression',
        'strengths': 'L2 regularization, handles multicollinearity',
        'best_for': 'Linear relationships with correlated features',
        'data_size': 'Any size',
        'training_speed': 'Fast'
    },
    'lasso': {
        'name': 'Lasso Regression',
        'strengths': 'L1 regularization, feature selection',
        'best_for': 'Linear relationships with feature selection',
        'data_size': 'Any size',
        'training_speed': 'Fast'
    },
    'knn': {
        'name': 'K-Nearest Neighbors',
        'strengths': 'Simple, non-parametric',
        'best_for': 'Pattern recognition, simple non-linear relationships',
        'data_size': 'Small to medium',
        'training_speed': 'Very fast (but slow predictions)'
    },
    'xgboost': {
        'name': 'XGBoost',
        'strengths': 'High performance, feature importance, handles missing values',
        'best_for': 'Complex relationships, competitions',
        'data_size': 'Medium to large',
        'training_speed': 'Medium (with GPU support)'
    },
    'lightgbm': {
        'name': 'LightGBM',
        'strengths': 'Fast training, low memory usage',
        'best_for': 'Large datasets, speed-critical applications',
        'data_size': 'Large',
        'training_speed': 'Very fast'
    },
    'catboost': {
        'name': 'CatBoost',
        'strengths': 'Handles categorical features, reduced overfitting',
        'best_for': 'Datasets with many categorical features',
        'data_size': 'Medium to large',
        'training_speed': 'Medium'
    }
}

def battle(self, data: pd.DataFrame, target: str, models: List[str] = ['linear_regression', 'random_forest', 'xgboost'], 
           metric: str = 'r2', n_trials: int = 30, early_stopping: bool = False, min_score: float = 0.5, verbose: int = 0) -> Dict:
    """Battle multiple models against each other and return the champion.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data for training
    target : str
        Name of target column
    models : list
        List of model names to compete
    metric : str
        Metric to compare models (default: 'r2')
    n_trials : int
        Number of optimization trials for boosting models (default: 30)
    early_stopping : bool, optional
        If True, stops training if any model exceeds `min_score`. Not recommended as it may miss better models later (default: False)
    min_score : float
        Minimum score to trigger early stopping (ignored if early_stopping=False) (default: 0.5)
    verbose : int
        Verbosity level for model training (default: 0)
    
    Returns
    -------
    dict
        Dictionary containing champion model name, model object, score, all results, details, and comparison plot
    """
    print("🦈⚔️ MODEL BATTLE ROYALE ⚔️🦈")
    print("Preparing the arena...")
    
    # Store data in main instance for later use
    if isinstance(data, pd.DataFrame):
        self.features = data.drop(columns=[target])
        self.target = data[target]
    else:
        raise ValueError("🦈 Data must be a DataFrame!")
    
    # Check data size suitability
    data_size = len(data)
    for model_name in models:
        details = MODEL_DETAILS.get(model_name, {})
        recommended_size = details.get('data_size', 'Any size')
        if recommended_size == 'Small to medium' and data_size > 10000:
            print(f"⚠️ Warning: {model_name} is best for small to medium datasets, but you have {data_size} rows.")
        elif recommended_size == 'Medium to large' and data_size < 1000:
            print(f"⚠️ Warning: {model_name} performs better with medium to large datasets, but you have {data_size} rows.")
        elif recommended_size == 'Large' and data_size < 10000:
            print(f"⚠️ Warning: {model_name} is optimized for large datasets, but you have {data_size} rows.")
    
    results = {}
    best_score = -float('inf')
    best_model = None
    
    # Train and evaluate all models
    for model_name in models:
        print(f"\n🗡️ {model_name.upper()} enters the fray!")
        details = MODEL_DETAILS.get(model_name, {})
        print(f"   Strengths: {details.get('strengths', 'Unknown')}")
        print(f"   Best for: {details.get('best_for', 'Unknown')}")
        
        # Create a temporary Shark instance for each model
        temp_shark = self.__class__()
        try:
            # Train the model
            temp_shark.learn(
                data=data,
                target=target,
                model_choice=model_name,
                n_trials=n_trials if model_name in ['xgboost', 'lightgbm', 'catboost'] else None,
                verbose=verbose  # Add this line
            )
            
            # Get metrics
            cv_results, _ = temp_shark.report(cv_folds=5, export_path=None)
            
            # Get the appropriate metric score
            if temp_shark.problem_type == "regression":
                score = cv_results['test_r2'].mean()
            else:
                score = cv_results['test_accuracy'].mean()
                
            results[model_name] = score
            print(f"   Score: {score:.4f}")
            
            # Update best model if score is better
            if score > best_score:
                best_score = score
                best_model = temp_shark.model
                self.problem_type = temp_shark.problem_type
                
        except Exception as e:
            print(f"   ❌ {model_name} failed: {str(e)}")
            results[model_name] = None
    
    # Find the champion among all models
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        champion = max(valid_results, key=valid_results.get)
        print(f"\n🏆 CHAMPION: {champion.upper()} with {metric}: {valid_results[champion]:.4f}")
        
        # Store the winning model and its data
        self.model = best_model
        
        return {
            'champion': champion,
            'model': self.model,
            'score': valid_results[champion],
            'all_results': results,
            'details': MODEL_DETAILS.get(champion, {}),
            'comparison': _visualize_battle_results(valid_results)
        }
    else:
        print("\n💀 ALL MODELS FAILED! The battle is a draw.")
        print("💡 Suggestion: Try simpler models (e.g., linear_regression) or check your data for issues like missing values or incorrect formats.")
        return {'champion': None, 'all_results': results}

def _visualize_battle_results(results):
    """Create a bar plot of model performances."""
    valid_results = {k: v for k, v in results.items() if v is not None}
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(valid_results.keys()), y=list(valid_results.values()))
    plt.title("🦈 Model Battle Results 🦈")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()