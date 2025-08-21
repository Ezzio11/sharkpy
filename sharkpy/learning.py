# sharkpy/learning.py

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import optuna
import optuna.logging
import lightgbm as lgb
import catboost as cb
from typing import Union, Optional, Any
import random

# Prediction intros for the learn function
PREDICTION_INTROS = [
    "🦈 Diving into {project_name}! Time to make some waves! 🌊",
    "🦈 Sharpening teeth on {project_name}! Ready to take a bite out of prediction! 🦈",
    "🦈 Commanding the seas of {project_name}! Prepare for precision strikes! 🎯",
    "🦈 Swimming through {project_name}... hunting for patterns! 🔍",
    "🦈 Unleashing the predator on {project_name}! No pattern is safe! ⚡",
    "🦈 Lurking in the data streams of {project_name}... ready to strike! 🌊",
    "🦈 Circling {project_name} like the apex predictor I am! 🏆",
    "🦈 Scenting blood in the data waters of {project_name}! 🩸",
    "🦈 Preparing to feast on the patterns in {project_name}! 🍽️",
    "🦈 Navigating the deep data trenches of {project_name}! 🗺️"
]

def learn(
    self,
    data: Union[str, pd.DataFrame],
    project_name: str = "your data",
    target: Optional[str] = None,
    problem_type: Optional[str] = None,
    model: Optional[Any] = None,
    model_choice: Optional[str] = None,
    detailed_stats: bool = False,
    n_trials: int = 30,
    verbose: bool = False
) -> 'Shark':
    """
    Train a machine learning model using the provided data and parameters.

    Parameters
    ----------
    self : Shark
        The Shark instance.
    data : str or pandas.DataFrame
        The dataset to use for training. Can be a file path (CSV) or a DataFrame.
    project_name : str, optional
        Name of the project for tracking and reporting.
    target : str, optional
        Name of the column to predict. If None, uses the last column.
    problem_type : str, optional
        Type of problem: "regression" or "classification". If None, tries to infer automatically.
    model : sklearn.base.BaseEstimator, optional
        A custom scikit-learn compatible model instance to use. If provided, overrides model_choice.
    model_choice : str, optional
        String identifier for built-in model selection. Options:
            - "random_forest": RandomForestRegressor or RandomForestClassifier
            - "svm": SVR or SVC
            - "ridge": Ridge Regression (L2 regularization)
            - "lasso": Lasso Regression (L1 regularization)
            - "knn": K-Nearest Neighbors
            - "xgboost": XGBoost with Optuna optimization
            - "lightgbm": LightGBM with Optuna optimization
            - "catboost": CatBoost with Optuna optimization
            - None: LinearRegression or LogisticRegression (default)
    detailed_stats : bool, optional
        If True, uses statsmodels for detailed statistical analysis
    n_trials : int, optional
        Number of optimization trials for boosting models (default: 30)
    verbose : bool, optional
        If True, enables verbose logging for Optuna optimization (default: False)

    Notes
    -----
    - Encodes categorical features and target automatically for classification.
    - Performs K-Fold cross-validation and prints mean and std of scores.
    - Fits the selected model on the entire dataset after cross-validation.
    - Sets self.model, self.problem_type, self.features, self.target, and self.encoders.
    - Warning: Avoid loading untrusted CSV files, as they may contain malicious data.
    """
    print(random.choice(PREDICTION_INTROS).format(project_name=project_name))
    
    # Set Optuna verbosity
    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)

    # Load data if string path provided
    if isinstance(data, str):
        try:
            data = pd.read_csv(data)
        except Exception as e:
            raise ValueError(f"🦈 Could not read data from {data}: {str(e)}")
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("🦈 Data must be a pandas DataFrame or a valid CSV file path")
    
    # Set project name
    self.project_name = project_name
    
    # Select target and features
    if target is None:
        print("🦈 No target specified, using last column as target")
        target = data.columns[-1]
    
    if target not in data.columns:
        raise ValueError(f"🦈 Target column '{target}' not found in data")
    
    self.features = data.drop(columns=[target])
    self.target = data[target]
    
    # Infer problem type if not specified
    if problem_type is None:
        if pd.api.types.is_numeric_dtype(self.target):
            self.problem_type = "regression"
            print(f"🦈 Looks like a regression problem (numeric target: {target})")
        else:
            self.problem_type = "classification"
            print(f"🦈 Looks like a classification problem (non-numeric target: {target})")
    else:
        self.problem_type = problem_type.lower()
        if self.problem_type not in ["regression", "classification"]:
            raise ValueError("🦈 Problem type must be 'regression' or 'classification'")
    
    # Encode target for classification if necessary
    if self.problem_type == "classification" and not pd.api.types.is_numeric_dtype(self.target):
        print(f"🦈 Encoding categorical target '{target}' to numeric labels")
        self.target_encoder = LabelEncoder()
        self.target = pd.Series(self.target_encoder.fit_transform(self.target), index=self.target.index, name=self.target.name)
    
    # Encode categorical features
    self.feature_names = self.features.columns
    for col in self.features.columns:
        if self.features[col].dtype == "object":
            print(f"🦈 Encoding categorical feature '{col}'")
            self.features[col] = self.features[col].astype("category")
            self.encoders[col] = dict(enumerate(self.features[col].cat.categories))
            self.features[col] = self.features[col].cat.codes
    
    # Model selection
    if model is not None:
        print(f"🦈 Using custom model: {type(model).__name__}")
        self.model = model.fit(self.features, self.target)
    else:
        if model_choice is None:
            model_choice = "linear_regression" if self.problem_type == "regression" else "logistic_regression"
            print(f"🦈 No model specified, defaulting to {model_choice}")
        
        model_choice = model_choice.lower()
        if model_choice == "random_forest":
            self.model = RandomForestRegressor(random_state=42).fit(self.features, self.target) if self.problem_type == "regression" else RandomForestClassifier(random_state=42).fit(self.features, self.target)
        elif model_choice == "svm":
            self.model = SVR().fit(self.features, self.target) if self.problem_type == "regression" else SVC().fit(self.features, self.target)
        elif model_choice == "ridge":
            self.model = Ridge(alpha=1.0, random_state=42).fit(self.features, self.target) if self.problem_type == "regression" else LogisticRegression(penalty='l2').fit(self.features, self.target)
        elif model_choice == "lasso":
            self.model = Lasso(alpha=1.0, random_state=42).fit(self.features, self.target) if self.problem_type == "regression" else LogisticRegression(penalty='l1', solver='liblinear').fit(self.features, self.target)
        elif model_choice == "knn":
            self.model = KNeighborsRegressor(n_neighbors=5).fit(self.features, self.target) if self.problem_type == "regression" else KNeighborsClassifier(n_neighbors=5).fit(self.features, self.target)
        elif model_choice == "xgboost":
            print("🦈 Optimizing XGBoost with Optuna...")
            self.model = _create_optimized_xgboost(self.features, self.target, self.problem_type, n_trials)
        elif model_choice == "lightgbm":
            print("🦈 Optimizing LightGBM with Optuna...")
            self.model = _create_optimized_lightgbm(self.features, self.target, self.problem_type, n_trials)
        elif model_choice == "catboost":
            print("🦈 Optimizing CatBoost with Optuna...")
            self.model = _create_optimized_catboost(self.features, self.target, self.problem_type, n_trials)
        else:
            self.model = LinearRegression().fit(self.features, self.target) if self.problem_type == "regression" else LogisticRegression().fit(self.features, self.target)
    
    # Detailed statistical analysis for regression
    if detailed_stats and self.problem_type == "regression":
        try:
            X_with_const = sm.add_constant(self.features)
            self.stats_model = sm.OLS(self.target, X_with_const).fit()
            self.statistical_summary = self.stats_model.summary().as_text()
            self.p_values = self.stats_model.pvalues
            self.conf_intervals = self.stats_model.conf_int()
        except Exception as e:
            print(f"⚠️ Could not compute statistical details: {str(e)}")
    
    print(f"🦈 Model training complete! Ready to make predictions.")
    return self

def _create_optimized_xgboost(X: pd.DataFrame, y: pd.Series, problem_type: str = "regression", n_trials: int = 30) -> Any:
    """
    Create and optimize an XGBoost model using Optuna.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame
    y : pd.Series
        Target series
    problem_type : str
        Type of problem: "regression" or "classification"
    n_trials : int
        Number of optimization trials (default: 30)
    
    Returns
    -------
    model : Any
        Trained XGBoost model with optimized parameters
    """
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': 42
        }
        
        if problem_type == "regression":
            model = xgb.XGBRegressor(**params)
        else:
            model = xgb.XGBClassifier(**params)
            
        return model.fit(X, y).score(X, y)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    if problem_type == "regression":
        final_model = xgb.XGBRegressor(**study.best_params)
    else:
        final_model = xgb.XGBClassifier(**study.best_params)
    
    print(f"🦈 Best XGBoost parameters found: {study.best_params}")
    return final_model.fit(X, y)

def _create_optimized_lightgbm(X: pd.DataFrame, y: pd.Series, problem_type: str = "regression", n_trials: int = 30) -> Any:
    """
    Create and optimize a LightGBM model using Optuna.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame
    y : pd.Series
        Target series
    problem_type : str
        Type of problem: "regression" or "classification"
    n_trials : int
        Number of optimization trials (default: 30)
    
    Returns
    -------
    model : Any
        Trained LightGBM model with optimized parameters
    """
    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'random_state': 42
        }
        
        if problem_type == "regression":
            model = lgb.LGBMRegressor(**params)
        else:
            model = lgb.LGBMClassifier(**params)
            
        return model.fit(X, y).score(X, y)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    if problem_type == "regression":
        final_model = lgb.LGBMRegressor(**study.best_params)
    else:
        final_model = lgb.LGBMClassifier(**study.best_params)
    
    print(f"🦈 Best LightGBM parameters found: {study.best_params}")
    return final_model.fit(X, y)

def _create_optimized_catboost(X: pd.DataFrame, y: pd.Series, problem_type: str = "regression", n_trials: int = 30) -> Union[xgb.XGBRegressor, xgb.XGBClassifier]:    
    """
    Create and optimize a CatBoost model using Optuna.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame
    y : pd.Series
        Target series
    problem_type : str
        Type of problem: "regression" or "classification"
    n_trials : int
        Number of optimization trials (default: 30)
    
    Returns
    -------
    model : Any
        Trained CatBoost model with optimized parameters
    """
    def objective(trial):
        params = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'iterations': trial.suggest_int('iterations', 50, 300),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1),
            'random_seed': 42,
            'verbose': False
        }
        
        if problem_type == "regression":
            model = cb.CatBoostRegressor(**params)
        else:
            model = cb.CatBoostClassifier(**params)
            
        return model.fit(X, y).score(X, y)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    if problem_type == "regression":
        final_model = cb.CatBoostRegressor(**study.best_params)
    else:
        final_model = cb.CatBoostClassifier(**study.best_params)
    
    print(f"🦈 Best CatBoost parameters found: {study.best_params}")
    return final_model.fit(X, y)