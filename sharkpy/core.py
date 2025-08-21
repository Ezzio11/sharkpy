# sharkpy/core.py

from typing import List, Union, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .learning import learn
from .predicting import predict, predict_baseline
from .reporting import report
from .plotting import plot_model
from .saving import save_model, load_model
from .explaining import explain_model
from .battle import battle, MODEL_DETAILS

try:
    from .shapash_integration import explain_with_shapash
except ImportError:
    explain_with_shapash = None

class Shark:
    """
    A machine learning model manager that simplifies training, prediction, and analysis.

    Attributes
    ----------
    model : object or None
        The trained machine learning model (e.g., LogisticRegression, RandomForestClassifier).
    problem_type : str or None
        Type of ML problem ('classification' or 'regression').
    features : pd.DataFrame or None
        Input features used for training.
    target : pd.Series or np.ndarray or None
        Target variable (encoded for classification, original for regression).
    target_name : str or None
        Name of the target column in the input data.
    data : pd.DataFrame or None
        Original input DataFrame, including features and target.
    project_name : str or None
        Name of the current project for tracking and reporting.
    feature_names : list of str or None
        Names of feature columns.
    encoders : dict
        Dictionary storing feature encoders (e.g., for categorical features).
    label_encoder : LabelEncoder or None
        Encoder for categorical target variable (for classification).
    stats_model : object or None
        Statistical model for detailed analysis (optional).
    statistical_summary : str or None
        Summary of statistical analysis (optional).
    p_values : pd.Series or None
        P-values from statistical analysis (optional).
    conf_intervals : pd.DataFrame or None
        Confidence intervals from statistical analysis (optional).

    Examples
    --------
    >>> from sharkpy import Shark
    >>> import pandas as pd
    >>> shark = Shark()
    >>> data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv', header=None)
    >>> data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    >>> shark.learn(data=data, target='species', model_choice='logistic_regression')
    >>> predictions = shark.predict(data)
    >>> shark.explain(export_path='explanation.pdf', format='pdf', depth='simple')
    >>> cv_results, train_metrics = shark.report(cv_folds=5)
    """

    def __init__(self):
        """Initialize Shark with empty attributes."""
        # Core attributes
        self.model = None
        self.features = None
        self.target = None
        self.problem_type = None
        self.target_name = None
        self.data = None
        self.label_encoder = None

        # Metadata
        self.project_name = None
        self.feature_names = None
        self.encoders = {}

        # Statistical analysis
        self.stats_model = None
        self.statistical_summary = None
        self.p_values = None
        self.conf_intervals = None

    def learn(self, data: Union[str, pd.DataFrame], project_name: str = "your data", target: Optional[str] = None, 
              problem_type: Optional[str] = None, model: Optional[object] = None, model_choice: Optional[str] = None, 
              detailed_stats: bool = False, n_trials: int = 30, verbose: bool = False) -> 'Shark':
        """
        Train a machine learning model on the provided data.

        Parameters
        ----------
        data : str or pd.DataFrame
            Dataset for training. Can be a file path (CSV) or a pandas DataFrame.
        project_name : str, optional
            Name of the project for tracking and reporting (default: "your data").
        target : str, optional
            Name of the target column to predict (default: None).
        problem_type : str, optional
            Type of problem: 'regression', 'classification', or None for auto-detection (default: None).
        model : object, optional
            Custom model instance to use (default: None).
        model_choice : str, optional
            Built-in model to use (e.g., 'logistic_regression', 'random_forest', 'xgboost') (default: None).
        detailed_stats : bool, optional
            Whether to compute detailed statistical analysis (e.g., p-values, confidence intervals) (default: False).
        n_trials : int, optional
            Number of optimization trials for boosting models (e.g., XGBoost) (default: 30).
        verbose : bool, optional
            Whether to print detailed output during training (default: False).

        Returns
        -------
        Shark
            The current Shark instance with trained model and updated attributes.

        Notes
        -----
        - Automatically encodes categorical features and target (for classification).
        - Stores the original DataFrame in `self.data` and target name in `self.target_name`.
        - For classification, stores the `LabelEncoder` in `self.label_encoder` to preserve category names.
        - Performs K-Fold cross-validation and prints mean and standard deviation of scores.
        - Fits the selected model on the entire dataset after cross-validation.
        - Warning: Avoid loading untrusted CSV files, as they may contain malicious data.

        Examples
        --------
        >>> shark = Shark()
        >>> data = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'a']})
        >>> shark.learn(data, target='y', model_choice='logistic_regression')
        🦈 Looks like a classification problem (non-numeric target: y)
        🦈 Encoding categorical target 'y' to numeric labels
        ...
        >>> shark.target_name
        'y'
        >>> shark.label_encoder.classes_
        array(['a', 'b'], dtype=object)
        """
        # Store metadata
        self.project_name = project_name
        self.target_name = target
        self.data = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)

        # Process feature names and target
        if isinstance(self.data, pd.DataFrame):
            self.feature_names = [col for col in self.data.columns if col != target]
            self.features = self.data.drop(columns=[target])
            self.target = self.data[target]
        else:
            raise ValueError("🦈 Data must be a pandas DataFrame!")

        # Encode categorical target for classification
        if problem_type == 'classification' or (problem_type is None and not np.issubdtype(self.target.dtype, np.number)):
            print(f"🦈 Looks like a classification problem (non-numeric target: {target})")
            self.label_encoder = LabelEncoder()
            self.target = self.label_encoder.fit_transform(self.target)
            print(f"🦈 Encoding categorical target '{target}' to numeric labels")

        return learn(self, self.data, project_name, target, problem_type, 
                     model, model_choice, detailed_stats, n_trials, verbose)

    def predict(self, X: Optional[Union[Dict, pd.DataFrame, List[Dict], np.ndarray]] = None) -> Union[float, str, np.ndarray]:
        """
        Make predictions using the trained model.

        Parameters
        ----------
        X : dict, pd.DataFrame, list of dict, np.ndarray, or None, optional
            Input samples to predict. If None, predicts on training data. Options:
            - dict: Single prediction (e.g., {'feature1': value1, 'feature2': value2}).
            - list of dict: Multiple scenarios (e.g., [{'feature1': value1}, {'feature1': value2}]).
            - pd.DataFrame: Multiple samples with feature columns.
            - np.ndarray: Raw feature values (must match training feature count).

        Returns
        -------
        float, str, or np.ndarray
            Predicted values. For classification, returns original category names if `label_encoder` is available.

        Raises
        ------
        ValueError
            If no model is trained or input data is invalid.

        Examples
        --------
        >>> shark = Shark()
        >>> data = pd.DataFrame({'x1': [1, 2], 'x2': [3, 4], 'y': ['cat', 'dog']})
        >>> shark.learn(data, target='y')
        >>> shark.predict({'x1': 1, 'x2': 3})
        'cat'
        >>> shark.predict(data[['x1', 'x2']])
        array(['cat', 'dog'], dtype=object)
        """
        return predict(self, X)

    def predict_baseline(self) -> Union[float, str]:
        """
        Make a baseline prediction using the minimum values of the training features.

        Returns
        -------
        float or str
            Baseline prediction for regression (mean) or classification (most frequent class).

        Raises
        ------
        ValueError
            If no model is trained.

        Examples
        --------
        >>> shark = Shark()
        >>> data = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})
        >>> shark.learn(data, target='y')
        >>> shark.predict_baseline()
        20.0
        """
        return predict_baseline(self)

    def plot(self, kind: str = "prediction", show: bool = True, 
            save_path: Optional[str] = None, colors: Optional[Dict[str, str]] = None):
        """
        Visualize model behavior based on the specified plot type.

        Parameters
        ----------
        kind : str, optional
            Type of plot: 'prediction', 'residuals', 'confusion_matrix', 'roc', 
            'pr_curve', 'proba_hist', or 'feature_importance' (default: 'prediction').
        show : bool, optional
            Whether to display the plot (default: True).
        save_path : str, optional
            Path to save the plot (default: None).
        colors : dict, optional
            Custom color specifications for the plot. If None, uses default SharkPy colors.
            Available keys: 'primary', 'secondary', 'accent', 'background', 'grid', 'text', 'bars'

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no model is trained or the plot type is invalid.

        Examples
        --------
        >>> shark = Shark()
        >>> data = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 1, 0]})
        >>> shark.learn(data, target='y')
        >>> shark.plot(kind='confusion_matrix')
        
        >>> # Custom colors example
        >>> custom_colors = {
        >>>     'primary': '#FF6B6B',    # Coral red
        >>>     'secondary': '#4ECDC4',  # Turquoise
        >>>     'background': '#F7FFF7'  # Light green
        >>> }
        >>> shark.plot(kind='feature_importance', colors=custom_colors)
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("No model trained yet. Call learn() first.")
        
        if not hasattr(self, 'features') or self.features is None:
            raise ValueError("No feature data available.")
        
        if not hasattr(self, 'target') or self.target is None:
            raise ValueError("No target data available.")
        
        return plot_model(
            model=self.model, 
            X=self.features, 
            y=self.target, 
            kind=kind, 
            show=show, 
            save_path=save_path,
            feature_names=self.feature_names if hasattr(self, 'feature_names') else None,
            colors=colors
        )

    def report(self, cv_folds: int = 5, export_path: Optional[str] = None, format: str = 'txt') -> tuple:
        """
        Generate a comprehensive performance report with cross-validation metrics.

        Parameters
        ----------
        cv_folds : int, optional
            Number of cross-validation folds (default: 5).
        export_path : str, optional
            Path to export the report (txt, docx, or pdf) (default: None).
        format : str, optional
            Export format: 'txt', 'docx', or 'pdf' (default: 'txt').

        Returns
        -------
        tuple
            (cv_results, train_metrics), where cv_results is a dict of cross-validation metrics and train_metrics is a dict of training metrics.

        Raises
        ------
        ValueError
            If no model is trained or the format is invalid.

        Examples
        --------
        >>> shark = Shark()
        >>> data = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 1, 0]})
        >>> shark.learn(data, target='y')
        >>> cv_results, train_metrics = shark.report(cv_folds=5)
        >>> print(cv_results['test_accuracy'].mean())
        """
        return report(self, cv_folds, export_path, format)

    def explain(self, cv_results=None, train_metrics=None, export_path: Optional[str] = None, format: str = 'txt', depth: str = 'deep', verbose: int = 1) -> Optional[pd.DataFrame]:
        """
        Explain the model's behavior and performance with customizable depth and export options.

        Parameters
        ----------
        cv_results : dict, optional
            Cross-validation results from report(), containing metrics like test_r2 or test_accuracy.
        train_metrics : dict, optional
            Training metrics from report(), containing metrics like r2 or accuracy.
        export_path : str, optional
            Path to export the explanation (txt, docx, or pdf) (default: None).
        format : str, optional
            Export format: 'txt', 'docx', or 'pdf' (default: 'txt').
        depth : str, optional
            Explanation depth: 'simple' (beginner overview), 'mechanics' (technical details),
            'interpretation' (performance analysis), 'actionable' (recommendations),
            'deep' (all levels, default), or 'shapash' (interactive SHAP dashboard).

        Returns
        -------
        pd.DataFrame or None
            Feature importance DataFrame if available, else None.

        Notes
        -----
        - Requires a trained model (call `learn` first).
        - For classification, uses `label_encoder` to display original category names (e.g., 'Iris-setosa' instead of 0).
        - If `export_path` is provided, saves the explanation in the specified format.
        - 'shapash' depth requires the `shapash` package to be installed.

        Examples
        --------
        >>> shark = Shark()
        >>> data = pd.DataFrame({'x1': [1, 2], 'x2': [3, 4], 'y': ['cat', 'dog']})
        >>> shark.learn(data, target='y')
        >>> shark.explain(depth='simple', export_path='explanation.txt')
        🦈 Sharky is diving into the LogisticRegression model explanation...
        ...
        >>> # explanation.txt contains: "This model predicts one of 2 categories (cat, dog)..."
        """
        if self.model is None:
            print("🦈 Oops! Sharky can't explain a model that hasn't been trained yet! Call .learn() first.")
            return None
        
        print(f"🦈 Sharky is diving into the {type(self.model).__name__} model explanation...")
        
        # Call explain_model with all parameters, including target_name, data, and label_encoder
        feature_df = explain_model(
            model=self.model,
            features=self.features,
            target=self.target,
            target_name=self.target_name,
            data=self.data,
            label_encoder=self.label_encoder,
            cv_results=cv_results,
            train_metrics=train_metrics,
            export_path=export_path,
            format=format,
            depth=depth
        )
        
        if feature_df is not None:
            print("\n🦈 Sharky found some key features driving the model! Check the output above.")
        else:
            print("\n🦈 Sharky couldn't extract feature importance for this model type.")
        
        return feature_df

    def save_model(self, name: str = "shark_model", directory: str = "models") -> str:
        """
        Save the trained model to a .joblib file.

        Parameters
        ----------
        name : str, optional
            Filename without extension (default: "shark_model").
        directory : str, optional
            Folder where the model will be saved (default: "models").

        Returns
        -------
        str
            Path to the saved model file.

        Raises
        ------
        ValueError
            If no model is trained.
        OSError
            If directory creation or file writing fails.

        Examples
        --------
        >>> shark = Shark()
        >>> data = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})
        >>> shark.learn(data, target='y')
        >>> shark.save_model(name='my_model')
        'models/my_model.joblib'
        """
        return save_model(self, self.model, name, directory)

    def load_model(self, model_path: str) -> object:
        """
        Load a saved SharkPy model from a .joblib file.

        Parameters
        ----------
        model_path : str
            Path to the saved .joblib model file.

        Returns
        -------
        object
            The loaded model object.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        ValueError
            If the file is not a valid model.

        Examples
        --------
        >>> shark = Shark()
        >>> shark.load_model('models/my_model.joblib')
        <sklearn.linear_model.LinearRegression object at ...>
        """
        return load_model(self, model_path)

    def battle(self, data: pd.DataFrame, target: str, models: List[str] = ['linear_regression', 'random_forest', 'xgboost'], 
            metric: str = 'r2', n_trials: int = 30, early_stopping: bool = False, min_score: float = 0.5, verbose: int = 0) -> Dict:
        """
        Compare multiple models and select the best performer.

        Parameters
        ----------
        data : pd.DataFrame
            Input data for training.
        target : str
            Name of the target column.
        models : list of str, optional
            List of model names to compare (e.g., ['linear_regression', 'random_forest']) (default: ['linear_regression', 'random_forest', 'xgboost']).
        metric : str, optional
            Metric to compare models (e.g., 'r2', 'accuracy') (default: 'r2').
        n_trials : int, optional
            Number of optimization trials for boosting models (default: 30).
        early_stopping : bool, optional
            If True, stops training if any model exceeds `min_score`. Not recommended as it may miss better models later (default: False).
        min_score : float, optional
            Minimum score to trigger early stopping (default: 0.5).
        verbose : int, optional
            Verbosity level for model training (default: 0)

        Returns
        -------
        dict
            Dictionary containing champion model name, model object, score, all results, details, and comparison plot.

        Examples
        --------
        >>> shark = Shark()
        >>> data = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})
        >>> result = shark.battle(data, target='y', models=['linear_regression', 'random_forest'])
        >>> print(result['champion'])
        'linear_regression'
        """
        return battle(self, data, target, models, metric, n_trials, early_stopping, min_score, verbose)

    def explain_with_shapash(self, title_story: Optional[str] = None, display: bool = True):
        """
        Create an interactive Shapash dashboard for model interpretation.

        Parameters
        ----------
        title_story : str, optional
            Title for the Shapash dashboard (default: None).
        display : bool, optional
            Whether to display the dashboard (default: True).

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If the `shapash` package is not installed.
        ValueError
            If no model is trained.

        Examples
        --------
        >>> shark = Shark()
        >>> data = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 1, 0]})
        >>> shark.learn(data, target='y')
        >>> shark.explain_with_shapash(title_story='My Model Analysis')
        """
        return explain_with_shapash(self, title_story, display)

    def available_models(self) -> Dict:
        """
        List all available models with their details and print a comparison table.

        Returns
        -------
        dict
            Dictionary of available models and their details.

        Examples
        --------
        >>> shark = Shark()
        >>> models = shark.available_models()
        🦈 Available Models in SharkPy 🦈
        ...
        >>> print(models.keys())
        dict_keys(['linear_regression', 'random_forest', 'xgboost', ...])
        """
        print("\n🦈 Available Models in SharkPy 🦈")
        df = pd.DataFrame.from_dict(MODEL_DETAILS, orient='index')
        print(df.to_string())
        return MODEL_DETAILS