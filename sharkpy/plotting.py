# sharkpy/plotting.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.font_manager as fm
import warnings
from typing import Dict, List, Optional, Union

# Suppress glyph warnings if font fallback fails
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Set fun, kid-friendly style
sns.set_palette("Set2")  # Accessible, vibrant palette
sns.set_context("notebook", font_scale=1.2)  # Slightly larger text

# Try to use a font that supports emojis
try:
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    emoji_fonts = ['Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji']
    selected_font = next((f for f in emoji_fonts if f in available_fonts), 'Arial')
    plt.rcParams['font.family'] = selected_font
except Exception as e:
    print(f"⚠️ Could not set emoji font, falling back to Arial: {e}")
    plt.rcParams['font.family'] = 'Arial'

# Default SharkPy color scheme (ocean/water theme)
SHARKPY_COLORS = {
    'primary': '#2E86AB',    # Deep ocean blue
    'secondary': '#A23B72',  # Coral pink
    'accent': '#F18F01',     # Sunny yellow (like shark eyes)
    'background': '#F7F7F7', # Light gray background
    'grid': '#D3D3D3',       # Grid lines
    'text': '#2B2D42',       # Dark blue-gray text
    'bars': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A8EAE', '#56E39F']  # Colorful bars
}

def validate_colors(colors: Dict[str, str]) -> Dict[str, str]:
    """
    Validate color dictionary and fall back to defaults if invalid.
    
    Parameters:
        colors: Dictionary of color specifications
        
    Returns:
        Validated color dictionary
    """
    if not colors or not isinstance(colors, dict):
        return SHARKPY_COLORS.copy()
    
    # Create a copy of default colors and update with provided values
    validated_colors = SHARKPY_COLORS.copy()
    
    for key, value in colors.items():
        if key in validated_colors:
            # Validate color format (basic check)
            if isinstance(value, str) and (value.startswith('#') or value in plt.colors.cnames):
                validated_colors[key] = value
            elif isinstance(value, list) and all(isinstance(c, str) for c in value):
                validated_colors[key] = value
            else:
                print(f"⚠️ Invalid color format for '{key}': {value}. Using default.")
    
    return validated_colors

def plot_model(model, X, y=None, kind="prediction", show=True, save_path=None, 
               feature_names=None, colors: Optional[Dict[str, str]] = None):
    """
    Visualizes model behavior depending on the specified kind.

    Parameters:
        model : trained model
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,), optional
        kind : str, one of {"prediction", "residuals", "confusion_matrix", "roc", "pr_curve", "proba_hist", "feature_importance"}
        show : bool, whether to display the plot
        save_path : str or None, path to save the plot
        feature_names : list, optional, names of features for plotting
        colors : dict, optional, custom color specifications
    """
    print(f"🦈 Debug: plot_model called with kind={kind}")
    if kind in ["prediction", "residuals", "confusion_matrix", "roc", "pr_curve", "proba_hist"] and y is None:
        raise ValueError(f"y must be provided for kind='{kind}'.")

    # Validate and prepare colors
    plot_colors = validate_colors(colors)

    # Dispatch to plotting functions
    if kind == "prediction":
        _plot_prediction(model, X, y, plot_colors)
    elif kind == "residuals":
        _plot_residuals(model, X, y, plot_colors)
    elif kind == "confusion_matrix":
        _plot_confusion_matrix(model, X, y, plot_colors)
    elif kind == "roc":
        _plot_roc(model, X, y, plot_colors)
    elif kind == "pr_curve":
        _plot_pr_curve(model, X, y, plot_colors)
    elif kind == "proba_hist":
        _plot_proba_hist(model, X, y, plot_colors)
    elif kind == "feature_importance":
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns
            else:
                raise ValueError("Feature names required for feature importance plot.")
        _plot_feature_importance(model, feature_names, plot_colors)
    else:
        raise ValueError(f"Unknown kind='{kind}'. Supported kinds: prediction, residuals, confusion_matrix, roc, pr_curve, proba_hist, feature_importance")

    if save_path:
        dir_ = os.path.dirname(save_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

# === Internal Helper Functions === #

def _plot_prediction(model, X, y, colors):
    y_pred = model.predict(X)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=colors['background'])
    
    # Fun scatter plot with star markers
    ax.scatter(y, y_pred, 
              alpha=0.7, 
              c=colors['primary'],
              edgecolor='white',
              s=120,
              marker='*')
    
    # Perfect prediction line with fun style
    line_range = [min(y), max(y)]
    ax.plot(line_range, line_range, 
            '--', 
            color=colors['secondary'],
            linewidth=3,
            label='Perfect Prediction')
    
    # Playful styling
    ax.set_xlabel("Actual Values", fontsize=12, color=colors['text'])
    ax.set_ylabel("Predicted Values", fontsize=12, color=colors['text'])
    ax.set_title("Actual vs Predicted Values\nShark's Predictions", 
                fontsize=14, 
                pad=20,
                color=colors['text'])
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.legend(fontsize=10)
    
    # Set background color
    ax.set_facecolor(colors['background'])
    
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

def _plot_residuals(model, X, y, colors):
    y_pred = model.predict(X)
    residuals = y - y_pred
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=colors['background'])
    
    # Fun scatter with different marker
    ax.scatter(y_pred, residuals, 
              alpha=0.7,
              c=colors['primary'],
              edgecolor='white',
              s=120,
              marker='o')
    
    # Zero line with fun style
    ax.axhline(y=0, 
               color=colors['secondary'],
               linestyle='--',
               linewidth=3,
               label='Zero Line')
    
    # Playful styling
    ax.set_xlabel("Predicted Values", fontsize=12, color=colors['text'])
    ax.set_ylabel("Residuals", fontsize=12, color=colors['text'])
    ax.set_title("Residual Plot\nShark's Accuracy Check",
                fontsize=14,
                pad=20,
                color=colors['text'])
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.legend(fontsize=10)
    
    # Set background color
    ax.set_facecolor(colors['background'])
    
    plt.tight_layout()

def _plot_confusion_matrix(model, X, y, colors):
    try:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=colors['background'])
        disp = ConfusionMatrixDisplay.from_estimator(
            model, X, y,
            cmap='Blues',  # Clearer for confusion matrix
            ax=ax,
            colorbar=False
        )
        ax.set_title("Confusion Matrix\nShark's Score Card",
                    fontsize=14,
                    pad=20,
                    color=colors['text'])
        
        # Set background color
        ax.set_facecolor(colors['background'])
        
        plt.tight_layout()
    except Exception as e:
        print(f"Could not plot confusion matrix: {e}")

def _plot_roc(model, X, y, colors):
    try:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=colors['background'])
        RocCurveDisplay.from_estimator(model, X, y, ax=ax)
        ax.set_title("ROC Curve\nShark's Sensitivity Radar",
                    fontsize=14,
                    pad=20,
                    color=colors['text'])
        
        # Set background color
        ax.set_facecolor(colors['background'])
        
        plt.tight_layout()
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")

def _plot_pr_curve(model, X, y, colors):
    try:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=colors['background'])
        PrecisionRecallDisplay.from_estimator(model, X, y, ax=ax)
        ax.set_title("Precision-Recall Curve\nShark's Precision Tracker",
                    fontsize=14,
                    pad=20,
                    color=colors['text'])
        
        # Set background color
        ax.set_facecolor(colors['background'])
        
        plt.tight_layout()
    except Exception as e:
        print(f"Could not plot Precision-Recall curve: {e}")

def _plot_proba_hist(model, X, y, colors):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] == 2:
                pos_proba = proba[:, 1]
                fig, ax = plt.subplots(figsize=(8, 6), facecolor=colors['background'])
                ax.hist(pos_proba, bins=20, color=colors['primary'], alpha=0.8)
                ax.set_title("Predicted Probabilities (Positive Class)\nShark's Confidence Meter",
                            fontsize=14,
                            pad=20,
                            color=colors['text'])
                ax.set_xlabel("Probability of Positive Class", fontsize=12, color=colors['text'])
                ax.set_ylabel("Frequency", fontsize=12, color=colors['text'])
                ax.grid(True, alpha=0.3, color=colors['grid'])
                
                # Set background color
                ax.set_facecolor(colors['background'])
                
                plt.tight_layout()
            else:
                fig, ax = plt.subplots(figsize=(8, 6), facecolor=colors['background'])
                for i in range(proba.shape[1]):
                    ax.hist(proba[:, i], bins=20, alpha=0.5, 
                           label=f"Class {i}", 
                           color=colors['bars'][i % len(colors['bars'])])
                ax.legend()
                ax.set_title("Predicted Probabilities by Class\nShark's Confidence Meter",
                            fontsize=14,
                            pad=20,
                            color=colors['text'])
                ax.set_xlabel("Predicted Probability", fontsize=12, color=colors['text'])
                ax.set_ylabel("Frequency", fontsize=12, color=colors['text'])
                ax.grid(True, alpha=0.3, color=colors['grid'])
                
                # Set background color
                ax.set_facecolor(colors['background'])
                
                plt.tight_layout()
        else:
            print("Model does not support predict_proba; skipping probability histogram.")
    except Exception as e:
        print(f"Could not plot predicted probability histogram: {e}")

def _plot_feature_importance(model, feature_names, colors):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        ylabel = "Importance Score"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        if len(importances.shape) > 1:  # Multi-class logistic regression
            importances = importances.mean(axis=0)
        ylabel = "Absolute Coefficient"
    else:
        raise AttributeError("Model does not have feature_importances_ or coef_.")
    
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=colors['background'])
    
    # Colorful bars
    bar_colors = [colors['bars'][i % len(colors['bars'])] for i in range(len(importances))]
    ax.bar(range(len(importances)), 
           importances[indices],
           color=bar_colors)
    
    # Set feature names as x-tick labels
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    
    # Playful styling
    ax.set_title("Feature Importance\nShark's Favorite Features",
                fontsize=14,
                pad=20,
                color=colors['text'])
    ax.set_xlabel("Features", fontsize=12, color=colors['text'])
    ax.set_ylabel(ylabel, fontsize=12, color=colors['text'])
    ax.grid(True, alpha=0.3, color=colors['grid'])
    
    # Set background color
    ax.set_facecolor(colors['background'])
    
    plt.tight_layout()

# Example usage function
def get_sharkpy_colors() -> Dict[str, str]:
    """
    Returns the default SharkPy color scheme.
    
    Returns:
        Dictionary of color specifications
    """
    return SHARKPY_COLORS.copy()