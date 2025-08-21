# sharkpy/plotting.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.font_manager as fm
import warnings

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

# Updated color scheme
COLORS = {
    'scatter': '#E57373',  # Softer red
    'line': '#4ECDC4',     # Turquoise (unchanged)
    'bars': ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F'],  # Set2-inspired
    'grid': '#D3D3D3'      # Slightly darker gray
}

def plot_model(model, X, y=None, kind="prediction", show=True, save_path=None, feature_names=None):
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
    """
    print(f"🦈 Debug: plot_model called with kind={kind}")
    if kind in ["prediction", "residuals", "confusion_matrix", "roc", "pr_curve", "proba_hist"] and y is None:
        raise ValueError(f"y must be provided for kind='{kind}'.")

    # Dispatch to plotting functions
    if kind == "prediction":
        _plot_prediction(model, X, y)
    elif kind == "residuals":
        _plot_residuals(model, X, y)
    elif kind == "confusion_matrix":
        _plot_confusion_matrix(model, X, y)
    elif kind == "roc":
        _plot_roc(model, X, y)
    elif kind == "pr_curve":
        _plot_pr_curve(model, X, y)
    elif kind == "proba_hist":
        _plot_proba_hist(model, X, y)
    elif kind == "feature_importance":
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns
            else:
                raise ValueError("Feature names required for feature importance plot.")
        _plot_feature_importance(model, feature_names)
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

def _plot_prediction(model, X, y):
    y_pred = model.predict(X)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    
    # Fun scatter plot with star markers
    ax.scatter(y, y_pred, 
              alpha=0.7, 
              c=COLORS['scatter'],
              edgecolor='white',
              s=120,
              marker='*')
    
    # Perfect prediction line with fun style
    line_range = [min(y), max(y)]
    ax.plot(line_range, line_range, 
            '--', 
            color=COLORS['line'],
            linewidth=3,
            label='Perfect Prediction')
    
    # Playful styling
    ax.set_xlabel("Actual Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.set_title("Actual vs Predicted Values\nShark's Predictions", 
                fontsize=14, 
                pad=20)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    ax.legend(fontsize=10)
    
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

def _plot_residuals(model, X, y):
    y_pred = model.predict(X)
    residuals = y - y_pred
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    
    # Fun scatter with different marker
    ax.scatter(y_pred, residuals, 
              alpha=0.7,
              c=COLORS['scatter'],
              edgecolor='white',
              s=120,
              marker='o')
    
    # Zero line with fun style
    ax.axhline(y=0, 
               color=COLORS['line'],
               linestyle='--',
               linewidth=3,
               label='Zero Line')
    
    # Playful styling
    ax.set_xlabel("Predicted Values", fontsize=12)
    ax.set_ylabel("Residuals", fontsize=12)
    ax.set_title("Residual Plot\nShark's Accuracy Check",
                fontsize=14,
                pad=20)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    ax.legend(fontsize=10)
    plt.tight_layout()

def _plot_confusion_matrix(model, X, y):
    try:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        disp = ConfusionMatrixDisplay.from_estimator(
            model, X, y,
            cmap='Blues',  # Clearer for confusion matrix
            ax=ax
        )
        ax.set_title("Confusion Matrix\nShark's Score Card",
                    fontsize=14,
                    pad=20)
        plt.tight_layout()
    except Exception as e:
        print(f"Could not plot confusion matrix: {e}")

def _plot_roc(model, X, y):
    try:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        RocCurveDisplay.from_estimator(model, X, y, ax=ax)
        ax.set_title("ROC Curve\nShark's Sensitivity Radar",
                    fontsize=14,
                    pad=20)
        plt.tight_layout()
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")

def _plot_pr_curve(model, X, y):
    try:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        PrecisionRecallDisplay.from_estimator(model, X, y, ax=ax)
        ax.set_title("Precision-Recall Curve\nShark's Precision Tracker",
                    fontsize=14,
                    pad=20)
        plt.tight_layout()
    except Exception as e:
        print(f"Could not plot Precision-Recall curve: {e}")

def _plot_proba_hist(model, X, y):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] == 2:
                pos_proba = proba[:, 1]
                fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
                ax.hist(pos_proba, bins=20, color=COLORS['scatter'], alpha=0.8)
                ax.set_title("Predicted Probabilities (Positive Class)\nShark's Confidence Meter",
                            fontsize=14,
                            pad=20)
                ax.set_xlabel("Probability of Positive Class", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.grid(True, alpha=0.3, color=COLORS['grid'])
                plt.tight_layout()
            else:
                fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
                for i in range(proba.shape[1]):
                    ax.hist(proba[:, i], bins=20, alpha=0.5, label=f"Class {i}", color=COLORS['bars'][i % len(COLORS['bars'])])
                ax.legend()
                ax.set_title("Predicted Probabilities by Class\nShark's Confidence Meter",
                            fontsize=14,
                            pad=20)
                ax.set_xlabel("Predicted Probability", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.grid(True, alpha=0.3, color=COLORS['grid'])
                plt.tight_layout()
        else:
            print("Model does not support predict_proba; skipping probability histogram.")
    except Exception as e:
        print(f"Could not plot predicted probability histogram: {e}")

def _plot_feature_importance(model, feature_names):
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
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    # Colorful bars
    ax.bar(range(len(importances)), 
           importances[indices],
           color=COLORS['bars'])
    
    # Set feature names as x-tick labels
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    
    # Playful styling
    ax.set_title("Feature Importance\nShark's Favorite Features",
                fontsize=14,
                pad=20)
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    plt.tight_layout()