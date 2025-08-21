# 🦈 SharkPy

A friendly machine learning framework with shark-themed feedback! SharkPy simplifies the machine learning workflow while making it fun and educational.

## Features

- **Model Battle Arena**: Compare multiple models automatically
- **Smart Reporting**: Generate reports in TXT, PDF, and DOCX formats
- **Model Visualization**: Beautiful plots with shark-themed styling
- **Model Explanations**: Clear explanations of model behavior
- **Automated Optimization**: Hyperparameter tuning with Optuna
- **Shapash Integration**: Interactive dashboards for model interpretation

## Quick Start

### Installation

```bash
# Basic installation
pip install sharkpy

# Full installation (includes LightGBM and CatBoost)
pip install sharkpy[full]

# Development installation
pip install sharkpy[dev]
```

### Basic Usage

```python
from sharkpy import Shark
import pandas as pd

# Create a Shark instance
shark = Shark()

# Load your data
data = pd.read_csv('your_data.csv')

# Train a model
shark.learn(
    data=data,
    target='target_column',
    model_choice='random_forest'
)

# Make predictions
predictions = shark.predict(new_data)

# Generate reports
shark.report(export_path='report.pdf', format='pdf')
```

### Model Battle Example

```python
# Define models to compete
models = [
    'linear_regression',
    'random_forest',
    'xgboost',
    'lightgbm',
    'catboost'
]

# Let them battle!
battle_results = shark.battle(
    data=data,
    target='target_column',
    models=models,
    metric='r2'
)

# Get the champion
print(f"Winner: {battle_results['champion']}")
print(f"Score: {battle_results['score']:.4f}")
```

## Supported Models

- Linear Regression
- Logistic Regression
- Random Forest
- Support Vector Machines
- Ridge Regression
- Lasso Regression
- K-Nearest Neighbors
- XGBoost
- LightGBM (with full installation)
- CatBoost (with full installation)

## Reports

SharkPy can generate comprehensive reports in multiple formats:

```python
# Text report
shark.report(export_path='report.txt', format='txt')

# PDF report (requires MS Word or LibreOffice)
shark.report(export_path='report.pdf', format='pdf')

# Word document
shark.report(export_path='report.docx', format='docx')
```

## Visualizations

```python
# Prediction plot
shark.plot(kind="prediction")

# Feature importance
shark.plot(kind="feature_importance")

# Interactive Shapash dashboard
shark.explain_with_shapash()
```

## Model Explanations

```python
# Get friendly explanations of your model
shark.explain()

# Create interactive Shapash dashboard
shark.explain_with_shapash(title_story="My Analysis")
```

## Save & Load Models

```python
# Save your model
shark.save_model(name="my_model")

# Load a saved model
shark.load_model("models/my_model.joblib")
```

## Development

```bash
# Install development dependencies
pip install sharkpy[dev]

# Run tests
pytest

# Format code
black sharkpy/
```

## Documentation

Full documentation is available at [sharkpy.readthedocs.io](https://sharkpy.readthedocs.io/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with scikit-learn, XGBoost, and other amazing open-source tools
- Inspired by making machine learning more accessible and fun

## Contact

- Author: Ezz Eldin Ahmed
- Email: ezzeldinahmad96@gmail.com
- GitHub: [Ezzio11](https://github.com/Ezzio11)

---

Made with 🦈 by SharkPy Team

