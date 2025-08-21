Getting Started
===============

Installation
------------

.. tab-set::

   .. tab-item:: Basic Installation
      
      .. code-block:: bash
         
         pip install sharkpy

   .. tab-item:: Full Installation
      
      .. code-block:: bash
         
         pip install sharkpy[full]

   .. tab-item:: Development
      
      .. code-block:: bash
         
         pip install sharkpy[dev]

Your First SharkPy Project
--------------------------

.. code-block:: python
   :caption: Basic usage example

   from sharkpy import Shark
   import pandas as pd

   # Load your data
   data = pd.read_csv('your_data.csv')

   # Create a Shark instance
   shark = Shark()

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