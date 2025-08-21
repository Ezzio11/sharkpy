# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'SharkPy'
copyright = '2025, Ezz Eldin Ahmed'
author = 'Ezz Eldin Ahmed'

release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'autoapi.extension',
]

autoapi_type = 'python'
autoapi_dirs = ['../sharkpy']
autoapi_add_toctree_entry = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']