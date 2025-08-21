# docs/conf.py
import os
import sys
import datetime
from importlib.metadata import version as get_version

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'SharkPy'
copyright = f'{datetime.datetime.now().year}, Ezz Eldin Ahmed'
author = 'Ezz Eldin Ahmed'

# The short X.Y version
version = get_version('sharkpy').split('+')[0]
# The full version, including alpha/beta/rc tags
release = get_version('sharkpy')

# -- General configuration ---------------------------------------------------
extensions = [
    # Core extensions
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    
    # Documentation enhancements
    'numpydoc',
    'sphinx_rtd_theme',
    'autoapi.extension',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinxext.opengraph',
    'sphinx_issues',
    'sphinx_panels',
    'sphinxcontrib.mermaid',
    'sphinx_gallery.gen_gallery',
    'sphinxcontrib.bibtex',
    'sphinx_togglebutton',
    'sphinx_favicon',
]

# AutoAPI configuration
autoapi_type = 'python'
autoapi_dirs = ['../sharkpy']
autoapi_add_toctree_entry = False
autoapi_keep_files = False
autoapi_root = 'api'
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]

# MyST configuration
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'linkify',
    'replacements',
    'smartquotes',
    'substitution',
    'tasklist',
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Sphinx Gallery for examples
sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'filename_pattern': r'\.py$',
    'ignore_pattern': r'__init__\.py',
    'min_reported_time': 5,
    'download_all_examples': False,
    'within_subsection_order': None,
    'backreferences_dir': None,
    'doc_module': ('sharkpy',),
    'reference_url': {'sharkpy': None},
}

# BibTeX for references
bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'plain'

# Templates and excludes
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['custom.js']

# Theme options
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2E86AB',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Favicon
html_favicon = '_static/favicon.ico'

# OpenGraph
ogp_site_url = "https://sharkpy.readthedocs.io/"
ogp_image = "_static/logo.png"

# -- Options for PDF output --------------------------------------------------
latex_engine = 'xelatex'
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': r'''
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{booktabs}
        \usepackage{multirow}
        \usepackage{graphicx}
    ''',
}

# -- Custom setup ------------------------------------------------------------
def setup(app):
    app.add_css_file('custom.css')
    app.add_js_file('custom.js')