import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project   = 'GeoXGB'
copyright = '2025, Jake Peace'
author    = 'Jake Peace'
release   = '0.2.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

templates_path   = ['_templates']
exclude_patterns = []
source_suffix    = {'.rst': 'restructuredtext', '.md': 'markdown'}

html_theme        = 'sphinx_rtd_theme'
html_static_path  = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
}

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring  = True
