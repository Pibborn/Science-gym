# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = 'science-gym'
copyright = '2025, Mattia Cerrato'
author = 'Mattia Cerrato'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'sphinx.ext.githubpages', 'sphinx.ext.autodoc', 
              'sphinx.ext.napoleon', 'sphinx_design']
autosummary_generate = True
autodoc_mock_imports = ['sympy', 'scipy', 'Box2D', 'cv2', 'pygame', 'pyglet',
    'stable_baselines3', 'stable_baselines', 'gym', 'gymnasium', 'torch',
    'pgu', 'PyQt4', 'numpy', 'matplotlib', 'pandas', 'ScaleEnvironment']
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_static_path = ['_static']
html_theme_options = {
    "nav_links": [
        {
            "title": "Quickstart",
            "url": "usage/quickstart"
        },
    ]
}
