import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "manolo-bot"
copyright = "2024, Carlos Cesar Caballero Díaz"
author = "Carlos Cesar Caballero Díaz"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
