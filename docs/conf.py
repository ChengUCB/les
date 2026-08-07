# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "LES"
copyright = "2026, Cheng Group"
author = "Cheng Group"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]
myst_enable_extensions = [
    "html_admonition",
    "dollarmath",
]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autosummary_generate = True
source_suffix = [".rst", ".md"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "e3nn": ("https://docs.e3nn.org/en/stable/", None),
    "nequip": ("https://nequip.readthedocs.io/en/latest/", None),
    "allegro": ("https://nequip.readthedocs.io/projects/allegro/en/latest/", None),
    "nequip_les": ("https://nequip-les.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
