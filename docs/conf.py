import os
import sys
from pathlib import Path
import tomllib

sys.path.insert(0, os.path.abspath("../src"))

project = "pysco"
author = "Alessandro Santini"
copyright = "2026, Alessandro Santini"

pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
with pyproject.open("rb") as f:
    release = tomllib.load(f)["project"]["version"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_mock_imports = [
    "numpy",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.colors",
    "matplotlib.ticker",
    "pandas",
    "corner",
    "GPUtil",
    "chainconsumer",
    "scipy",
    "scipy.special",
    "tqdm",
    "h5py",
    "eryn",
    "eryn.utils",
    "eryn.backends",
    "eryn.moves",
    "cycler",
]

html_theme = "sphinx_rtd_theme"
