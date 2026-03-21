# pysco: PYthon ShortCuts & Others

[![Documentation Status](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://asantini29.github.io/pysco/)

`pysco` is a Python module that contains some plotting routines I usually use and other useful snippets.

## Installing:
1. Clone the repository:
   ```
   git clone https://github.com/asantini29/pysco.git
   cd pysco
   ```
2. Install [uv](https://docs.astral.sh/uv/) if you haven't already:
   ```
   # on MacOS or Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # on Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
3. Install the package and its dependencies using `uv`:
   ```
   uv sync
   ```
   To include the optional `eryn` dependencies:
   ```
   uv sync --extra eryn
   ```

## Usage:
Run Python commands using `uv run` to ensure the correct environment is used:
```
uv run python your_script.py
```

## Package structure
The package follows a `src` layout:
```
pysco/
├── src/pysco/
│   ├── __init__.py
│   ├── pysco.py
│   ├── utils.py
│   ├── eryn.py
│   └── plots/
│       ├── __init__.py
│       ├── plot.py
│       └── journals.py
│       └── mplfiles/
```

## Tools
Some of the main routines currently implemented in `pysco`:
1. **plots**: `pysco.plots` contains custom settings for `matplotlib`([here](https://matplotlib.org/stable/)) and `corner`([here](https://corner.readthedocs.io/)). Currently, some features of `pysco.plots.corner` are available only when using the `dev` branch of my edited fork of `corner`(available [here](https://github.com/asantini29/corner.py)). In order to use another custom fork of `corner` you have to set its path as an environment variable:
   
   ```
   export CORNER_PATH=your-path-to-corner.py
   ```
   The module contains three different colorblind-friendly color palettes based on the results of [arXiv:2107.02270](https://arxiv.org/abs/2107.02270).
   
2. **utils**: `pysco.utils` contains basic timing and benchmarking operations.

3. **eryn**: the module `pysco.eryn` contains useful routines for the [Eryn](https://github.com/mikekatz04/Eryn) MCMC sampler. Most of the snippets currently implemented are tailored for a diagnostic plots-oriented `update_fn`.

## Documentation
Sphinx documentation is available under `docs/` and can be built locally with:
```
python -m sphinx -b html docs docs/_build/html
```
Built documentation is available on GitHub Pages: <https://asantini29.github.io/pysco/>

## Versioning

We use [SemVer](http://semver.org/) for versioning. 

Current Version: 0.1.1 (see [src/pysco/pysco.py](src/pysco/pysco.py))

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Citing

If you use `pysco` in your research, you can cite it in the following way:

```
@software{pysco_2024_13930440,
  author       = {Alessandro Santini},
  title        = {asantini29/pysco: First release},
  month        = oct,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.13930440},
  url          = {https://doi.org/10.5281/zenodo.13930440}
}
```
