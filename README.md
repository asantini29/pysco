# pysco: PYthon ShortCuts & Others

`pysco` is a Python module that contains some plotting routines I usually use and other useful snippets.

## Installing:
1. Clone the repository:
   ```
   git clone https://github.com/asantini29/pysco.git   
   cd pysco
   ```
3. Run install:
   ```
   python setup.py install
   ```

## Tools:
Some of the main routines currently implemented in `pysco` :
1. **plot**: `pysco.plot` contains custom settings for `matplotlib`([here](https://matplotlib.org/stable/)) and `corner`([here](https://corner.readthedocs.io/)). Currently, some features of `pysco.plot.corner` are available only when using the `dev` branch of my edited fork of `corner`(available [here](https://github.com/asantini29/corner.py)). In order to use another custom fork of `corner` you have to set its path as an environment variable:
   
   ```
   export CORNER_PATH=your-path-to-corner.py
   ```
   The module contains three different colorblind-friendly color palettes based on the results of [arXiv:2107.02270](https://arxiv.org/abs/2107.02270).
   
3. **utils**: `pysco.utils` contains basic timing and benchmarking operations.

4. **eryn**: the module `pysco.eryn` contains useful routines for the [Eryn](https://github.com/mikekatz04/Eryn) MCMC sampler. Most of the snippets currently implemented are tailored for a diagnostic plots-oriented `update_fn`.

5. **lisautils**: This module contains common samples-related operations for LISA data analysis.

## Versioning

We use [SemVer](http://semver.org/) for versioning. 

Current Version: 0.0.5

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
