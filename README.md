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
2. **performance**: perform basic timing and benchmarking operations.

3. **eryn**: this module contains useful routines for the `Eryn` [MCMC sampler](https://github.com/mikekatz04/Eryn). Most of the snippets currently implemented are tailored for a diagnostic plots-oriented `update_fn`.
