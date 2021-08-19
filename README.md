turboPy
=======================
[![DOI](https://zenodo.org/badge/268071520.svg)](https://zenodo.org/badge/latestdoi/268071520)
[![PyPI version](https://badge.fury.io/py/turbopy.svg)](https://badge.fury.io/py/turbopy)
[![Documentation Status](https://readthedocs.org/projects/turbopy/badge/?version=latest)](https://turbopy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/NRL-Plasma-Physics-Division/turbopy.svg?branch=main)](https://travis-ci.org/NRL-Plasma-Physics-Division/turbopy)
[![codecov](https://codecov.io/gh/NRL-Plasma-Physics-Division/turbopy/branch/main/graph/badge.svg)](https://codecov.io/gh/NRL-Plasma-Physics-Division/turbopy)
![GitHub](https://img.shields.io/github/license/NRL-Plasma-Physics-Division/turbopy)

A lightweight computational physics framework, based on the organization of turboWAVE. Implements a "Simulation, PhysicsModule, ComputeTool" class hierarchy.

Motivation
----------

Computational physics problems often have a common set of aspects to them that any particular numerical code will have to address. Because these aspects are common to many problems, having a framework already designed and ready to use will not only speed the development of new codes, but also enhance compatibility between codes. 

Some of the most common aspects of computational physics problems are: a grid, a clock which tracks the flow of the simulation, and a set of models describing the dynamics of various quantities on the grid. Having a framework that could deal with these basic aspects of the simulation in a common way could provide great value to computational scientists by solving various numerical and class design issues that routinely arise.

This paper describes the newly developed computational framework that we have built for rapidly prototyping new physics codes. This framework, called turboPy, is a lightweight physics modeling framework based on the design of the particle-in-cell code `turboWAVE`. It implements a class (called `Simulation`) which drives the simulation and manages communication between physics modules, a class (called `PhysicsModule`) which handles the details of the dynamics of the various parts of the problem, and some additional classes such as a `Grid` class and a `Diagnostic` class to handle various ancillary issues that commonly arise.


More Resources
--------------

-   [Online turboPy documentation](https://turbopy.readthedocs.io/en/latest)
-   [The published turboPy paper](https://doi.org/10.1016/j.cpc.2020.107607)
-   [Official turboWAVE repository](https://github.com/USNavalResearchLaboratory/turboWAVE)
-   [TurboWAVE Documentation](https://turbowave.readthedocs.io)


Install turboPy
---------------

-   Install: `pip install turbopy`


turboPy development environment
-------------------------------

-   Create a conda environment for turboPy: `conda env create -f environment.yml`
-   Activate: `conda activate turbopy`
-   Install turboPy in editable mode (i.e. setuptools "develop mode") if you are modifying turboPy itself: `pip install -e .` 
-   Run tests: `pytest`

If using `pylint` (which you should!) add `variable-rgx=[a-z0-9_]{1,30}$` to your .pylintrc file to allow single character variable names.

Merge requests are encouraged!



Attribution
-----------

If you use turboPy for a project, first, you're awesome, thanks! :tada:

Also, we would appreciate it if you would cite turboPy. There are a few ways you can cite this project. 

#### Cite a specific version

If you used turboPy, please cite the specific version of the code that you used. We use Zenodo to create DOIs and archive our code. You can find the DOI for the version that you used on [our Zenodo page](https://doi.org/10.5281/zenodo.3973692)

For example, a citation for version v2020.10.14 should look like this:

> A. S. Richardson, P. E. Adamson, G. Tang, A. Ostenfeld, G. T. Morgan, C. G. Sun, D. J. Watkins, O. S. Grannis, K. L. Phlips, and S. B. Swanekamp. (2020, October 14). NRL-Plasma-Physics-Division/turbopy: v2020.10.14 (Version v2020.10.14). Zenodo. https://doi.org/10.5281/zenodo.4088189

While bibtex styles vary, the above output can be created by an entry something like this:

```bibtex
@software{turbopy_v2020.10.14,
	author = {A. S. Richardson and P. E. Adamson and G. Tang and A. Ostenfeld and G. T. Morgan and C. G. Sun and D. J. Watkins and O. S. Grannis and K. L. Phlips and S. B. Swanekamp},
	doi = {10.5281/zenodo.4088189},
	month = {October 14},
	publisher = {Zenodo},
	title = {{NRL-Plasma-Physics-Division/turbopy: v2020.10.14}},
	url = {https://doi.org/10.5281/zenodo.4088189},
	version = {v2020.10.14},
	year = 2020,
}
```

Note that the author names above have been lightly edited to put them in a standard format. The Zenodo page for a specific version of the code will try to infer author names from the GitHub accounts of contributers. That author list would be fine, too, espcially as additional GitHub users contribute to turboPy.

#### Cite the turboPy project

If you are refering to the turboPy project as a whole, rather than a specific version of the code, you also have the option of using the [Zenodo concept DOI](http://help.zenodo.org/#versioning) for turboPy. This DOI always resolves to the latest version of the code. 

The concept DOI for turboPy is [10.5281/zenodo.3973692](https://doi.org/10.5281/zenodo.3973692).

An example citation and bibtex entry for the concept DOI would look something like this:

> The turboPy Development Team. NRL-Plasma-Physics-Division/turbopy. Zenodo. https://doi.org/10.5281/zenodo.3973692

```bibtex
@software{turbopy_project,
	author = {The turboPy Development Team},
	doi = {10.5281/zenodo.3973692},
	publisher = {Zenodo},
	title = {{NRL-Plasma-Physics-Division/turbopy}},
	url = {https://doi.org/10.5281/zenodo.3973692},
	year = 2020,
}
```

#### Cite the turboPy paper

If you are looking for a paper to cite, rather than source code, this is the reference to use. This citation is appropriate when you are discussing the functionality of turboPy.

> A. S. Richardson, D. F. Gordon, S. B. Swanekamp, I. M. Rittersdorf, P. E. Adamson, O. S. Grannis, G. T. Morgan, A. Ostenfeld, K. L. Phlips, C. G. Sun, G. Tang, and D. J. Watkins. TurboPy: A lightweight python framework for computational physics. _Computer Physics Communications_, 258:107607, January 2021. https://doi.org/10.1016/j.cpc.2020.107607

```bibtex
@article{RICHARDSON2020107607,
	author = {A. S. Richardson and D. F. Gordon and S. B. Swanekamp and I. M. Rittersdorf and P. E. Adamson and O. S. Grannis and G. T. Morgan and A. Ostenfeld and K. L. Phlips and C. G. Sun and G. Tang and D. J. Watkins},
	doi = {10.1016/j.cpc.2020.107607},
	issn = {0010-4655},
	journal = {Computer Physics Communications},
	keywords = {Framework, Physics, Computational physics, Python, Dynamic factory pattern, Resource sharing},
	month = {January},
	pages = {107607},
	title = {{TurboPy}: A lightweight python framework for computational physics},
	url = {http://www.sciencedirect.com/science/article/pii/S0010465520302897},
	volume = {258},
	year = {2021},
}
```

