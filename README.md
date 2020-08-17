turboPy
=======================
[![DOI](https://zenodo.org/badge/268071520.svg)](https://zenodo.org/badge/latestdoi/268071520)
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

-   [Official turboWAVE Repo](https://github.com/USNavalResearchLaboratory/turboWAVE)
-   [TurboWAVE Documentation](https://turbowave.readthedocs.io)
-   The current html turboPy documentation can be found [here](https://turbopy.readthedocs.io/en/latest)
-   The preprint of the turboPy paper can be found [here](https://arxiv.org/pdf/2002.08842.pdf)


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
