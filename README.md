# MATLAB/SciPy Examples

<a href="https://github.com/adam-rumpf/matlab-scipy-examples/search?type=code"><img src="https://img.shields.io/badge/languages-matlab | python-blue"/></a> <a href="https://github.com/adam-rumpf/matlab-scipy-examples/releases"><img src="https://img.shields.io/github/v/tag/adam-rumpf/matlab-scipy-examples"/></a> <a href="https://github.com/adam-rumpf/matlab-scipy-examples/blob/master/LICENSE"><img src="https://img.shields.io/github/license/adam-rumpf/matlab-scipy-examples"/></a> <a href="https://github.com/adam-rumpf/matlab-scipy-examples/commits/main"><img src="https://img.shields.io/maintenance/yes/2021"/></a>

A collection of small scripts demonstrating how to accomplish mathematical tasks in both [MATLAB](https://www.mathworks.com/products/matlab.html) (or [Octave](https://www.gnu.org/software/octave/index)) and in [Python](https://www.python.org/) using [SciPy](https://scipy.org/).

## Description

This repo contains a collection of small scripts for common applications of mathematical software, such as numerically solving differential equations and plotting their solutions curves. It is primarily aimed at mathematicians whom are already used to using specialized software but wish to expand their programming options.

Commerical software packages like MATLAB can be very good for making common mathematical tasks simple, but they can also be somewhat limited. Being able to accomplish the same tasks in a fully-fledged programming language like Python allows access to a much wider variety of modules, and allows these mathematical applications to be built into a larger program as a subroutine.

## Contents

This repo contains a variety of directories, each of which includes two files: a MATLAB `.m` file and a Python `.py` file. The two files are meant to correspond to each other (line-for-line where possible), and to show how the same set of steps can be accomplished in each program. The following examples are included:

* [**Basics:**](https://github.com/adam-rumpf/matlab-scipy-examples/tree/main/basics) A variety of basic programming structures (such as loops, conditional statements, and functions) and mathematical commands (such as matrix and vector operations, solving linear systems, and creating plots).
* [**Logistic Map:**](https://github.com/adam-rumpf/matlab-scipy-examples/tree/main/logistic-map) Numerically generating the bifurcation diagram for the [logistic map](https://en.wikipedia.org/wiki/Logistic_map).
* [**Lorenz System:**](https://github.com/adam-rumpf/matlab-scipy-examples/tree/main/lorenz) Numerically solving the [Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system) and plotting the resulting curve in 3D space.
* [**Lotka-Volterra System:**](https://github.com/adam-rumpf/matlab-scipy-examples/tree/main/lotka-volterra) Numerically solving versions of the [Lotka-Volterra](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) (predator/prey) system, and plotting the results in the phase plane.
* [**SIR Model:**](https://github.com/adam-rumpf/matlab-scipy-examples/tree/main/sir) Numerically solving the [SIR](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model_without_vital_dynamics) infectious disease model, and plotting the results.
* [**SVD:**](https://github.com/adam-rumpf/matlab-scipy-examples/tree/main/svd) Computing the [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) of a matrix and illustrating the effects of the matrix on the unit ball.
