# MATLAB/SciPy Examples

<a href="https://github.com/adam-rumpf/matlab-scipy-examples/search?type=code"><img src="https://img.shields.io/badge/languages-matlab | python-blue"/></a> <a href="https://github.com/adam-rumpf/matlab-scipy-examples/releases"><img src="https://img.shields.io/github/v/tag/adam-rumpf/matlab-scipy-examples"/></a> <a href="https://github.com/adam-rumpf/matlab-scipy-examples/blob/main/LICENSE"><img src="https://img.shields.io/github/license/adam-rumpf/matlab-scipy-examples"/></a> <a href="https://github.com/adam-rumpf/matlab-scipy-examples/commits/main"><img src="https://img.shields.io/maintenance/yes/2022"/></a>

A collection of small scripts demonstrating how to accomplish mathematical tasks in both [MATLAB](https://www.mathworks.com/products/matlab.html) (or [Octave](https://www.gnu.org/software/octave/index)) and in [Python](https://www.python.org/) using [SciPy](https://scipy.org/), [NumPy](https://numpy.org/), and [Matplotlib](https://matplotlib.org/).

<p align="center"><a href="https://github.com/adam-rumpf/matlab-scipy-examples/tree/main/basics"><img src="img/basics_cover.png" height="300px" title="Basic 3D plot demo." /></a></p>

## Description

This repo is primarily aimed at mathematicians and scientists whom are already used to using mathematical software packages for performing scientific computations, but whom wish to expand their options by working in a fully-fledged programming language. Commercial software packages can be somewhat limited in their functionality, and being able to perform the same tasks using a general programming language not only grants access to a much wider array of packages, but also allows these mathematical applications to be built into larger programs.

This repo contains a collection of example scripts to show how common mathematical tasks can be performed in MATLAB and in Python, with the goal of demonstrating how to convert a MATLAB program into an equivalent Python program. This is accomplished primarily through the use of the [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) object and the [`linalg`](https://numpy.org/doc/stable/reference/routines.linalg.html) module from [NumPy](https://numpy.org/), which provide a MATLAB-like syntax for working with vectors and matrices, and the [`pyplot`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html) module from [Matplotlib](https://matplotlib.org/), which provides a MATLAB-like syntax for creating plots.

Each subdirectory contains MATLAB `.m` files and Python `.py` files. The two files are meant to correspond to each other (line-for-line where possible) and to show how the same set of steps can be accomplished in each program. The files named after the directory are standalone versions of the MATLAB and Python scripts, while the files with an `_appended` suffix include both versions alongside each other to make the line-by-line comparison more clear.

## Contents

* [The Basics](#the-basics)
* [The Lorenz System](#the-lorenz-system)

### The Basics

<p align="center"><img src="img/basics_m.png" height="200px" title="MATLAB 3D plot." /> <img src="img/basics_py.png" height="200px" title="Python 3D plot." /></p>

[Directory Link](https://github.com/adam-rumpf/matlab-scipy-examples/tree/main/basics)

Demonstrations for a variety of basic programming structures (such as loops, conditional statements, and functions) and mathematical commands (such as matrix and vector operations, solving linear systems, and creating plots).

### The Lorenz System

<p align="center"><img src="img/lorenz_m.png" height="200px" title="MATLAB Lorenz system plots." /> <img src="img/lorenz_py.png" height="200px" title="Python Lorenz system plots." /></p>

[Directory Link](https://github.com/adam-rumpf/matlab-scipy-examples/tree/main/lorenz)

An example of numerically solving the [Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system) of ordinary differential equations and plotting the resulting curve in 3D space viewed from various perspectives. This demonstrates the general approach for numerically solving initial value problems as well as displaying figures arranged in a grid.
