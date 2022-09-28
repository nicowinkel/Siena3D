# Siena3D

## Scope
SIENA is a tool to analyse the morphology of the extended emission line region morphology in 3D spectroscopic observations of AGN.
It performs a multi-Gaussian fit of the AGN emission line spectrum and traces different kinematic components across the field of view by keeping the kinematic parameters tied. Furthermore, SIENA utilizes the emprical PSF extracted from the data to measure the projected size, location and luminosity of the emitting structure. SIENA is developed based on observations of unobscured (type 1) AGN that were aquired with
ESO VLT/MUSE. The technique can in principle be to 3D cubes obtained with other IFU instruments.


SIENA is currently being developed.

## Installation instructions
The easiest way to install Siena3D is to install the the PyPi Package: "pip install Siena3D"

Alternatively one can download the simply clone the repository:
- "git clone https://github.com/nicowinkel/siena.git" or download the .zip file from https://github.com/nicowinkel/siena

- Go to the "Siena3D-main" sub-directory and install the package with "pip install ."


## Basic usage and commands
The code is run entirely from the command line, and is set up to run on the files that are located relative to the working directory.
An demo is provided in the "example/" folder where also the parameter files and an example data cube are located.

SIENA can be run by executing the runall.py file: "python runall.py".

*** More information tba ***

## Manual for more information on parameter file setup
*** Information tba soon ***
