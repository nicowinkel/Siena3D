# Siena3D

![Siena3D](https://github.com/nicowinkel/Siena3D/blob/main/src/siena3d/data/logo.png)

## Scope
Siena3D is a tool to analyse the morphology of the ionised gas emission in 3D spectroscopic data cubes.
It uses a set of emission lines to decompose an input spectrum and trace the components via their kinematics.
This allows to resolve and  characterize the individual components with a sub-pixel resolution,
well below the width of the point spread function.

Siena3D was originally developed to analyse the extended emission line region (EELR) in unobscured AGN obtained with
ESO VLT/MUSE. However, the method may also be applied to other science cases or IFU data that were aquired with other instruments.

## Installation
The easiest way to install is to first create a dedicated conda environment prior to the installation:

    conda create --name siena3d python=3.9
    conda activate Siena3D

In order to install Siena3D, first clone the repository with

    git clone https://github.com/nicowinkel/Siena3D

Go to the directory Siena3D and install the package with

    pip install .

In addition, the user needs to manually install the Maoppy package with the commnand

    pip install git+https://gitlab.lam.fr/lam-grd-public/maoppy.git@master

More information on the Maoppy package can be found [here](https://gitlab.lam.fr/lam-grd-public/maoppy).

## Basic usage
The Siena3D code can be run entirely from the command line.  It is set up such that the files are placed in the "Input"
and "Output" subdirectories of the working directory.
A demo is provided in the example directory where also the parameter files together with a set of example data cube are provided.
To test the installation and run the example,  go to the example directory and execute:

    python runall.py

We refer to the Siena3D user manual for more information on running Siena3D and the underlying algorithms.

## Manual for more information on parameter file setup
More details on the underlying algorithms and principles of Siena3D can be found in the
[Siena3D User Manual](https://github.com/nicowinkel/Siena3D/blob/main/docs/Siena3D_User_Manual.pdf).
It provides an overview of the underlying algorithm, requirements for the input data as well as options and tips
on how to setup the parameter files.

![Siena3D](https://github.com/nicowinkel/Siena3D/blob/main/src/siena3d/data/schematic.png)
