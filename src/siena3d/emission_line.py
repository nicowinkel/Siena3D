"""
This file contains the EmissionLine class
"""
import numpy as np
import sys
from astropy.modeling import models
if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    import importlib.resources as importlib_resources


class EmissionLine:
    """A class which contains information about an emission line.
    This includes the parameters in the eline.par file and the
    the best-fit model parameters and its spectrum.

     Parameters
     ----------
    name : `string`
        emission line name
    component: 'string'
        kinematic component to which the line belongs, e.g. 'broad' or 'wing'
    tied: 'string'
        emission line to which the line is kinematically tied, e.g. 'Hb_broad' or 'OIII_wing'
    idx: 'int'
        index in compound model
    amplitude: 'float'
        amplitude
    amplitude_err: 'float'
        amplitude error
    vel : `float`, optional
        velocity [kms-1]
    vel_err : `float`, optional
        uncertainty of the velocity [kms-1]
    disp : `float`, optional
        velocity dispersion [kms-1]
    disp_err : 'float', optional
        uncertainty of the velocity dispersion [kms-1]
    model: 'astropy.model.Gaussian1D'
        astropy model with the parameters for the Gaussian1D
    wave: 'numpy.array', optional
        wavelength array of spectrum
    flux: 'numpy.array', optional
        flux array of spectrum
    error: 'numpy.array', optional
        error array of spectrum
    """

    def __init__(
                self,
                name=None,
                component=None,
                tied = None,
                idx = None,
                amplitude = np.array([np.nan]),
                amplitude_err =np.array([np.nan]),
                vel=np.array([np.nan]),
                vel_err=np.array([np.nan]),
                disp=np.array([np.nan]),
                disp_err=np.array([np.nan]),
                wavelength=np.nan,
                flux=np.array([np.nan]),
                error=np.array([np.nan]),
                model = np.nan
                ):

        self.c = 2.99792458e5
        self.name = name
        self.component = component
        self.tied = tied
        self.idx = idx
        self.amplitude = amplitude
        self.amplitude_err = amplitude_err
        self.vel = vel
        self.vel_err = vel_err
        self.disp = disp
        self.disp_err = disp_err
        self.wave_rest = self.get_waverf()
        self.flux = flux
        self.error = error
        self.model = self.get_model()

    def get_waverf(self):
        """Read in the rest-frame wavelength of single emission line

        Returns
        -------
        lambdarest: `float`
            emission line names with values of their rest-frame central wavelengths
        """

        pkg = importlib_resources.files("siena3d")
        pkg_eline_file = pkg / "data" / "eline_rf.txt"
        with pkg_eline_file.open() as f:
            lines = [line for line in f if not (line.startswith('#') or (line.split()==[]))]

        lambdarest = np.nan
        foundname = False
        for line in lines:
            line, wave = line.split()
            if  (line in self.name):
                lambdarest = float(wave)
                foundname = True

        if not foundname:
            raise ValueError('Emission line name not present int wave_rf.txt')

        return lambdarest

    def get_model(self):
        model = models.Gaussian1D(self.amplitude,
                                  self.wave_rest * (1 + self.vel / self.c),
                                  self.disp / self.c *  self.wave_rest
                                  )

        return model


class EmissionLineSet:
    """Contains dictionary of EmissionLine objects, called by their names;
    """
    def __init__(self):
        self.elines = {}

    def add_line(self, emission_line):
        """adds an emission line to an existing emission line set

         Parameters
         ----------
        name : `string`
            emission line name
        """
        self.elines[emission_line.name] = emission_line
        #self.check_shapes(emission_line.name)

    def check_shapes(self, emission_line_name):
        shape = self.elines[emission_line_name].flux.shape
        self.a_v = EmissionLine.check_shape(self.a_v, shape)
        self.sf_fraction = EmissionLine.check_shape(self.sf_fraction, shape)
        self.sf_fraction_error = EmissionLine.check_shape(self.sf_fraction_error, shape)

    def append(self, emission_line_set):
        result = EmissionLineSet()
        for line_name in list(self.elines.keys()):
            result.add_line(self.elines[line_name].append(emission_line_set.emission_lines[line_name]))
        for key in self.BPT.keys():
            result.BPT[key] = np.append(self.BPT[key], emission_line_set.BPT[key])
        result.a_v = np.append(self.a_v, emission_line_set.a_v)
        result.sf_fraction = np.append(self.sf_fraction, emission_line_set.sf_fraction)
        result.sf_fraction_error = np.append(self.sf_fraction_error, emission_line_set.sf_fraction_error)
        return result