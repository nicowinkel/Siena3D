"""
This file contains the EmissionLine class
"""
import numpy as np


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
        self.model = model
        self.wavelength = wavelength
        self.flux = flux
        self.error = error
        self.model = model




class EmissionLineSet:
    def __init__(self):
        """
        Emission Line Set class:
        contains dictionary of EmissionLine objects, called by their names;
        """
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