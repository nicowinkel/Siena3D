"""
This file contains the spectrum class
"""

from .cube import Cube
from .emission_line import EmissionLine, EmissionLineSet
import siena3d.plot

import numpy as np
import sys
from pathlib import Path
from astropy.modeling import models, fitting
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_lines
from astropy.nddata.nduncertainty import StdDevUncertainty

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    import importlib.resources as importlib_resources



class Spectrum(Cube):
    """
    A class representing 1D spectra.

    `Spectra` is a subclass of Cube which allows for handling and organizing of
    one-dimensional spectrum. The class supports a multi-Gaussian model fitting
    of AGN spectra.

    Parameters
    ----------
    cube : `siena.cube`
        cube object from which attributes can be inherited
    wvl_start : `float`, optional
        lower wavelength limit of the AGN spectrum modelling.
    wvl_end : `float`, optional
        lower wavelength limit of the AGN spectrum modelling.
    """

    def __init__(self, cube, par):

        # init parameters
        self.par = par
        self.fit_range = (self.par.wvl_start*u.Angstrom, self.par.wvl_end*u.Angstrom)     # wavelength window for fit

        # inherited instances
        self.wvl = getattr(cube, 'wvl')                                 # rest-frame wavelength
        self.AGN_spectrum = getattr(cube, 'AGN_spectrum')               # full AGN spectrum
        self.AGN_error = getattr(cube, 'AGN_error')                     # full AGN error spectrum
        self.cz = getattr(par, 'cz')
        self.c = 2.99792458e5

        # load data from input files in working directory
        self.lambdarest = self.load_lambdarest()                        # rest-frame wavelengths of emission lines
        self.elines = self.load_elines()                                # loads eline.par file
        self.components = self.load_components()

        self.incl_cont = self.load_incl_cont()                          # loads eline.par file

        # setup emission line models
        self.eline_models = self.setup_eline_models()                   # generates astropy models
        self.compound_model = self.setup_compound_model()               # combines all models to one
        self.couple_amplitudes()                                        # couples line ratios
        self.couple_kinematics()                                        # kin. coupling as specified in eline.par file

    def load_lambdarest(self):
        """Read in the rest-frame wavelengths of the emission lines
        required to model the AGN spectrum. File comes with package.

        Returns
        -------
        lambdarest: `dictionary`
            emission line names with values of their rest-frame central wavelengths
        """

        pkg = importlib_resources.files("siena3d")
        pkg_eline_file = pkg / "data" / "eline_rf.txt"
        with pkg_eline_file.open() as f:
            lines = [line for line in f if not (line.startswith('#') or (line.split()==[]))]

        lambdarest = {}
        for line in lines:
            line, wave = line.split()
            lambdarest[line] = float(wave)


        return lambdarest

    def load_components(self):
        """Read in the emission line parameters file

       Returns
       -------
       components: `dictionary`
           dictionary with the list of emission lines contained in each of the kinematic components
       """

        elines_file = self.par.elines_par
        with open(elines_file) as f:
            lines = [line for line in f if not (line.startswith('#') or (line.split() == []))]

        components = {}

        for idx, line in enumerate(lines):

            eline, component, tied, amp_init, offset_init, stddev_init = line.split()

            # new key for every kin. component
            if component not in components.keys():
                components[component] = [eline]
            else:
                components[component].append(eline)

        return components

    def load_elines(self):
        """Read in the emission line parameters file

        Returns
        -------
        elines: `instance`
                object of type 'Eline' with the initial guess parameters
        """

        elines_file = self.par.elines_par
        with open(elines_file) as f:
            lines = [line for line in f if not (line.startswith('#') or (line.split() == []))]

        elines = EmissionLineSet()

        for idx, line in enumerate(lines):

            eline, component, tied, amp_init, vel_init, disp_init  = line.split()
            emissionline = EmissionLine(name=eline,
                                         component=component,
                                         tied=tied,
                                         idx=idx,
                                         amplitude=float(amp_init),
                                         vel=float(vel_init),
                                         disp=float(disp_init)
                                    )

            elines.add_line(emissionline)

        return elines.elines

    def load_incl_cont(self):
        """Read in the file that contains the AGN continuum regions

        Returns
        -------
        incl_cont: `dictionary`
            values contain start and end wavelength of the regions
        """

        para_file = self.par.incl_cont
        with open(para_file) as f:
            lines = [line for line in f if not (line.startswith('#') or (line.split() == []))]

        # store start/end wavelength in dictionary
        incl_cont = {}
        for idx, line in enumerate(lines):
            wvl_min, wvl_max = line.split()
            incl_cont[idx] = [float(wvl_min), float(wvl_max)]

        return incl_cont

    def load_spectrum(self, file=None):
        """Read in 1D spectrum from fits file

        Parameters
        ----------
        file : `string`
            relative path to incl.cont file

        Returns
        -------
        wvl: `numpy array`
            wavelength array, truncated to range of the AGN fit
        spec: `numpy array`
            flux array
        error: `numpy array`
            flux error array
        """

        with fits.open(file) as hdul:
            wvl = hdul[0].data
            spec = hdul[1].data
            err = hdul[2].data

        select = [(wvl>self.fit_range[0].value) & (wvl<self.fit_range[1].value)]

        wvl = wvl[select]
        spec = spec[select]
        err = err[select]

        return wvl, spec, err

    def couple_amplitudes(self):

        """
            This functions couples the line ratios (i.e. amplitudes of the Gaussians)
            to their theoretical predictions, as specified in the elines.par input file
        """

        def makeFunc_amplitude(model, factor, idx_ref):
            """
                This nested function generates a lambda function that is required by astropy to
                 tie parameters. This is necessary since the argument (i.e. model and factor)
                 needs to be bound for each function created.
            """
            return lambda model: factor * getattr(model, 'amplitude_'+str(idx_ref))

        for idx_eline, eline in enumerate(self.elines.keys()):


            # split name and component to which the emission line belongs
            # find the reference line to which the amplitude is coupled
            name, component = eline.split('_')

            # define line-specific amplitude ratio
            if (name == 'OIII5007') or (name == 'FeII5018'):
                if (name == 'OIII5007'):
                    ref_name = 'OIII4959'
                    factor = 3
                elif (name == 'FeII5018'):
                    ref_name = 'FeII4924'
                    factor = 1.29
                else:
                    ref_name = None
                    factor = None

                # index in compound model of reference line
                idx_ref = self.elines[ref_name+'_'+component].idx

                # print(eline, ref_name, factor, idx_eline, idx_ref) # inspect which lines are coupled to which

                # set compound model value to (some factor) x (reference line value)
                # which is an attribute of the compound model
                getattr(self.compound_model, 'amplitude_'+str(idx_eline)).tied = makeFunc_amplitude(self.compound_model,
                                                                                                    factor, idx_ref)

        return None

    def couple_kinematics(self):
        """
        This functions couples the kinematics, as according to the components specified
        in the elines.par input file
        """


        def makeFunc_vel(model, eline, idx_ref):
            """
            This nested function generates a lambda function that is required by astropy to
            tie parameters. This is necessary since the argument (i.e. model and factor)
            needs to be bound for each function created.
            """
            return lambda model: (getattr(model, 'mean_'+str(idx_ref)) *
                                (self.lambdarest[eline.split('_')[0]] / self.lambdarest[ref.split('_')[0]])
                                )

        def makeFunc_disp(model, idx_eline, idx_ref):
            """ Same as the above function for dispersion.
            """

            return lambda model: (getattr(model, 'stddev_'+str(idx_ref)) *
                                getattr(model, 'mean_' + str(idx_eline)) / (getattr(model, 'mean_' + str(idx_ref)))
                                )


        # get all emission lines that are 'reference lines', i.e. a line to which others are kinematically coupled
        # for the 'reference lines', their name equals the column 'tied' in the elines.par file
        isref = np.array([(eline == self.elines[eline].tied) for eline in self.elines.keys()])

        for component in self.components:

            # get emission lines that belong to component
            iscomponent = np.array([(component in eline) for eline in self.elines.keys()])
            component_lines = np.array(list(self.elines.keys()))[iscomponent]

            # get 'reference line' for this component
            ref = (np.array(list(self.elines.keys()))[iscomponent & isref])[0]

            # index in compound model of reference line
            idx_ref = self.elines[ref].idx

            # tie kinematics to reference line
            for eline in component_lines:
                # do not couple line to itself
                if not (eline == ref):

                    # index in compound model of emission line
                    idx_eline = self.elines[eline].idx

                    # print(eline, idx_eline, ref, idx_ref) # inspect which elines are coupled to which

                    # (1) couple velocity based on rest-frame wavelengths
                    getattr(self.compound_model, 'mean_'+str(idx_eline)).tied = makeFunc_vel(self.compound_model,
                                                                                             eline, idx_ref)

                    # (2) couple dispersion based on rest-frame wavelengths
                    getattr(self.compound_model, 'stddev_'+str(idx_eline)).tied = makeFunc_disp(self.compound_model,
                                                                                                idx_eline, idx_ref)


    def setup_eline_models(self):
        """Set up emission line models

        Returns
        -------
        eline_models: `instance`
            astropy models generated from the inital guess parameters
        """

        # Scale all initial guess amplitudes w.r.t. maximum flux density in AGN spectrum
        a0 = np.nanmax(self.subtract_continuum(self.wvl,self.AGN_spectrum))

        # empty instance
        eline_models = type('', (), {})()

        for eline in self.elines.keys():
            wave_rest = self.lambdarest[eline.split('_')[0]]

            model = models.Gaussian1D(self.elines[eline].amplitude * a0,
                                      wave_rest * (1+ self.elines[eline].vel/self.c),
                                      self.elines[eline].disp / self.c * wave_rest
                                      )

            setattr(eline_models, eline, model)

        return eline_models

    def setup_compound_model(self):
        """
        Generates a model from the individual emission line models.
        The output can be used by a fitter to find the best solution.

        Returns
        -------
        compound_model: `astropy.models Gaussian`
            combined model from the individual astropy models of the emission lines
        """

        # get all eline_models in component
        basemodels = np.full(len(self.elines.keys()), models.Gaussian1D())

        for idx, eline in enumerate(self.elines.keys()):
            basemodels[idx] = getattr(self.eline_models, eline)

        # compound model
        for idx in range(len(basemodels))[1:]: # add all models to first element in array
            basemodels[0] += basemodels[idx]

        compound_model = basemodels[0]

        for i in compound_model:
            i.amplitude.min = 0

        return compound_model

    def subtract_continuum(self, wvl, spectrum):
        """
        Subtract a power-law continuum from a spectrum.
        The regions used to fit the continuum are specified in the incl.cont file

        Parameters
        -------
        wvl: `numpy array`
            wavelength array
        spectrum: `numpy array`
            spectrum from which the continuum shall be subtracted

        Returns
        -------
        wvl: `numpy array`
            wavelength array
        spectrum: `numpy array`
            AGN continuum subtracted spectrum
        """

        select = np.zeros(wvl.shape).astype(bool)
        for i in self.incl_cont:
            select = select + ((wvl> self.incl_cont[i][0]) & (wvl< self.incl_cont[i][1]))

        fit = fitting.LinearLSQFitter() # initialize linear fitter
        line_init = models.Polynomial1D(degree=1)

        cont_model = fit(line_init, wvl[select], spectrum[select])
        cont = cont_model(wvl)

        eline = spectrum - cont

        return eline, cont

    def fit(self):
        """ Executes the workflow for fitting the 1D AGN spectrum.
        """

        # subtract power law continuum
        self.eline, self.cont = self.subtract_continuum(self.wvl, self.AGN_spectrum)

        spectrum = Spectrum1D(flux=self.eline*u.Jy, spectral_axis=self.wvl*u.Angstrom,
                              uncertainty=StdDevUncertainty(self.AGN_error))

        self.bestfit_model = fit_lines(spectrum, self.compound_model, window=self.fit_range, weights='unc')
        self.eline_model = self.bestfit_model(self.wvl*u.Angstrom)


        self.write()

        #siena3d.plot.plot_AGNspec_model(self, savefig=True)
        siena3d.plot.plotly_spectrum(self, savefig=True)

        return None

    def write(self, path='Output/'):
        """
        Writes (1) a table that contains the best-fit parameters of the emission line model
               (1) a table that contains the spectra of the best-fit components
        """

        # (1) par_table
        t = Table([[eline for eline in self.elines.keys()],
                   *(np.array([(self.bestfit_model[i].parameters) for i in range(len(self.elines.keys()))])).T],
                    names=('eline',  'amplitude', 'mean', 'stddev')
                  )

        hdul = fits.BinTableHDU(t)
        hdul.writeto(self.par.output_dir + '/' + self.par.obj + '.par_table.fits', overwrite=True)

        # (2) best_model_components

        t = Table()
        t['wvl'] = self.wvl
        t['data'] = self.AGN_spectrum
        t['error'] = self.AGN_error
        t['eline_model'] = self.eline_model.value
        t['powerlaw'] = self.cont

        # ???

        for idx,eline in enumerate(self.elines.keys()):
            t[eline] = self.bestfit_model[idx](self.wvl*u.Angstrom).value

        t.write(self.par.output_dir + '/' + self.par.obj + '.agnspec_components.fits', overwrite=True)

        return None