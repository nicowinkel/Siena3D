"""
This file contains the spectrum class
"""

from .cube import Cube
import siena3d.plot

import numpy as np
import sys
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



class Eline(object):

    """
    A class that represents an emission line as characterized by the parameters in the eline.par file
    """

    def __init__(self, component=None, tied=False, amp_init=0, offset_init=0, stddev_init=0, idx=None):

        self.component = component
        self.tied = tied                                                # bool whether component is kinematically tied
        self.amp_init = amp_init                                        # initial guess for amplitude rel. to maxflux
        self.offset_init = offset_init                                  # initial guess for offset from rest-frame
        self.stddev_init = stddev_init                                  # initial guess for stddev
        self.idx = idx                                                  # index in compound model

class Spectrum(Cube):

    """
    A class contains the spectrum properties and performs a modelling of the AGN spectrum.

    Parameters
    ----------
    Cube: cube object from which attributes can be inherited
    """

    def __init__(self, cube, wvl_start=4750, wvl_end=9300):

        self.components = {'broad': ['Hb_broad', 'Hb_medium',
                                     'FeII4924_medium', 'FeII4924_broad',
                                     'FeII5018_medium', 'FeII5018_broad'],
                           'core': ['Hb_core', 'OIII4959_core', 'OIII5007_core'],
                           # 'wing_Hb':['Hb_wing'],
                           'wing': ['OIII4959_wing', 'OIII5007_wing']
                           }

        # init parameters
        self.fit_range = (wvl_start*u.Angstrom, wvl_end*u.Angstrom)     # wavelength window for fit

        # inherited instances
        self.wvl = getattr(cube, 'wvl')                                 # rest-frame wavelength
        self.AGN_spectrum = getattr(cube, 'AGN_spectrum')               # full AGN spectrum
        self.AGN_error = getattr(cube, 'AGN_error')                     # full AGN error spectrum

        # load data from input files in working directory
        self.lambdarest = self.load_lambdarest()                        # rest-frame wavelengths of emission lines
        self.elines_par = self.load_elines_par()                        # loads eline.par file
        self.incl_cont = self.load_incl_cont()                          # loads eline.par file

        # setup emission line models
        self.eline_models = self.setup_eline_models()                   # generates astropy models
        self.compound_model = self.setup_compound_model()               # combines all models to one
        self.couple_amplitudes()                                        # couples line ratios
        self.couple_kinematics()                                        # kin. coupling as specified in eline.par file

    def load_lambdarest(self):

        """
            Read in the rest-frame wavelengths of the emission lines
            required to model the AGN spectrum. File comes with package.
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

    def load_elines_par(self, path='./'):

        """
            Read in the emission line parameters file

            Parameters
            ----------
            path : `string`
                relative path to elines.par file
        """

        elines_file = path + "elines.par"
        with open(elines_file) as f:
            lines = [line for line in f if not (line.startswith('#') or (line.split() == []))]

        elines = type('', (), {})()
        for idx, line in enumerate(lines):

            eline, component, tied, amp_init, offset_init, stddev_init = line.split()
            setattr(elines, eline,
                    Eline(component, tied, float(amp_init), float(offset_init), float(stddev_init), idx)
                    )

        return elines

    def load_incl_cont(self, path='./'):

        """
            Read in the file that contains the AGN continuum regions

            Parameters
            ----------
            path : `string`
                relative path to incl.cont file
        """

        para_file = path + "incl.cont"
        with open(para_file) as f:
            lines = [line for line in f if not (line.startswith('#') or (line.split() == []))]

        # store start/end wavelength in dictionary
        incl_cont = {}
        for idx, line in enumerate(lines):
            wvl_min, wvl_max = line.split()
            incl_cont[idx] = [float(wvl_min), float(wvl_max)]

        return incl_cont

    def load_spectrum(self, file=None):

        with fits.open(file) as hdul:
            wvl = hdul[0].data
            spec = hdul[1].data
            err = hdul[2].data

        select = [(wvl>self.fit_range[0].value) & (wvl<self.fit_range[1].value)]

        return wvl[select], spec[select], err[select]

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

        for idx_eline, eline in enumerate(self.elines_par.__dict__):

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
                idx_ref = getattr(self.elines_par, ref_name+'_'+component).idx

                # print(eline, ref_name, factor, idx_eline, idx_ref) # inspect which lines are coupled to which

                # set compound model value to (some factor) x (reference line value)
                # which is an attribute of the compound model
                getattr(self.compound_model, 'amplitude_'+str(idx_eline)).tied = makeFunc_amplitude(self.compound_model,
                                                                                                   factor, idx_ref)

        return None

    def couple_kinematics(self):

        """
            This functions couples the kinematics, as specified in the elines.par input file
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

            """
                Same as the above function for dispersion.
            """

            return lambda model: (getattr(model, 'stddev_'+str(idx_ref)) *
                                getattr(model, 'mean_' + str(idx_eline)) / (getattr(model, 'mean_' + str(idx_ref)))
                                )


        # get all emission lines that are 'reference lines', i.e. a line to which others are kinematically coupled
        # for the 'reference lines', their name equals the column 'tied' in the elines.par file
        isref = np.array([(eline == getattr(self.elines_par, eline).tied) for eline in self.elines_par.__dict__])

        for component in self.components:

            # get emission lines that belong to component
            iscomponent = np.array([(component in eline) for eline in self.elines_par.__dict__])
            component_lines = np.array([*vars(self.elines_par).keys()])[iscomponent]

            # get 'reference line' for this component
            ref = (np.array([*vars(self.elines_par).keys()])[iscomponent & isref])[0]

            # index in compound model of reference line
            idx_ref = getattr(self.elines_par, ref).idx

            # tie kinematics to reference line
            for eline in component_lines:
                # do not couple line to itself
                if not (eline == ref):

                    # index in compound model of emission line
                    idx_eline = getattr(self.elines_par, eline).idx

                    # print(eline, idx_eline, ref, idx_ref) # inspect which elines are coupled to which

                    # (1) couple velocity based on rest-frame wavelengths
                    getattr(self.compound_model, 'mean_'+str(idx_eline)).tied = makeFunc_vel(self.compound_model,
                                                                                             eline, idx_ref)

                    #lambda model: \
                    #    (getattr(model, 'mean_'+str(idx_ref)) *
                    #         (self.lambdarest[eline.split('_')[0]] / self.lambdarest[ref.split('_')[0]]))

                    # (2) couple dispersion based on rest-frame wavelengths
                    getattr(self.compound_model, 'stddev_'+str(idx_eline)).tied = makeFunc_disp(self.compound_model,
                                                                                                idx_eline, idx_ref)

                    #lambda model: \
                    #    (getattr(model, 'stddev_'+str(idx_ref)) *
                    #         getattr(model, 'mean_' + str(idx_eline)) / (getattr(model, 'mean_' + str(idx_ref))))

    def setup_eline_models(self):
        """
            scale all initial guess amplitudes w.r.t. maximum flux density in AGN spectrum
        """

        a0 = np.nanmax(self.subtract_continuum(self.wvl,self.AGN_spectrum))

        # empty instance
        eline_models = type('', (), {})()

        for eline in self.elines_par.__dict__:

            model = models.Gaussian1D(getattr(self.elines_par, eline).amp_init * a0,
                                      getattr(self.elines_par, eline).offset_init + self.lambdarest[eline.split('_')[0]],
                                      getattr(self.elines_par, eline).stddev_init
                                      )

            setattr(eline_models, eline, model)

        return eline_models

    def setup_compound_model(self):

        # from all emission lines found in the QSO spectrum
        # this function kinematically ties the components that stem from the same region

        # get all eline_models in component
        basemodels = np.full(len(self.elines_par.__dict__), models.Gaussian1D())
        for idx, eline in enumerate(self.elines_par.__dict__):
            basemodels[idx] = getattr(self.eline_models, eline)

        # compound model
        for idx in range(len(basemodels))[1:]: # add all models to first element in array
            basemodels[0] += basemodels[idx]

        compound_model = basemodels[0]

        for i in compound_model:
            i.amplitude.min = 0

        return compound_model

    def subtract_continuum(self,wvl,spectrum):

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

        # subtract power law continuum
        self.eline, self.cont = self.subtract_continuum(self.wvl, self.AGN_spectrum)

        spectrum = Spectrum1D(flux=self.eline*u.Jy, spectral_axis=self.wvl*u.Angstrom,
                              uncertainty=StdDevUncertainty(self.AGN_error))

        self.bestfit_model = fit_lines(spectrum, self.compound_model, window=self.fit_range, weights='unc')
        self.eline_model = self.bestfit_model(self.wvl*u.Angstrom)


        self.write()
        siena3d.plot.plot_AGNspec_model(self, savefig=True)

        return None

    def write(self, path='Output/'):

        # (1) par_table
        t = Table([[eline for eline in self.elines_par.__dict__],
                   *(np.array([(self.bestfit_model[i].parameters) for i in range(len(self.elines_par.__dict__))])).T],
                    names=('eline',  'amplitude', 'mean', 'stddev')
                  )

        hdul = fits.BinTableHDU(t)
        hdul.writeto(path+'par_table.fits', overwrite=True)

        # (2) best_model_components

        t = Table()
        t['wvl'] = self.wvl
        for idx,eline in enumerate(self.elines_par.__dict__):
            t[eline] = self.bestfit_model[idx](self.wvl*u.Angstrom).value

        t.write(path+'best_model_components.fits', overwrite=True)
