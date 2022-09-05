"""
This file contains the spectrum class
"""

from .cube import Cube
import siena.plot

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



class Eline(Cube):
    def __init__(self, component=None, tied=False, amp_init=0, offset_init=0, stddev_init=0):

        self.component = component
        self.tied = tied
        self.amp_init = amp_init
        self.offset_init = offset_init
        self.stddev_init = stddev_init

class Spectrum(Cube):
    def __init__(self, cube, wvl_start=4750, wvl_end=9300):

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
        self.setup_compound_model()                                     # combines all models to one
        self.couple_amplitudes()                                        # couples line ratios
        self.couple_kinematics()                                        # kin. coupling as specified in eline.par file


    def load_lambdarest(self):

        """
            Read in the rest-frame wavelengths of the emission lines
            required to model the AGN spectrum. File comes with package.
        """

        pkg = importlib_resources.files("siena")
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
        for line in lines:
            eline, component, tied, amp_init, offset_init, stddev_init = line.split()
            setattr(elines, eline,
                    Eline(component, bool(tied), float(amp_init), float(offset_init), float(stddev_init))
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

        self.eline_models.OIII5007_core.amplitude.tied = self.tie_OIII5007_core_amplitude
        self.eline_models.OIII5007_wing.amplitude.tied = self.tie_OIII5007_wing_amplitude
        self.eline_models.FeII5018_medium.amplitude.tied = self.tie_FeII5018_medium_amplitude
        self.eline_models.FeII5018_broad.amplitude.tied = self.tie_FeII5018_broad_amplitude

    #       Line Ratios
    # note that we couple the fluxes (amplitude * stddev), not the amplitudes!
    def tie_OIII5007_core_amplitude(self,model):
        return 3 * model.amplitude_1  # *(model.stddev_1/model.stddev_2)# couple to OIII4959

    def tie_OIII5007_wing_amplitude(self,model):
        return 3 * model.amplitude_10  # *(model.stddev_10/model.stddev_11) # couple to OIII4959_wing

    def tie_FeII5018_medium_amplitude(self,model):
        return 1.29 * model.amplitude_4  # *(model.stddev_4/model.stddev_5) # couple to FeII4924_medium

    def tie_FeII5018_broad_amplitude(self,model):
        return 1.29 * model.amplitude_7  # *(model.stddev_7/model.stddev_8) # couple to FeII4924_broad

        #return None

    def couple_kinematics(self):
        """
            This functions couples the kinematics, as specified in the elines.par input file
        """

        # Velocities
        self.eline_models.OIII4959_core.mean.tied = self.tie_OIII4959_core_pos
        self.eline_models.OIII5007_core.mean.tied = self.tie_OIII5007_core_pos
        self.eline_models.FeII4924_medium.mean.tied = self.tie_FeII4924_medium_pos
        self.eline_models.FeII5018_medium.mean.tied = self.tie_FeII5018_medium_pos
        self.eline_models.FeII4924_broad.mean.tied = self.tie_FeII4924_broad_pos
        self.eline_models.FeII5018_broad.mean.tied = self.tie_FeII5018_broad_pos
        self.eline_models.OIII4959_wing.mean.tied = self.tie_OIII4959_wing_pos
        self.eline_models.OIII5007_wing.mean.tied = self.tie_OIII5007_wing_pos

        # Dispersions
        self.eline_models.OIII4959_core.stddev.tied = self.tie_OIII4959_core_stddev
        self.eline_models.OIII5007_core.stddev.tied = self.tie_OIII5007_core_stddev
        self.eline_models.FeII4924_medium.stddev.tied = self.tie_FeII4924_medium_stddev
        self.eline_models.FeII4924_broad.stddev.tied = self.tie_FeII4924_broad_stddev
        self.eline_models.FeII5018_medium.stddev.tied = self.tie_FeII5018_medium_stddev
        self.eline_models.FeII5018_broad.stddev.tied = self.tie_FeII5018_broad_stddev
        self.eline_models.OIII4959_wing.stddev.tied = self.tie_OIII4959_wing_stddev
        self.eline_models.OIII5007_wing.stddev.tied = self.tie_OIII5007_wing_stddev

        #       Velocities
        # narrow

    def tie_OIII4959_core_pos(self, model):
        return model.mean_0 * (self.lambdarest['OIII4959'] / self.lambdarest['Hb'])  # tie to Hb_core

    def tie_OIII5007_core_pos(self, model):
        return model.mean_0 * (self.lambdarest['OIII5007'] / self.lambdarest['Hb'])  # " Hb_core

    # broad
    def tie_FeII4924_medium_pos(self, model):
        return model.mean_3 * (self.lambdarest['FeII4924'] / self.lambdarest['Hb'])  # couple to Hb_medium

    def tie_FeII5018_medium_pos(self, model):
        return model.mean_3 * (self.lambdarest['FeII5018'] / self.lambdarest['Hb'])  # couple to Hb_medium

    def tie_FeII4924_broad_pos(self, model):
        return model.mean_6 * (self.lambdarest['FeII4924'] / self.lambdarest['Hb'])  # couple to Hb_broad

    def tie_FeII5018_broad_pos(self, model):
        return model.mean_6 * (self.lambdarest['FeII5018'] / self.lambdarest['Hb'])  # couple to Hb_broad

    # outflow
    def tie_OIII4959_wing_pos(self, model):
        return model.mean_9 * (self.lambdarest['OIII4959'] / self.lambdarest['Hb'])  # couple to Hb_wing

    def tie_OIII5007_wing_pos(self, model):
        return model.mean_9 * (self.lambdarest['OIII5007'] / self.lambdarest['Hb'])  # couple to Hb_wing

    #        Dispersions
    # narrow
    def tie_OIII4959_core_stddev(self, model):
        return model.stddev_0 * (model.mean_1 / model.mean_0)  # couple to Hb_core

    def tie_OIII5007_core_stddev(self, model):
        return model.stddev_0 * (model.mean_2 / model.mean_0)  # couple to Hb narrow

    # broad
    def tie_FeII4924_medium_stddev(self, model):
        return model.stddev_3 * (model.mean_4 / model.mean_3)  # couple to Hb_medium

    def tie_FeII5018_medium_stddev(self, model):
        return model.stddev_3 * (model.mean_5 / model.mean_3)  # couple to Hb medium

    def tie_FeII4924_broad_stddev(self, model):
        return model.stddev_6 * (model.mean_7 / model.mean_6)  # couple to Hb_broad

    def tie_FeII5018_broad_stddev(self, model):
        return model.stddev_6 * (model.mean_8 / model.mean_6)  # couple to FeII4924_broad

    # wing
    def tie_OIII4959_wing_stddev(self, model):
        return model.stddev_9 * (model.mean_10 / model.mean_9)  # couple to Hb_wing

    def tie_OIII5007_wing_stddev(self, model):
        return model.stddev_9 * (model.mean_11 / model.mean_9)  # couple to Hb_wing


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

    def setup_eline_models(self):

        # scale all initial guess amplitudes w.r.t. maximum flux density in AGN spectrum
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

        self.compound_model = basemodels[0]

        for i in self.compound_model:
            i.amplitude.min = 0

        return None

    def fit(self):
        
        # subtract power law continuum
        self.eline, self.cont = self.subtract_continuum(self.wvl, self.AGN_spectrum)
        
        spectrum = Spectrum1D(flux=self.eline*u.Jy, spectral_axis=self.wvl*u.Angstrom,
                              uncertainty=StdDevUncertainty(self.AGN_error))

        self.bestfit_model = fit_lines(spectrum, self.compound_model, window=self.fit_range, weights='unc')
        self.eline_model = self.bestfit_model(self.wvl*u.Angstrom)

        self.write()
        siena.plot.plot_AGNspec_model(self, savefig=True)

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
