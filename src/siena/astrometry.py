"""
This file contains the spectroastrometry class
"""

from .cube import Cube
from .spectrum import Spectrum
import siena.plot
import pkg_resources

import sys
import os
import numpy as np
from scipy.optimize import nnls
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models, fitting
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
from tqdm import tqdm

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    import importlib.resources as importlib_resources



class Astrometry(Cube):

    """
    A class that performs a spectroastrometric analysis of the emission lines
    present in the AGN spectrum across an excerpt of the data cube.

    Parameters
    ----------
    cubefile : `string`
        path to the original datacube
    eline_table : `string`
        path to the output file from the AGN fitting
    """

    def __init__(self, cubefile):

        # load cube
        self.cubefile = cubefile

        # load object-specific parameters
        self.load_parameters_par()

        # setup components to which emission lines belong

        self.components = {'broad': ['Hb_broad', 'Hb_medium',
                                     'FeII4924_medium', 'FeII4924_broad',
                                     'FeII5018_medium', 'FeII5018_broad'],
                           'core_Hb': ['Hb_core'],
                           'core_OIII': ['OIII4959_core', 'OIII5007_core'],
                           # 'wing_Hb':['Hb_wing'],
                           'wing_OIII': ['OIII4959_wing', 'OIII5007_wing']
                           }

        # setup working data
        self.print_logo()


    def print_logo(self):

        """
            Prints the SIENA logo
        """

        pkg = importlib_resources.files("siena")
        pkg_logo_file = pkg / "data" / "logo.txt"
        with  pkg_logo_file.open() as f:
            logo = f.readlines()

        terminalsize = os.get_terminal_size()
        logosize = (len(logo[0]), len(logo))  # (x,y) assume rectengular shape

        for i in logo:
            line = i.rsplit('\n')[0]
            string = ("%s".center((terminalsize[0] - logosize[0] // 2) // 2) % line)[:terminalsize[0]]
            print(string)

    def load_parameters_par(self, path='./'):

        """
            Read in the parameters file

            Parameters
            ----------
            path : `string`
                relative path to parameters.par file
        """

        parameters_file = path + "parameters.par"
        with open(parameters_file) as f:
            lines = [line for line in f if not (line.startswith('#') or (line.split() == []))]

        # store start/end wavelength in dictionary
        for line in lines:
            parameter, value = line.split()[:2]
            setattr(self, parameter, float(value))

        return None

    def setup_AGN_spectrum(self, cubefile):

        """
            This function reads the original data cube and
                (1) extracts a minicube, i.e. a cube that is both truncated in
                    wavelength and spatial dimensions on which the spectroastrometry
                    will be performed
                (2) extracts the highest S/N spectrum which will be adopted as the
                    AGN spectrum
                (3) fit the spectrum with the multi-Gaussian model and writes a table
                    that contains all emission lines

            Parameters
            ----------
            cubefile : `string`
                file path of the original data cube

        """

        # initialize cube
        self.cube = Cube(cz=self.cz)
        self.cube.loadFitsCube(cubefile, cz=self.cz, extension_hdr=1, extension_data=1, extension_error=2)

        # get minicube
        self.cube.get_minicube(writecube=True, path='Output/')

        # get full AGN spectrum and coordinates in original data cube
        self.cube.AGN_loc, self.cube.AGN_spectrum, self.cube.AGN_error = self.cube.get_AGN_spectrum(writespec=True, path='Output/')

        # get AGN spectrum
        self.spectrum = Spectrum(self.cube, wvl_start=self.wvl_start, wvl_end=self.wvl_end)

        # get mini-wvl array from truncated cube
        self.wvl = self.cube.wvl

        # fit AGN spectrum
        self.spectrum.fit()

        # load fit result from written file
        self.par_table = self.load_table(file='Output/par_table.fits')

        return None

    def load_table(self, file):

        """
            Reads table that contains parameters of the QSO spectrum model

            Parameters
            ----------
            file : `string`
                file path of eline table

            Returns
            -------
            table: `astropy.table`
                astropy table with data from eline table file
        """

        hdul = fits.open(file)
        table = Table(hdul[1].data)

        return table

    '''
    def setup_eline_models(self, wvl, par_table):

        """
            Set up astropy models from initial guess parameters
            in the AGNfit table

            Parameters
            ----------
            wvl : `numpy array`
                wavelength array
            par_table : `astropy table`
                contains emission line parameters

            Returns
            -------
            eline_models: `dictionary`
                contains the Gaussian 1D models generated from the input eline table paramters

        """

        eline_models = type('', (), {})()  # empty object

        for eline in self.elines:

            # find row in which eline parameters are listed
            idx = np.argwhere(par_table['eline'] == eline)
            param = [par_table['amplitude'][idx].value[0],
                     par_table['mean'][idx].value[0],
                     par_table['stddev'][idx].value[0]]

            # adopt model if parameters are finite
            if ~(np.any(np.isnan(param)) or np.any((param == 0))):
                eline_model = models.Gaussian1D(*param)
                setattr(eline_models, eline, eline_model)

        return eline_models
    '''
    def setup_basis_models(self, gaussmodels, components):

        """
            This function combines models for which the flux ratio
            is known and was already fixed in the AGNfit.
            In this, basis_models contains all
            kinematically (and flux-) tied base components

            Parameters
            ----------
            gaussmodels : `astropy models`
                emission line models
            components : `strings`
                names of the kinematic components which may contain several different emission
                line components

            Returns
            -------
            basis_models: `astropy models`
                collection of kinematic components
        """

        basis_models = type('', (), {})()

        for component in components:

            # get all elines that belong to that component
            basemodels = np.full(len(components[component]), models.Gaussian1D())
            for idx, eline in enumerate(components[component]):
                basemodels[idx] = getattr(self.spectrum.eline_models, eline)

            # combine the eline models
            for idx in range(len(basemodels))[1:]:
                basemodels[0] += basemodels[idx]

            setattr(basis_models, component, basemodels[0])

        return basis_models

    def setup_basis_arrays(self, wvl, models):

        """
            Evaluates the model for a given wavelength array
            returns normalized spectrum for the base components
            i.e. broad, core_Hb, core_OIII, wing_Hb, wing_OIII

            returns arrays that are normalized to the peak flux
            of the resp. component

            Parameters
            ----------
            wvl : `numpy array`
                wavelength array
            models : `astropy models`
                collection of kinematic components

            Returns
            -------
            basis: `arrays`
                collection of normalized spectrum of the
                the respective kinematic component
        """

        basis = type('', (), {})()  # empty object to store spetra

        for component in tqdm(self.components.keys()):
            spectrum = getattr(self.basis_models, component)(wvl)
            spectrum_norm = spectrum / np.nansum(spectrum)

            setattr(basis, component, spectrum_norm)

        return basis

    def fit_spectrum(self, wvl, spectrum, error):

        """
            fit an individual spectrum with the basis

            Parameters
            ----------
            wvl : `numpy array`
                wavelength
            spectrum : `numpy array`
                spectrum, must have the same dimension as wvl
            error : `numpy array`
                error, must have the same dimension as wvl

            Returns
            -------
            popt: `array`
                Optimal values for the parameters so that the sum
                of the squared residuals of f(xdata, *popt) - ydata is minimized.
            model_spec: `array`
                best-fitting model spectrum. Has the same dimension as wvl
        """

        # Subtract continuum
        spec_eline, continuum = self.spectrum.subtract_continuum(wvl, spectrum)

        A = np.zeros([wvl.shape[0], len(self.components.keys())])

        for idx, i in enumerate(self.components.keys()):
            A[:, idx] = getattr(self.basis, i)

        b = spec_eline
        w = 1 / error

        wmatrix = np.full((len(self.components), w.shape[0]), w).T

        popt, rnorm = nnls(A * wmatrix, b * w)
        model_spec = np.zeros(spec_eline.shape)
        for idx, i in enumerate(self.components.keys()):
            model_spec += popt[idx] * getattr(self.basis, i)

        return popt, model_spec

    def mock_spec(self, wvl, spectrum, error):

        """
            Generates an artifical spectrum by drawing a random flux from the
            Gaussian probability distribution given by the flux measurement
            and its error at each wavelength

            Parameters
            ----------
            wvl : `numpy array`
                wavelength
            spectrum : `numpy array`
                spectrum, must have the same dimension as wvl
            error : `numpy array`
                error, must have the same dimension as wvl

            Returns
            -------
            new_spectrum: `array`
                mock spectrum, has the same dimension as wvl
        """

        new_spectrum = np.array([np.random.normal(spectrum[i], error[i]) for i in np.arange(wvl.shape[0])])

        return new_spectrum

    def fit_cube(self, wvl, data, error):

        """
            Performs a linear fitting of the basis components
            to each spectrum of the data cube

            Parameters
            ----------
            wvl : `numpy array`
                wavelength
            spectrum : `numpy array`
                spectrum, must have the same dimension as wvl
            error : `numpy array`
                error, must have the same dimension as wvl

            Returns
            -------
            flux: `numpy array`
                Collection of 2D surface brightness maps for the kinematic components.
                Array has the shape [data.shape[1],data.shape[2],#components]
            dflux: `numpy array`
                Collection of 2D surface brightness errors for the kinematic components.
                Has the same shape as flux.
        """

        scalefactor_map = np.full([data.shape[1], data.shape[2], len(self.components)], np.nan)
        dscalefactor_map = np.copy(scalefactor_map)

        for i in tqdm(np.arange(data.shape[1])):
            for j in np.arange(data.shape[2]):

                spec = data[:, i, j]
                err = error[:, i, j]
                error_expanded = np.full((len(self.components), err.shape[0]), err).T

                # linear regression
                # fit use fluxes as fitparameter rather than amplitudes!
                scalefactor, model_spectrum = self.fit_spectrum(wvl, spec, err)
                scalefactor_map[i, j] = scalefactor

                # MC error estimation
                scalefactor_mcmc = np.zeros((30, len(self.components)))
                for k in np.arange(30):
                    spec_mcmc = self.mock_spec(wvl, spec, err)
                    scalefactork, _ = self.fit_spectrum(wvl, spec_mcmc, err)
                    scalefactor_mcmc[k] = scalefactork

                dscalefactor = np.std(scalefactor_mcmc, axis=0)

                # store results in array
                scalefactor_map[i, j] = scalefactor
                dscalefactor_map[i, j] = dscalefactor

        # convert fit results from 3D array to self attributes

        flux = type('', (), {})()
        dflux = type('', (), {})()
        for idx, component in enumerate(self.components.keys()):
            setattr(flux, component, scalefactor_map[:, :, idx])
            setattr(dflux, component, dscalefactor_map[:, :, idx])

        return flux, dflux

    def get_PSFmodel(self):

        """
            PSF model for the broad line emission that is point-like

            Parameters
            ----------
            line : `string`
                component to whose surface brightness profile which the Moffat model will be fitted

            Returns
            -------
            model: `astropy.model Moffat2D`
                 best-fitting model for the 2D surface brightness profile of the input component
        """

        # Image of broad line emission
        image = self.fluxmap.broad
        x, y = np.mgrid[:np.shape(image)[0], :np.shape(image)[1]]

        # initialize PSF model
        model_init = models.Moffat2D(amplitude=np.nanmax(image),
                                     x_0=image.shape[0],
                                     y_0=image.shape[1],
                                     gamma=1,
                                     alpha=1)
        # Initialize Fitter
        fit = fitting.LevMarLSQFitter()

        # Fit the data using a 2D Moffat Profile
        model = fit(model_init,x ,y, image)

        return model

    def fit_PSFloc(self, image, error):

        """
            Fit PSF model to light distribution where the only free parameters are
            (x,y,flux)

            Parameters
            ----------
            image : `numpy array`
                2D light distribution
            error : `numpy array`
                2D light distribution error, must be of the same shape as image

            Returns
            -------
            model: `astropy.model Moffat2D`
                 best-fitting model for the 2D surface brightness profile of the input component
        """

        # setup coordinates
        x, y = np.mgrid[:np.shape(image)[0], :np.shape(image)[1]]

        # Initialize PSF model
        model_init = models.Moffat2D(amplitude=np.nanmax(image),
                                     x_0=image.shape[0]/2,
                                     y_0=image.shape[1]/2,
                                     gamma=1,
                                     alpha=1)

        # Tie Moffat shape parameters
        def tie_gamma(model): return self.PSFmodel.gamma
        model_init.gamma.tied = tie_gamma
        def tie_alpha(model): return self.PSFmodel.alpha
        model_init.alpha.tied = tie_alpha

        # Initialize Fitter
        fit = fitting.LevMarLSQFitter()

        # Fit the data using a 2D Moffat Profile
        model = fit(model_init, x, y, image, weights=1/error)

        # Model image
        img_model = model(y,x)

        return img_model, model

    def get_loc(self, bootstrapping=True, nmcmc=20):

        """
            Fits the PSF to the 2D light distribution for each of the kinematic components.

            Returns
            -------
            models: `astropy models`
                contains    (1) the image of the model surface brightness distribution and
                            (2) a tuple with the coordinates of the centroid
        """

        # Initialize two attributes for images and centroid coordinates respectively
        fluxmodels = type('', (), {})()
        locs = type('', (), {})()

        # Fit Moffat PSF to each of the components light profiles
        for component in tqdm(self.components):

            image = getattr(self.fluxmap, component)
            error = getattr(self.errmap, component)

            img_model, model = self.fit_PSFloc(image,error)

            if bootstrapping:
                loc_mcmc = np.zeros((nmcmc, 2))
                for i in np.arange(nmcmc):
                    image_mcmc = np.random.normal(image, error)
                    _, model_i = self.fit_PSFloc(image_mcmc,error)
                    loc_mcmc[i] = np.array([model_i.x_0.value, model_i.y_0.value])
                loc_err = np.std(loc_mcmc, axis=0)

            else:
                loc_err = (np.nan, np.nan)

            setattr(fluxmodels, component, img_model)
            setattr(locs, component, np.array([model.x_0.value, model.y_0.value,
                                               loc_err[0], loc_err[1]
                                               ]
                                              )
                    )

        return fluxmodels, locs

    def get_offset(self, component):

        """
            This function computes the offset px
            and from the PSF centroids from the BLR

            Parameters
            ----------
            component : `string`
                component for which the offset from the AGN position is computed
        """

        px = np.sqrt((self.loc.broad[0] - getattr(self.loc, component)[0]) ** 2 \
                     + (self.loc.broad[1] - getattr(self.loc, component)[1]) ** 2
                     )
        dpx = np.sqrt(self.loc.broad[2] ** 2 + getattr(self.loc, component)[2] ** 2 \
                      + self.loc.broad[3] ** 2 + getattr(self.loc, component)[3] ** 2
                      )

        return px, dpx

    def print_result(self):

        """
            Print the spectroastrometry result
        """
        print('\n')
        for component in self.components:
            # [px]
            px, dpx = self.get_offset(component)

            # [arcsec]
            arcsec = px * 0.025
            darcsec = dpx * 0.025

            # [pc]
            d_obj = cosmo.comoving_distance(self.cz / 3e5)
            pc = (d_obj * arcsec / 206265).to(u.pc).value
            dpc = (d_obj * darcsec / 206265).to(u.pc).value

            print('%15s  ' % (component) +
                  'd = (%.2f\u00B1%.2f) px ' % (px, dpx)
                  + '= (%.2f\u00B1%.2f) mas' % (arcsec * 1e3, darcsec * 1e3)
                  + '= (%.2f\u00B1%.2f) pc' % (pc, dpc)
                  )

        # print flux

        print('\n')
        for component in self.components:
            print('%15s  F = (%2.2f \u00B1% 2.2f) x %15s' % (component,
                                                             np.nansum(getattr(self.fluxmap, component)),
                                                             np.nansum(getattr(self.errmap, component)),
                                                             '1e-16 ergs-1cm-2'
                                                             )
                  )
        print('\n')

    def makedir(self,path='.'):
        """
           Creates output directory
        """
        if not os.path.exists(path+'/Output/'):
            os.makedirs(path+'/Output/')
        if not os.path.exists(path+'/Output/maps'):
            os.makedirs(path+'/Output/maps')

        return None

    def write(self, path):

        """
            Write flux maps

            Parameters
            ----------
            path : `string`
                output directory
        """

        self.makedir(path)

        for component in self.components:

            fluxmap = getattr(self.fluxmap, component)
            dfluxmap = getattr(self.errmap, component)
            modelmap = getattr(self.fluxmodel, component)
            residuals = fluxmap - modelmap / getattr(self.errmap, component)

            hdu_primary = fits.PrimaryHDU()
            hdul = fits.HDUList([hdu_primary])
            hdul.append(fits.ImageHDU(fluxmap))
            hdul.append(fits.ImageHDU(dfluxmap))
            hdul.append(fits.ImageHDU(modelmap))
            hdul.append(fits.ImageHDU(residuals))

            for i in self.cube.header:
                hdul[1].header[i] = self.cube.header[i]

            hdul[1].header['extname'] = 'Flux'
            hdul[2].header['extname'] = 'Flux_err'
            hdul[3].header['extname'] = 'Model'
            hdul[4].header['extname'] = 'Residuals'

            hdul.writeto(path + '/Output/maps/' + component + '.fits', overwrite=True)



    def run(self):

        # retrieve minicube, AGN spectrum and fit
        print(' [1] Get AGN spectrum')
        self.setup_AGN_spectrum(self.cubefile)

        # initialize astropy models from best fit parameters
        #self.eline_models = self.setup_eline_models(self.wvl, self.par_table)
        #self.setup_compound_model(self.elines_par, self.eline_models)

        # combine astropy models
        #print(' [2] Setup basis')
        self.basis_models = self.setup_basis_models(self.spectrum.eline_models, self.components)

        # initialize arrays containing normalized spectra
        self.basis = self.setup_basis_arrays(self.wvl, self.basis_models)

        # fit components to cube
        print(' [3] Fit components to cube')
        self.fluxmap, self.errmap = self.fit_cube(self.wvl, self.cube.data, self.cube.error)

        # find PSF model parameters from 'broad' component
        print(' [4] Find PSF model parameters')
        self.PSFmodel = self.get_PSFmodel()

        # find centroid for each of the kinematic components' light distribution
        print(' [5] Find centroids')
        self.fluxmodel, self.loc = self.get_loc(bootstrapping=True)

        # print and plot results
        print(' [6] Print & plot result')
        siena.plot.print_result(self)
        self.write('.')

        # plot the result
        siena.plot.plot_all(self, coor=[0,0])
