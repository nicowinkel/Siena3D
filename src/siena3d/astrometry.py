"""
This file contains the spectroastrometry class
"""

from .cube import Cube
from .spectrum import Spectrum
import siena3d.parameters
import siena3d.plot

import sys
import os
import numpy as np
from scipy.optimize import nnls
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.modeling import models, fitting
from maoppy.psfmodel import Psfao, psffit
from maoppy.instrument import muse_nfm
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    def __init__(self, cubefile, parfile='parameters.par', psf_model='PSFAO19'):

        self.print_logo()

        # setup working data and files
        self.cubefile = cubefile
        self.parfile = parfile


    def print_logo(self):

        """
        Prints the SIENA logo
        """

        pkg = importlib_resources.files("siena3d")
        pkg_logo_file = pkg / "data" / "logo.txt"
        with  pkg_logo_file.open() as f:
            logo = f.readlines()

        terminalsize = os.get_terminal_size()
        logosize = (len(logo[0]), len(logo))  # (x,y) assume rectangular shape

        for i in logo:
            line = i.rsplit('\n')[0]
            string = ("%s".center((terminalsize[0] - logosize[0] // 2) // 2) % line)[:terminalsize[0]]
            print(string)


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
        self.cube = Cube(cz=self.par.cz)
        self.cube.loadFitsCube(cubefile, cz=self.par.cz, extension_hdr=1, extension_data=1, extension_error=2)

        # get minicube
        self.cube.get_minicube(writecube=True, path='Output/')

        # get full AGN spectrum and coordinates in original data cube
        self.cube.AGN_loc, self.cube.AGN_spectrum, self.cube.AGN_error = self.cube.get_AGN_spectrum(writespec=True,
                                                                                                    path='Output/')
        # get AGN spectrum
        self.spectrum = Spectrum(self.cube, wvl_start=self.par.wvl_start, wvl_end=self.par.wvl_end)

        # get mini-wvl array from truncated cube
        self.wvl = self.cube.wvl

        # fit AGN spectrum
        self.spectrum.fit()

        # load fit result from written file
        self.par_table = self.load_table(file='Output/par_table.fits')


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

    def setup_basis_models(self, gaussmodels, components):
        """
        This function combines models for which the flux ratio and kinematics
        were determined from the AGN spectrum.
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

        # load best-fit parameters from AGN spectrum
        with fits.open('Output/par_table.fits') as hdul:
            t = Table(hdul[1].data)
        basis_models = type('', (), {})()

        for component in components:

            # get all elines that belong to that component
            basemodels = np.full(len(components[component]), models.Gaussian1D())
            for idx, eline in enumerate(components[component]):
                row = np.argwhere(t['eline'] == eline)[0]
                model = models.Gaussian1D(t['amplitude'][row], t['mean'][row], t['stddev'][row])
                basemodels[idx] = model

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

        for component in tqdm(self.spectrum.components.keys()):
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

        # Subtract continuum from input spectrum
        spec_eline, continuum = self.spectrum.subtract_continuum(wvl, spectrum)

        # Solve argmin_x w * (|| Ax - b ||^2) for x>=0
        # A is the matrix that contains the base spectra
        A = np.zeros([wvl.shape[0], len(self.spectrum.components.keys())])

        for idx, i in enumerate(self.spectrum.components.keys()):
            A[:, idx] = getattr(self.basis, i)

        # must contain only finite values
        A[np.isnan(A)] = 0

        b = spec_eline
        w = 1 / error

        wmatrix = np.full((len(self.spectrum.components), w.shape[0]), w).T

        popt, rnorm = nnls(A * wmatrix, b * w)
        model_spec = np.zeros(spec_eline.shape)
        for idx, i in enumerate(self.spectrum.components.keys()):
            model_spec += popt[idx] * getattr(self.basis, i)

        return popt, model_spec


    def fit_cube(self, wvl, data, error):
        """
        Performs a linear fitting of the basis components
        to each spectrum of the data cube

        Parameters
        ----------
        wvl : `numpy array`
            wavelength
        data : `numpy array`
            data, must have the same dimension as wvl
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

        scalefactor_map = np.full([data.shape[1], data.shape[2], len(self.spectrum.components)], np.nan)
        dscalefactor_map = np.copy(scalefactor_map)

        for i in tqdm(np.arange(data.shape[1])):
            for j in np.arange(data.shape[2]):

                spec = data[:, i, j]
                err = error[:, i, j]

                # linear regression
                # fit use fluxes as fitparameter rather than amplitudes!
                # using MC error estimation
                scalefactor_mcmc = np.zeros((100, len(self.spectrum.components)))
                for k in np.arange(100):
                    spec_mcmc = np.random.normal(spec, err)
                    scalefactork, _ = self.fit_spectrum(wvl, spec_mcmc, err)
                    scalefactor_mcmc[k] = scalefactork
                scalefactor = np.nanmedian(scalefactor_mcmc, axis=0)
                dscalefactor = np.std(scalefactor_mcmc, axis=0)

                # store results in array
                scalefactor_map[i, j] = scalefactor
                dscalefactor_map[i, j] = dscalefactor
        '''
        for idx, component in enumerate(self.spectrum.components.keys()):
            plt.imshow(scalefactor_map[:, :, idx]/dscalefactor_map[:, :, idx])
            plt.colorbar()
            plt.show()
        '''
        # convert fit results from 3D array to attributes
        flux = type('', (), {})()
        dflux = type('', (), {})()
        for idx, component in enumerate(self.spectrum.components.keys()):
            setattr(flux, component, scalefactor_map[:, :, idx])
            setattr(dflux, component, dscalefactor_map[:, :, idx])

        return flux, dflux

    def get_PSFmodel(self):
        """
        PSF model for the broad line emission that is point-like

        Parameters
        ----------
        model : `string`
            model with which the 2D light distritbution will be fitted.
            Available options: 'Gauss', 'Moffat', 'PSFAO19'

        Returns
        -------
        model: `astropy.model Moffat2D`
             best-fitting model for the 2D surface brightness profile of the input component
        """

        # Image of broad line emission
        image = self.fluxmap.broad
        x, y = np.mgrid[:np.shape(image)[0], :np.shape(image)[1]]

        # initialize PSF model

        if self.par.psf_model != 'PSFAO19':

            # fits 'normal' analytic model to core band line form data cube
            # returns image of PSF, PSF parameters
            if self.par.psf_model == 'Moffat':
                model_init = models.Moffat2D(amplitude=np.nanmax(image),
                                             x_0=image.shape[0],
                                             y_0=image.shape[1],
                                             gamma=1,
                                             alpha=1)
            elif self.par.psf_model == 'Gauss':
                model_init = models.Gaussian2D(amplitude=np.nanmax(image),
                                               x_mean=image.shape[0],
                                               y_mean=image.shape[1],
                                               x_stddev=image.shape[1],
                                               y_stddev=image.shape[1],
                                               theta=0
                                               )
                # rename attribute for consistent nomenclature among models
                model_init.__dict__['x_0'] = model_init.__dict__.pop('x_mean')
                model_init.__dict__['y_0'] = model_init.__dict__.pop('y_mean')

            # Initialize Fitter
            fit = fitting.LevMarLSQFitter()

            # Fit the data using a 2D Moffat Profile
            model = fit(model_init, x, y, image)

            return model

        else:
            # fits PSFAO19 model to core band line form data cube
            # returns attribute with PSF parameters and PSF position

            # find wavelength position in cube
            line = 'Hb'
            wvl_rf = {'Ha': 6563.8, 'Hb': 4861.4, '8450A': 8450}
            wavelength = wvl_rf[line]  # wavelength [m]

            # initialize PSF model
            samp = muse_nfm.samp(wavelength * 1e-10)  # sampling (2.0 for Shannon-Nyquist)

            # fit the image with Psfao
            guess = [0.081, 1.07, 11.8, 0.06, 0.99, 0.016, 1.62]
            fixed = [False, False, False, False, False, False, False]

            # with warnings.catch_warnings():
            #    warnings.simplefilter("ignore")
            psfao = psffit(image, Psfao, guess, weights=None,
                           fixed=fixed, npixfit=None,  # fit keywords
                           system=muse_nfm, samp=samp  # MUSE NFM keywords
                           )

            # flux_fit, bck_fit = psfao.flux_bck
            # fitao = flux_fit * psfao.psf + bck_fit

            model = type('', (), {})()  # contains the position attributes
            model.parameters = psfao.x
            model.x_0 = (psfao.dxdy[0] + self.cube.ncrop / 2) * u.pix
            model.y_0 = (psfao.dxdy[1] + self.cube.ncrop / 2) * u.pix

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

        if self.par.psf_model != 'PSFAO19':

            if self.par.psf_model == 'Moffat':
                model_init = models.Moffat2D(amplitude=np.nanmax(image),
                                             x_0=image.shape[0],
                                             y_0=image.shape[1],
                                             gamma=1,
                                             alpha=1)

                # Tie PSF shape parameters
                def tie_gamma(model):
                    return self.PSFmodel.gamma

                model_init.gamma.tied = tie_gamma

                def tie_alpha(model):
                    return self.PSFmodel.alpha

                model_init.alpha.tied = tie_alpha

            elif self.par.psf_model == 'Gauss':
                model_init = models.Gaussian2D(amplitude=np.nanmax(image),
                                               x_mean=image.shape[0] / 2,
                                               y_mean=image.shape[1] / 2,
                                               x_stddev=image.shape[0] / 2,
                                               y_stddev=image.shape[1] / 2,
                                               theta=0
                                               )
                # rename attribute for consistent nomenclature among models
                model_init.__dict__['x_0'] = model_init.__dict__.pop('x_mean')
                model_init.__dict__['y_0'] = model_init.__dict__.pop('y_mean')

                # Tie PSF shape parameters
                def tie_x_stddev(model):
                    return self.PSFmodel.x_stddev

                model_init.x_stddev.tied = tie_x_stddev

                def tie_y_stddev(model):
                    return self.PSFmodel.y_stddev

                model_init.y_stddev.tied = tie_y_stddev

            # Initialize Fitter
            fit = fitting.LevMarLSQFitter()

            # Fit light profile with model
            model = fit(model_init, x, y, image, weights=1 / error)

            # Model image
            img_model = model(y, x)

        else:

            if (np.nansum(image) == 0) or (np.sum(error <= 0) > 0):
                image[image <= 0] = 1e-19
                error[error <= 0] = 1e19

                # find wavelength position in cube
            line = 'Hb'
            wvl_rf = {'Ha': 6563.8, 'Hb': 4861.4, '8450A': 8450}
            wavelength = wvl_rf[line] * (1 + self.par.cz / 3e5)  # wavelength [m]

            # initialize PSF model
            samp = muse_nfm.samp(wavelength * 1e-10)  # sampling (2.0 for Shannon-Nyquist)
            fixed = [True, True, True, True, True, True, True]

            psfao = psffit(image, Psfao, self.PSFmodel.parameters, weights=1 / error, fixed=fixed,
                           npixfit=image.shape[0], system=muse_nfm, samp=samp)

            fitao = psfao.flux_bck[0] * psfao.psf + psfao.flux_bck[1]
            img_model = fitao

            model = type('', (), {})()  # contains the position attributes
            model.x_0 = (psfao.dxdy[0] + self.cube.ncrop / 2) * u.pix
            model.y_0 = (psfao.dxdy[1] + self.cube.ncrop / 2) * u.pix

        return img_model, model


    def get_COMPlocs(self, mcmc=True, nmcmc=20):
        """
        For each of the kin. component, this function fits the PSF to the 2D light distribution.
        Computes the position of the centroid relative to the PSF model centroid.

        Parameters
        ----------
        mcmc : `boolean`
            if errors on the location should be computed using bootstrapping
        nmcmc : `numpy array`
            number of bootstraps

        Returns
        -------
        fluxmodels: `astropy models`
             images of the model surface brightness distribution
        loxs: 'tuple'
             coordinates of the centroid relative to the PSF location
        """

        # Initialize two attributes for images and centroid coordinates respectively
        fluxmodels = type('', (), {})()
        locs = type('', (), {})()

        # Fit Moffat PSF to each of the components light profiles
        for component in tqdm(self.spectrum.components):

            image = getattr(self.fluxmap, component)
            error = getattr(self.errmap, component)

            img_model, model = self.fit_PSFloc(image, error)

            if mcmc:
                loc_mcmc = np.zeros((nmcmc, 2))
                for i in np.arange(nmcmc):
                    image_mcmc = np.random.normal(image, error)
                    image_i, model_i = self.fit_PSFloc(image_mcmc, error)
                    loc_mcmc[i] = np.array([model_i.x_0.value, model_i.y_0.value])

                loc = np.nanmedian(loc_mcmc, axis=0)
                loc_err = np.std(loc_mcmc, axis=0)

            else:
                loc = (np.nan, np.nan)
                loc_err = (np.nan, np.nan)

            setattr(fluxmodels, component, img_model)
            setattr(locs, component, np.array([loc[0], loc[1], loc_err[0], loc_err[1]]))

        return fluxmodels, locs

    def estimate_sys(self):
        """
        For MUSE NFM-AO the systematic error from the PSF model in the center is larger that
        the statistical error of the noise in the data cube.
        estimate the systematic error from the residuals of the point-like emission from the BLR

        Returns
        -------
        sysmap: `numpy array`
            normalized systematic error of the PSF model
        """

        image = getattr(self.fluxmap, 'broad')
        model = getattr(self.fluxmodel, 'broad')
        sysmap = image / model

        return sysmap

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

    def makedir(self, path='.'):
        """
        Creates output directory
        """
        if not os.path.exists(path + '/Output/'):
            os.makedirs(path + '/Output/')
        if not os.path.exists(path + '/Output/maps'):
            os.makedirs(path + '/Output/maps')


    def write(self, path):
        """
        Write flux maps

        Parameters
        ----------
        path : `string`
            output directory
        """

        for component in self.spectrum.components:

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

        self.makedir('.')

        # retrieve minicube, AGN spectrum and fit
        self.par = siena3d.parameters.load_parList(self.parfile)

        print(' [1] Get AGN spectrum')
        self.setup_AGN_spectrum(self.cubefile)

        # combine astropy models
        print(' [2] Setup basis')
        self.basis_models = self.setup_basis_models(self.spectrum.eline_models, self.spectrum.components)

        # initialize arrays containing normalized spectra
        self.basis = self.setup_basis_arrays(self.wvl, self.basis_models)

        # fit components to cube
        print(' [3] Fit components to cube')
        self.fluxmap, self.errmap = self.fit_cube(self.wvl, self.cube.data, self.cube.error)

        # find PSF model parameters from 'broad' component
        self.PSFmodel = self.get_PSFmodel()

        # find centroid for each of the kinematic components' light distribution
        print(' [4] Find centroids')
        self.fluxmodel, self.loc = self.get_COMPlocs(mcmc=True)

        # estimate the systematic error
        self.sysmap = self.estimate_sys()

        # print and plot results
        siena3d.plot.print_result(self)
        self.write('.')

        # plot the final result
        final_plot = siena3d.plot.Final_Plot()
        final_plot.plot_all(self, coor=self.par.coor, plotmaps=self.par.plotmaps)
