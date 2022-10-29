"""
This file contains the spectroastrometry class
"""

from .cube import Cube
from .spectrum import Spectrum
from .basis import Basis
from .psf import PSF
import siena3d.parameters
import siena3d.plot

import sys
import os
import numpy as np
from scipy.optimize import nnls
from astropy.io import fits
from astropy.table import Table
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
    parfile : `string`, optional with default: "parameters.par"
        file name of the parameters file
    """

    def __init__(self, cubefile, parfile='parameters.par'):
        """  Setup working data and files
        """
        self.print_logo()
        self.parfile = parfile

    def print_logo(self):
        """ Prints the Siena3D logo
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

    def setup_agn_spectrum(self, cubefile):
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
            file name of the input data cube
        """

        # initialize cube
        self.cube = Cube(cz=self.par.cz)
        self.cube.loadFitsCube(cubefile, cz=self.par.cz, extension_hdr=1, extension_data=1, extension_error=2)

        # get minicube
        self.cube.get_minicube(wvl_min=4750, wvl_max=5100, ncrop=self.par.ncrop, write=True,
                               output=self.par.output_dir + '/' + self.par.obj + '.minicube.fits')

        # get full AGN spectrum and coordinates in original data cube
        self.cube.AGN_loc, self.cube.AGN_spectrum, self.cube.AGN_error = \
            self.cube.get_AGN_spectrum(write=False, path=self.par.output_dir + '/' + self.par.obj+'.')
        # get AGN spectrum
        self.spectrum = Spectrum(self.cube, self.par)

        # get mini-wvl array from truncated cube
        self.wvl = self.cube.wvl

        # fit AGN spectrum
        self.spectrum.fit()

        # load fit result from written file
        #self.par_table = self.load_table(filename=self.par.output_dir + '/' + self.par.obj + '.par_table.fits')

    def load_table(self, filename):
        """
        Reads table that contains parameters of the QSO spectrum model

        Parameters
        ----------
        filename : `string`
            file path of eline table

        Returns
        -------
        table: `astropy.table.Table`
            astropy table with data from eline table file
        """

        hdul = fits.open(filename)
        table = Table(hdul[1].data)

        return table


    def fit_spectrum(self, wvl, spectrum, error):
        """  NNLS fit of an individual spectrum using the basis spectra.

        Parameters
        ----------
        wvl : `numpy.array`
            wavelength array
        spectrum : `numpy.array`
            1D spectrum, must have the same dimension as wvl
        error : `numpy.array`
            1D error array, must have the same dimension as wvl

        Returns
        -------
        popt: `numpy.array`
            Optimal values for the parameters so that the sum
            of the squared residuals of f(xdata, *popt) - ydata is minimized.
        model_spec: `numpy.array`
            best-fitting model spectrum. Has the same dimension as wvl.
        """

        # Subtract continuum from input spectrum
        spec_eline, continuum = self.spectrum.subtract_continuum(wvl, spectrum)

        # Solve argmin_x w * (|| Ax - b ||^2) for x>=0
        # A is the matrix that contains the base spectra
        A = np.zeros([wvl.shape[0], len(self.spectrum.components.keys())])

        for idx, i in enumerate(self.spectrum.components.keys()):
            A[:, idx] = self.basis.arrays[i]

        # must contain only finite values
        A[np.isnan(A)] = 0

        b = spec_eline
        w = 1 / error**2

        wmatrix = np.full((len(self.spectrum.components), w.shape[0]), w).T

        popt, rnorm = nnls(A * wmatrix, b * w)
        model_spec = np.zeros(spec_eline.shape)
        for idx, i in enumerate(self.spectrum.components.keys()):
            model_spec += popt[idx] * self.basis.arrays[i]

        return popt, model_spec

    def fit_cube(self, wvl, data, error):
        """
        Performs NNLS of the basis components to each spectrum of the (cropped) input data cube.

        Parameters
        ----------
        wvl : `numpy.array`
            wavelength
        data : `numpy.array`
            data, must have the same dimension as wvl
        error : `numpy.array`
            error, must have the same dimension as wvl

        Returns
        -------
        flux: `numpy.array`
            Collection of 2D surface brightness maps for the kinematic components.
            Array has the shape [data.shape[1],data.shape[2],#components]
        error: `numpy.array`
            Collection of 2D surface brightness errors for the kinematic components.
            Has the same shape as flux.
        """

        scalefactor_map = np.full([data.shape[1], data.shape[2], len(self.spectrum.components)], np.nan)
        scaleerr_map = np.copy(scalefactor_map)

        for i in tqdm(np.arange(data.shape[1])):
            for j in np.arange(data.shape[2]):

                spec = data[:, i, j]
                err = error[:, i, j]

                # linear regression
                # fit use fluxes as fitparameter rather than amplitudes!
                # using MC error estimation
                scalefactor_mc = np.zeros((self.par.samples_spectro, len(self.spectrum.components)))
                for k in np.arange(self.par.samples_spectro):
                    spec_mc = np.random.normal(spec, err)
                    scalefactork, _ = self.fit_spectrum(wvl, spec_mc, err)
                    scalefactor_mc[k] = scalefactork
                scalefactor = np.nanmedian(scalefactor_mc, axis=0)
                scaleerr = np.std(scalefactor_mc, axis=0)

                # store results in array
                scalefactor_map[i, j] = scalefactor
                scaleerr_map[i, j] = scaleerr

        # convert fit results from 3D array to attributes
        flux = type('', (), {})()
        error = type('', (), {})()
        for idx, component in enumerate(self.spectrum.components.keys()):
            setattr(flux, component, scalefactor_map[:, :, idx])
            setattr(error, component, scaleerr_map[:, :, idx])

        return flux, error

    def get_COMPlocs(self, mc_error=True):
        """
        For each of the kinematic components, this function fits the PSF to the 2D light distribution.
        Computes the position of the centroid relative to the PSF model centroid.

        Parameters
        ----------
        mc_error : `boolean`, optional with default: True
            if errors on the location should be computed using bootstrapping

        Returns
        -------
        fluxmodels: `numpy.array`
             images of the model surface brightness distribution
        locs: 'tuple'
             (x,y) coordinates of the centroid in the minicube frame
        """

        # Initialize two attributes for images and centroid coordinates respectively
        fluxmodels = type('', (), {})()
        locs = type('', (), {})()

        # Fit Moffat PSF to each of the components light profiles
        for component in tqdm(self.spectrum.components):

            image = getattr(self.fluxmap, component)
            error = getattr(self.errmap, component)

            img_model, model = self.psf.fit_PSFloc(image, error)

            if mc_error:
                loc_mc = np.zeros((self.par.samples_eline, 2))
                for i in np.arange(self.par.samples_eline):
                    image_mc = np.random.normal(image, error)
                    image_i, model_i = self.psf.fit_PSFloc(image_mc, error)
                    loc_mc[i] = np.array([model_i.x_0.value, model_i.y_0.value])

                loc = np.nanmedian(loc_mc, axis=0)
                loc_err = np.std(loc_mc, axis=0)
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
        Estimate the systematic error from the residuals of the point-like emission from the BLR

        Returns
        -------
        sysmap: `numpy.array`
            normalized systematic error of the PSF model
        """

        image = getattr(self.fluxmap, 'broad')
        model = getattr(self.fluxmodel, 'broad')
        sysmap = image / model

        return sysmap

    def get_offset(self, component):
        """ Computes the centroids' spatial offset from the broad line emission.

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

    def makedir(self):
        """ Creates output directories
        """
        if not os.path.exists(self.par.output_dir):
            os.makedirs(self.par.output_dir)
        if not os.path.exists(self.par.output_dir + '/' + self.par.obj + '_' + 'maps/'):
            os.makedirs(self.par.output_dir + '/' + self.par.obj + '_' + 'maps/')

    def write(self):
        """ Write flux maps
        """

        for component in self.spectrum.components:

            fluxmap = getattr(self.fluxmap, component)
            errormap = getattr(self.errmap, component)
            modelmap = getattr(self.fluxmodel, component)
            residuals = (fluxmap - modelmap) / getattr(self.errmap, component)

            hdu_primary = fits.PrimaryHDU()
            hdul = fits.HDUList([hdu_primary])
            hdul.append(fits.ImageHDU(fluxmap))
            hdul.append(fits.ImageHDU(errormap))
            hdul.append(fits.ImageHDU(modelmap))
            hdul.append(fits.ImageHDU(residuals))

            for i in self.cube.header:
                hdul[1].header[i] = self.cube.header[i]

            hdul[1].header['extname'] = 'Flux'
            hdul[2].header['extname'] = 'Error'
            hdul[3].header['extname'] = 'Model'
            hdul[4].header['extname'] = 'Residuals'

            hdul.writeto(self.par.output_dir + '/' + self.par.obj + '_' + 'maps/' + component + '.fits', overwrite=True)

    def run(self):

        # read parameter file
        self.par = siena3d.parameters.load_parlist(self.parfile)
        self.makedir()

        # setup kinematic basis
        self.basis = Basis(self.par)

        print(' [1] Get AGN spectrum')
        self.setup_agn_spectrum(self.par.input)

        # setup kinematic basis with best-fit parameters from AGN spectrum
        self.basis.setup_basis_models()
        self.basis.setup_basis_arrays(self.wvl)

        # fit components to cube
        print(' [2] Fit components to cube')
        self.fluxmap, self.errmap = self.fit_cube(self.wvl, self.cube.data, self.cube.error)

        # find PSF model parameters from 'broad' component
        self.psf = PSF(data=self.fluxmap.broad, error=np.ones_like(self.fluxmap.broad),
                       psf_model=self.par.psf_model, ncrop=self.par.ncrop, cz=self.par.cz)

        # find centroid for each of the kinematic components' light distribution
        print(' [3] Find centroids')
        self.fluxmodel, self.loc = self.get_COMPlocs(mc_error=True)

        # estimate the systematic error
        self.sysmap = self.estimate_sys()

        # print and plot results
        siena3d.plot.print_result(self)
        self.write()

        # plot the final result
        final_plot = siena3d.plot.FinalPlot()
        final_plot.plot_all(self, coor=self.par.coor, plotmaps=self.par.plotmaps)
