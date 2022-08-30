"""
This file contains the spectroastrometry class
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.colors import LogNorm
from scipy.optimize import nnls
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models, fitting
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
from specutils import Spectrum1D, SpectralRegion
from maoppy.psfmodel import Psfao, psffit
from maoppy.instrument import muse_nfm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import warnings

class Astrometry(Cube):

    """
    A class which performs a spectroastrometry of the emission lines
    present in the AGN data cube.

    Parameters
    ----------
    cubefile : `string`
        path to the original datacube
    eline_table : `string`
        path to the output file from the AGN fitting
    """


    def __init__(self, cubefile, eline_table, cz):

        self.cz = cz
        self.redshift = self.cz/3e5
        self.elines = [  'Hb_broad','Hb_medium', 'Hb_core', 'Hb_wing',
                         'FeII4924_medium','FeII4924_broad',
                         'FeII5018_medium','FeII5018_broad',
                         'OIII4959_core', 'OIII4959_wing',
                         'OIII5007_core', 'OIII5007_wing',
                      ]

        self.components = {'broad':['Hb_broad', 'Hb_medium',
                                    'FeII4924_medium','FeII4924_broad',
                                    'FeII5018_medium','FeII5018_broad'],
                           'core_Hb':['Hb_core'],
                           'core_OIII':['OIII4959_core', 'OIII5007_core'],
                           'wing_Hb':['Hb_wing'],
                           'wing_OIII':['OIII4959_wing', 'OIII5007_wing']
                  }


        self.cube = Cube()
        self.cube.loadFitsCube(cubefile,cz=self.cz, extension_hdr=1, extension_data=1, extension_error=2)
        self.cube.get_minicube()  # get minicube centered at AGN location
        self.wvl = self.cube.wvl
        self.qso_loc, self.qso_spectrum, self.qso_error = self.get_qso_spectrum(self.cube.data, self.cube.error)
        self.qso_eline, self.continuum = self.subtract_continuum(self.wvl, self.qso_spectrum)

        # read in best fit parameters QSO spectrum model
        self.qsotable = self.read_in_table(eline_table)

        # initialize astropy models from best fit parameters
        self.eline_models = self.setup_eline_models(self.wvl, self.qsotable)

        # combine astropy moodels
        print('Setup basis')
        self.basis_models = self.setup_basis_models(self.eline_models, self.components)

        # initialize arrays containing normalized spectra for components
        self.basis = self.setup_basis_arrays(self.wvl, self.basis_models)

        # fit components to cube
        print('Fit components to cube')
        self.flux, self.dflux = self.fit_cube(self.wvl, self.cube.data, self.cube.error)

        # find PSF model parameters
        print('Find PSF model parameters')
        self.PSF_image, self.PSF_param = tqdm(self.get_PSF_param('Hb'))

        # fit model position
        print('Find centroids')
        self.model = self.findpos()


    def read_in_table(self,file):

        # reads table that contains parameters of the QSO spectrum model

        hdul = fits.open(file)
        table = Table(hdul[1].data)

        return table


    def get_qso_spectrum(self, data, error):

        # returns position and spectrum brightest pixel

        white_image = np.nansum(data, axis=0)
        qso_loc = np.unravel_index(np.nanargmax(white_image), white_image.shape)
        qso_spec = data[:,qso_loc[0], qso_loc[1]]
        qso_err = error[:,qso_loc[0], qso_loc[1]]

        return qso_loc, qso_spec, qso_err


    def subtract_continuum(self,wvl,spectrum):

        # extract and subtract linear continuum in the Hb window
        # by defining two regions in the rest-frame
        # (blue from Hb, red from [OIII])
        # returns the continuum-subtracted emission lines and the continuum

        continuum_rf = {1:[4750,4755], 4:[5080,5090]}

        iscont = np.zeros(wvl.shape).astype(bool)
        for i in continuum_rf:
            iscont = iscont + ((wvl> continuum_rf[i][0]) &
                                (wvl< continuum_rf[i][1])
                               )
        # initialize linear fitter
        fit = fitting.LinearLSQFitter()
        line_init=models.Polynomial1D(degree=1)

        cont_model = fit(line_init, wvl[iscont], spectrum[iscont])
        cont = cont_model(wvl)
        eline = spectrum - cont

        plt.plot(wvl, cont)
        plt.plot(wvl, spectrum)


        return eline, cont

    def setup_eline_models(self, wvl, qsotable):

        # set up astropy models from initial guess
        # parameters in the AGNfit table
        # returns the models as attributes

        i = np.where(qsotable['parameter'] == 'amplitude')[0]
        j = np.where(qsotable['parameter'] == 'mean')[0]
        k = np.where(qsotable['parameter'] == 'stddev')[0]

        eline_models = type('', (), {})() # empty object

        for i in self.elines:
            param = qsotable[i]
            gauss = models.Gaussian1D(param[0],param[1],param[2])
            setattr(eline_models, i, gauss)

        return eline_models


    def setup_basis_models(self, gaussmodels, components):

        # this function combines models for which the flux ratio
        # is known and was already fixed in the AGNfit
        # in this, basis_models contains all
        # kinematically (and flux-) tied base components

        basis_models = type('', (), {})()

        for component in components:

            # get all elines that belong to that component
            basemodels = np.full(len(components[component]), models.Gaussian1D())
            for idx, eline in enumerate(components[component]):
                basemodels[idx] = getattr(self.eline_models, eline)

            # combine the eline models
            for idx in range(len(basemodels))[1:]:
                basemodels[0] += basemodels[idx]

            setattr(basis_models, component, basemodels[0])

        return basis_models


    def setup_basis_arrays(self, wvl, models):

        # evaluates the model for a given wavelength array
        # returns normalized spectrum for the base components
        # i.e. broad, core_Hb, core_OIII, wing_Hb, wing_OIII

        # returns arrays that are normalized to the peak flux
        # of the resp. component

        basis = type('', (), {})() # empty object to store spetra

        for component in tqdm(self.components.keys()):
            spectrum = getattr(self.basis_models, component)(wvl)
            spectrum_norm = spectrum / np.nansum(spectrum)


            setattr(basis, component, spectrum_norm)

        return basis


    def fit_spectrum(self, wvl, spectrum, error):

        # fit an individual spectrum with the basis

        # Subtract continuum
        spec_eline, continuum = self.subtract_continuum(wvl,spectrum)

        A = np.zeros([wvl.shape[0],len(self.components.keys())])

        for idx,i in enumerate(self.components.keys()):
            A[:,idx] = getattr(self.basis, i)


        b = spec_eline
        w = 1/error
        wmatrix = np.full((5,w.shape[0]),w).T

        popt, rnorm= nnls(A*wmatrix, b*w)
        model_spec = np.zeros(spec_eline.shape)
        for idx,i in enumerate(self.components.keys()):

            model_spec += popt[idx]*getattr(self.basis, i)

        return popt, model_spec

    def mock_spec(self, wvl, spectrum, error):

        # generates an artifical spectrum by drawing a random flux from the
        # gaussian probability distribution given by the flux measurement
        # and its error at each wavelength

        new_spectrum = [np.random.normal(spectrum[i],error[i]) for i in np.arange(wvl.shape[0])]

        return np.array(new_spectrum)


    def fit_cube(self, wvl, data, error):

        # performs a linear fitting of the basis components to
        # to each spectrum of the data cube
        # returns array with shape [ata.shape[1],data.shape[2],#components]
        # that contains flux maps of the individual components

        scalefactor_map = np.full([data.shape[1],data.shape[2],5], np.nan)
        dscalefactor_map = np.copy(scalefactor_map)

        for i in tqdm(np.arange(data.shape[1])):
            for j in np.arange(data.shape[2]):

                spec = data[:,i,j]
                err = error[:,i,j]
                error_expanded = np.full((5,err.shape[0]),err).T

                # linear regression
                # fit use fluxes as fitparameter rather than amplitudes!
                scalefactor,model_spectrum = self.fit_spectrum(wvl, spec, err)
                scalefactor_map[i,j] = scalefactor

                # MC error estimation
                scalefactor_mcmc=np.zeros((30,5))
                for k in np.arange(30):
                    spec_mcmc = self.mock_spec(wvl,spec,err)
                    scalefactork,_ =  self.fit_spectrum(wvl, spec_mcmc, err)
                    scalefactor_mcmc[k] = scalefactork

                dscalefactor = np.std(scalefactor_mcmc,axis=0)

                # store results in array
                scalefactor_map[i,j] = scalefactor
                dscalefactor_map[i,j] = dscalefactor

        # convert fit results from 3D array to self attributes

        flux = type('', (), {})()
        dflux = type('', (), {})()
        for idx,component in enumerate(self.components.keys()):
            setattr(flux, component, scalefactor_map[:,:,idx])
            setattr(dflux, component, dscalefactor_map[:,:,idx])

        return flux, dflux


    def get_PSF_param(self, line):

        # fits PSFAO19 model to core band line form data cube
        # returns image of PSF, PSF parameters

        # find wavelength position in cube
        line = 'Hb'
        wvl_rf = {'Ha': 6563.8, 'Hb': 4861.4, '8450A': 8450}
        wavelength = wvl_rf[line]                              # wavelength [m]

        # initialize PSF model
        samp = muse_nfm.samp(wavelength*1e-10)                 # sampling (2.0 for Shannon-Nyquist)

        # fit the image with Psfao
        guess = [ 0.081, 1.07, 11.8, 0.06, 0.99, 0.016, 1.62]
        fixed = [False,False,False,False,False,False,False]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psfao = psffit(self.flux.broad,Psfao,guess,weights=1/(self.dflux.broad),
                           fixed=fixed,npixfit=14,                 # fit keywords
                           system=muse_nfm,samp=samp               # MUSE NFM keywords
                          )

        flux_fit, bck_fit = psfao.flux_bck
        fitao = flux_fit*psfao.psf + bck_fit
        parameters = psfao.x

        return fitao, np.array(parameters)


    def fit_PSFAO19(self,image,error):

        # fits PSFAO19 to an image
        # returns centroid position with errors

        if (np.nansum(image)==0) or (np.sum(error<=0)>0):
            image[image<=0] = 1e-19
            error[error<=0] = 1e19

        # find wavelength position in cube
        line = 'Hb'
        wvl_rf = {'Ha': 6563.8, 'Hb': 4861.4, '8450A': 8450}
        wavelength = wvl_rf[line]*(1+self.redshift) # wavelength [m]

        # initialize PSF model
        samp = muse_nfm.samp(wavelength*1e-10)                  # sampling (2.0 for Shannon-Nyquist)
        fixed = [True,True,True,True,True,True,True]

        out = psffit(image,Psfao,self.PSF_param,weights=1/error,fixed=fixed,
                     npixfit=image.shape[0],system=muse_nfm,samp=samp)

        fitao = out.flux_bck[0]*out.psf + out.flux_bck[1]
        dx=out.dxdy[0]
        dy=out.dxdy[1]

        # MC error
        dxdy_mcmc = np.zeros((20,2))

        for i in np.arange(20):
            image_mcmc = np.random.normal(image,error)
            out_mcmc = psffit(image_mcmc,Psfao,self.PSF_param,weights=1/error,
                              fixed=fixed,npixfit=image.shape[0], system=muse_nfm,samp=samp)
            dxdy_mcmc[i] = out_mcmc.dxdy


        ddx = np.std(dxdy_mcmc,axis=0)[0]
        ddy = np.std(dxdy_mcmc,axis=0)[1]

        position =  np.array([dx,dy,ddx,ddy])

        return fitao, position


    def findpos(self):

        models = type('', (), {})()
        for component in tqdm(self.components):

            image_fit,pos_fit = self.fit_PSFAO19(getattr(self.flux, component),
                                                 getattr(self.dflux, component)
                                                )

            model = type('', (), {})()
            model.image = image_fit
            model.centroid = pos_fit

            setattr(models, component, model)


        return models

    def offset(self,component):

        # this function computes the offset px
        # and
        # from the PSF centroids from the BLR

        px = np.sqrt(  (self.model.broad.centroid[0] - getattr(self.model, component).centroid[0])**2 \
                     + (self.model.broad.centroid[1] - getattr(self.model, component).centroid[1])**2
                     )
        dpx = np.sqrt(  self.model.broad.centroid[2]**2 + getattr(self.model, component).centroid[2]**2 \
                      + self.model.broad.centroid[3]**2 + getattr(self.model, component).centroid[3]**2
                      )

        return px,dpx

    def print_result(self):

        # print offset

        for component in self.components:

            # [px]
            px,dpx = self.offset(component)

            # [arcsec]
            arcsec = px*0.025
            darcsec = dpx*0.025

            # [pc]
            d_obj = cosmo.comoving_distance(self.cz/3e5)
            pc = (d_obj*arcsec/206265).to(u.pc).value
            dpc = (d_obj*darcsec/206265).to(u.pc).value

            print('%15s  ' %(component)+
                  'd = (%.2f\u00B1%.2f) px ' %(px,dpx)
                  +' = (%.2f\u00B1%.2f) mas' %(arcsec*1e3,darcsec*1e3)
                  +' = (%.2f\u00B1%.2f) pc' %(pc,dpc)
                 )

        # print flux

        print('\n')
        for component in self.components:
            print('%15s  F = (%2.2f \u00B1% 2.2f) x %15s' %(component,
                                              np.nansum(getattr(self.flux, component)),
                                              np.nansum(getattr(self.dflux, component)),
                                             '10-16 ergs-1cm-2'
                                             )
                 )

    def write(self, path):

        # write flux maps
        for component in self.components:

            fluxmap = getattr(astrometry.flux, component)
            dfluxmap = getattr(astrometry.dflux, component)
            modelmap = getattr(astrometry.model, component).image
            residuals = fluxmap-modelmap/getattr(astrometry.dflux, component)

            hdu_primary = fits.PrimaryHDU()
            hdul = fits.HDUList([hdu_primary])
            hdul.append(fits.ImageHDU(fluxmap))
            hdul.append(fits.ImageHDU(dfluxmap))
            hdul.append(fits.ImageHDU(modelmap))
            hdul.append(fits.ImageHDU(residuals))


            for i in astrometry.cube.header:
                hdul[1].header[i] = astrometry.cube.header[i]

            hdul[1].header['extname'] = 'Flux'
            hdul[2].header['extname'] = 'Flux_err'
            hdul[3].header['extname'] = 'PSFmodel'
            hdul[4].header['extname'] = 'Residuals'

            hdul.writeto(path+'Mrk1044_'+component+'.fits', overwrite=True)
