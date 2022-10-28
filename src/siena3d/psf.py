"""
This file contains the psf class
"""

import numpy as np
from astropy import units as u
from astropy.modeling import models, fitting
from maoppy.psfmodel import Psfao
from maoppy.psffit import psffit
from maoppy.instrument import muse_nfm

class PSF():
    def __init__(self, data, error, psf_model, ncrop, cz):
        """
        A class representing 2D PSF.

        Parameters
        ----------
        data: 'numpy.ndarray'
            input image
        error: 'numpy.ndarray'
            input image error
        psf_model: 'string'
            key for the PSF model that will be adopted
        ncrop: 'integer'
            size of the PSF model
        cz: 'float'
            systemic velocity of the source. This parameter is important if the PSFAO19 model is used.

        """
        self.data = data
        self.error = error
        self.psf_model = psf_model
        self.ncrop = ncrop
        self.cz = cz
        self.PSFmodel = self.get_PSFmodel()

    def get_PSFmodel(self):
        """ PSF model for the broad (point-like) emission.

        Returns
        -------
        model: `maoppy.PSFAO`
             best-fitting model for the 2D surface brightness profile of the input component
        """

        # Image of broad line emission
        image = self.data
        x, y = np.mgrid[:np.shape(image)[0], :np.shape(image)[1]]

        # initialize PSF model

        if self.psf_model != 'PSFAO19':

            # fits 'normal' analytic model to core band line form data cube
            # returns image of PSF, PSF parameters
            if self.psf_model == 'Moffat':
                model_init = models.Moffat2D(amplitude=np.nanmax(image),
                                             x_0=image.shape[0] / 2,
                                             y_0=image.shape[1] / 2,
                                             gamma=1,
                                             alpha=1)
            elif self.psf_model == 'Gauss':
                model_init = models.Gaussian2D(amplitude=np.nanmax(image),
                                               x_mean=image.shape[0] / 2,
                                               y_mean=image.shape[1] / 2,
                                               x_stddev=image.shape[0] / 3,
                                               y_stddev=image.shape[1] / 3,
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
            Pmodel = Psfao(image.shape, system=muse_nfm, samp=samp)
            psfao = psffit(image, Pmodel, guess, weights=None,
                           fixed=fixed, npixfit=self.ncrop  # ,  # fit keywords
                           # system=muse_nfm, samp=samp  # MUSE NFM keywords
                           )

            # flux_fit, bck_fit = psfao.flux_bck
            # fitao = flux_fit * psfao.psf + bck_fit

            model = type('', (), {})()  # contains the position attributes
            model.parameters = psfao.x
            model.x_0 = (psfao.dxdy[0] + self.ncrop / 2) * u.pix
            model.y_0 = (psfao.dxdy[1] + self.ncrop / 2) * u.pix

            return model

    def fit_PSFloc(self, image, error):
        """ Fit PSF model to light distribution where the only free parameters are (x, y, amplitude)

        Parameters
        ----------
        image : `numpy.array`
            2D light distribution
        error : `numpy.array`
            2D light distribution error, must be of the same shape as image

        Returns
        -------
        model: `maoppy.psfmodel.Psfao` or `astropy.modeling.functional_models.Moffat2D` or `astropy.model.Gaussian2D`
             best-fitting model for the 2D surface brightness profile of the input component
        """

        # setup coordinates
        x, y = np.mgrid[:np.shape(image)[0], :np.shape(image)[1]]

        if self.psf_model != 'PSFAO19':

            if self.psf_model == 'Moffat':
                model_init = models.Moffat2D(amplitude=np.nanmax(image),
                                             x_0=image.shape[0] / 2,
                                             y_0=image.shape[1] / 2,
                                             gamma=1,
                                             alpha=1)

                # Tie PSF shape parameters
                def tie_gamma(model):
                    return self.PSFmodel.gamma

                model_init.gamma.tied = tie_gamma

                def tie_alpha(model):
                    return self.PSFmodel.alpha

                model_init.alpha.tied = tie_alpha

            elif self.psf_model == 'Gauss':
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
            wavelength = wvl_rf[line] * (1 + self.cz / 3e5)  # wavelength [m]

            # initialize PSF model
            samp = muse_nfm.samp(wavelength * 1e-10)  # sampling (2.0 for Shannon-Nyquist)
            fixed = [True, True, True, True, True, True, True]

            Pmodel = Psfao(image.shape, system=muse_nfm, samp=samp)
            psfao = psffit(image, Pmodel, self.PSFmodel.parameters, weights=1 / error, fixed=fixed,
                           npixfit=image.shape[0])

            fitao = psfao.flux_bck[0] * psfao.psf + psfao.flux_bck[1]
            img_model = fitao

            model = type('', (), {})()  # contains the position attributes
            model.x_0 = (psfao.dxdy[0] + self.ncrop / 2) * u.pix
            model.y_0 = (psfao.dxdy[1] + self.ncrop / 2) * u.pix

        return img_model, model