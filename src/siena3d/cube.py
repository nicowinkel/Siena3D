"""
This file contains the cube class
"""

from .data import Data
from .header import Header

import numpy as np
from astropy.io import fits


class Cube(Data):

    """
    A class representing 3D spectra.

    `Cube` is a subclass of Data which allows for handling and organizing a
    three-dimensional spectrum. The class supports reading and writing FITS
    files, resampling and rebinning, velocity shifting and broadening, the
    application of extinction corrections and various advanced fitting
    functions.

    Parameters
    ----------
    data : `np.ndarray`
        The spectra as a 2D np array structured such that the different
        spectra are located along the second and third dimension.
    wvl : `np.ndarray`
        The wavelength elements corresponding to the different data points
        along the third dimension of the `data`.
    error : `np.ndarray`, optional
        The error spectrum, should be of the same shape as `data`.
        If `error` equals None, it is assumed the error spectrum is not
        known.
    mask : `np.ndarray`
        A boolean array where True represents a masked (invalid) data point
        and False a good data point. Should be of the same shape as `data`.
    normalization : `np.ndarray`
        An array which is used to normalize the data/error; both data and
        error are divided by `normalization`. Should be of the same shape
        as `data`.
    inst_fwhm : float
        The instrumental FWHM in the same units as `wavelength`.
    header : Header, optional
        Contains information for reading and writing data to and from Fits
        files.
    """

    def __init__(self, header=None, cz=None, data=None, error=None, mask=None, ncrop=14):

        Header.__init__(self, header=header)
        self.ncrop = ncrop

    def get_minicube(self, wvl_min=4750, wvl_max=5100, ncrop=14, writecube=True, path='Output/'):
        """
        Truncates the initial data cube in both wavelength
        and spatial dimension.
        The resulting minicube has a shape of [something,ncrop,ncrop]
        and has the brightest pixel in its center
        """

        # crop wvl axis in rest frame
        select = ((self.wvl>wvl_min) & (self.wvl<wvl_max))

        self.data = self.data[select]
        self.error = self.error[select]
        self.wvl = self.wvl[select]

        # find flux maximum as qso position
        ycen, xcen = np.unravel_index(np.nanargmax(np.nansum(self.data, axis=0)), self.data.shape[1:])

        # crop spatial axis
        s = ncrop//2
        self.data = self.data[:,(ycen-s):(ycen+s),(xcen-s):(xcen+s)]
        self.error = self.error[:,ycen-s:ycen+s,xcen-s:xcen+s]

        if writecube:
            primary = fits.PrimaryHDU(self.data)
            hdu1 = fits.ImageHDU(self.error)
            hdul = fits.HDUList([primary, hdu1])
            hdul.writeto(path+'minicube.fits', overwrite=True)

        return None

    def loadFitsCube(self, filename, cz=None, extension_hdr=None, extension_data=None,
                     extension_mask=None, extension_error=None,
                     extension_errorweight=None, extensionProjects_hdr=0):
        """
        Load data from a FITS image into a Data object

        Parameters
        --------------
        filename : string
            Name or Path of the FITS image from which the data shall be loaded


        extension_hdr : int or string, optional with default: None
            Number or name of the FITS extension containing the fits header to be used for the cube information like
            wavelength or WCS system.

        extension_data : int or string, optional with default: None
            Number or name of the FITS extension containing the data

        extension_error : int or string, optional with default: None
            Number or string of the FITS extension containing the errors for the values

        extension_mask : int or string, optional with default: None
            Number or name of the FITS extension containing the masked pixels
        """

        hdu = fits.open(filename)
        self.header = hdu[extension_hdr].header
        self.data = hdu[extension_data].data

        if hdu[extension_error].header['EXTNAME'].split()[0] == 'STAT':
                self.error = np.sqrt(hdu[extension_error].data)
        elif hdu[extension_error].header['EXTNAME'].split()[0] == 'ERROR':
                self.error = hdu[extension_error].data
        else:
            print('First extension is neither ERROR nor STAT')

        self.dim = self.data.shape

        try:
            self.wvl = self.header['CDELT3']*np.arange(self.dim[0]) + self.header['CRVAL3']
        except:
            self.wvl = self.header['CD3_3']*np.arange(self.dim[0]) + self.header['CRVAL3']

        else:
            pass

        self.wvl /= (1+cz/3e5)

        hdu.close()

    def get_AGN_spectrum(self, writespec=True, path='Ouput/'):
        """
        Reads table that contains parameters of the QSO spectrum model

        Parameters
        ----------
        writespec : `boolean`
            if TRUE: writes output file
        path : `str`
            path where output file will be written to

        Returns
        -------
        coor: `tuple`
            (x,y) coordinates of AGN in data cube
        spectrum: `numpy array`
            1D spectrum extracted from the AGN spaxel
        error: `numpy array`
            1D error spectrum extracted from the AGN spaxel
        """

        white_image = np.nansum(self.data, axis=0)
        coor = np.unravel_index(np.nanargmax(white_image), white_image.shape)
        spectrum = self.data[:, coor[0], coor[1]]
        error = self.error[:, coor[0], coor[1]]

        if writespec:
            primary = fits.PrimaryHDU(self.wvl)
            hdu0 = fits.ImageHDU(spectrum)
            hdu1 = fits.ImageHDU(error)
            hdul = fits.HDUList([primary, hdu0, hdu1])
            hdul.writeto(path + '/AGNspec.fits', overwrite=True)


        return coor, spectrum, error
