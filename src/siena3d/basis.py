"""
This file contains the EmissionLine class
"""
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models

class Component:
        """
        A class which represents the kinematic components.

         Parameters
         ----------
        name : `string`
            emission line name
        """

        def __init__(self, wvl, components, par):
            """
            A class which contains the kinematic basis.
            This includes the parameters in the eline.par file and the
            the best-fit model parameters and its spectrum.

             Parameters
             ----------
            name : `string`
                emission line name
            """

            self.wvl = wvl
            self.components = components
            self.par = par

class Basis:
    def __init__(self, wvl, components, par):
        """
        A class which contains the kinematic basis.
        This includes the parameters in the eline.par file and the
        the best-fit model parameters and its spectrum.

         Parameters
         ----------
        name : `string`
            emission line name
        """

        self.wvl =wvl
        self.components = components
        self.par = par

        self.components = self.load_components()
        # combine astropy models
        self.models = self.setup_basis_models(self.components)
        # initialize arrays containing normalized spectra
        self.arrays = self.setup_basis_arrays()

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

    def setup_basis_models(self, components):
        """
        This function combines models for which the flux ratio and kinematics
        are determined from the best-fit AGN spectrum. Thus, a basis_model contains
        all emission lines and ties the kinematic and flux-ratios amongst them.

        Parameters
        ----------
        components : `list`
            names of the kinematic components which may contain contributions
            from multiple emission lines

        Returns
        -------
        basis_models: class with attributes `astropy.modeling.functional_models.Gaussian1D`
            attributes contain the combined models for the respective kinematic component
        """

        # load best-fit parameters from AGN spectrum
        par_table = self.par.output_dir + '/' + self.par.obj + '.par_table.fits'
        with fits.open(par_table) as hdul:
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

    def setup_basis_arrays(self):
        """
        Evaluates the model for a given wavelength array
        returns normalized spectrum for the base components
        i.e. broad, core_Hb, core_OIII, wing_Hb, wing_OIII

        returns arrays that are normalized to the peak flux
        of the resp. component

        Returns
        -------
        basis: `arrays`
            collection of normalized spectrum of the
            the respective kinematic component
        """

        basis = type('', (), {})()  # empty object to store spectra

        for component in self.components.keys():
            spectrum = getattr(self.models, component)(self.wvl)
            spectrum_norm = spectrum / np.nansum(spectrum)

            setattr(basis, component, spectrum_norm)

        return basis