"""
This file contains the EmissionLine class
"""
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models
from .emission_line import EmissionLine, EmissionLineSet

class Component:
        """
        A class which represents a kinematic component.
        It contains information on its spectrum, its 2D flux map measured in
        the data cube and its spectroastrometrically determined parameters.

         Parameters
         ----------
        name : `string`
            emission line name
        fluxmap: 'numpy.ndarray'
            2d flux distribution measured in the data cube
        errmap: 'numpy.ndarray'
            2d error of the component's flux measured in the data cube
        locs: 'tuple'
             (x,y) coordinates of the centroid in the minicube frame
        """

        def __init__(self, elines, wvl, spectrum=None, error=None, fluxmap=None, errmap=None, fluxmodel=None, centroid=None):

            self.elines = elines
            self.wvl = wvl,
            self.spectrum = spectrum
            self.error = error
            self.fluxmap = fluxmap
            self.errmap = errmap
            self.fluxmodel = fluxmodel
            self.centroid = centroid


class Basis:
    def __init__(self, par):
        """
        A class which contains the kinematic basis.
        This includes the parameters in the eline.par file and the
        the best-fit model parameters and its spectrum.

         Parameters
         ----------
        par : `string`
            'Parameter' object containint the input parameters of the 'parameters.par' file
        """

        self.par = par
        self.components = self.load_components()

    def load_components(self):
        """Read in the emission line parameters file

       Returns
       -------
       components: `dictionary`
           dictionary with the EmissionLineSets which the indiviual emission lines components that belong to
           the kinematic component.
       """

        elines_file = self.par.elines_par
        with open(elines_file) as f:
            lines = [line for line in f if not (line.startswith('#') or (line.split() == []))]

        components = {}

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

            # new attribute for every kin. component
            if component not in components.keys():
                components[component] = EmissionLineSet()

            components[component].add_line(emissionline)

        return components

    def setup_basis_models(self):
        """
        This function combines models for which the flux ratio and kinematics
        are determined from the best-fit AGN spectrum. Thus, a basis_model contains
        all emission lines and ties the kinematic and flux-ratios amongst them.

        Returns
        -------
        models: class with attributes `astropy.modeling.functional_models.Gaussian1D`
            attributes contain the combined models for the respective kinematic component
        """

        par_table = self.par.output_dir + '/' + self.par.obj + '.par_table.fits'
        with fits.open(par_table) as hdul:
            t = Table(hdul[1].data)

        comp_models = {}

        for component in self.components:

            # acquire elines that belong to component from AGN fit output file
            compmodels = np.full(len(self.components.keys()), models.Gaussian1D())
            for idx, eline in enumerate(self.components[component].elines):
                row = self.components[component].elines[eline].idx
                eline = EmissionLine(name=t['name'][row],
                                     component=t['component'][row],
                                     tied=t['tied'][row],
                                     idx=idx,
                                     amplitude= t['amplitude'][row],
                                     vel=t['vel'][row],
                                     disp=t['disp'][row]
                                     )
                compmodels[idx] = eline.model

            # combine the eline models
            for idx in range(len(compmodels))[1:]:
                compmodels[0] += compmodels[idx]

            comp_models[component] = compmodels[0]

        self.models = comp_models

    def setup_basis_arrays(self, wvl):
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

        arrays = {}  # empty object to store spectra

        for component in self.components.keys():
            spectrum = self.models[component](wvl)
            spectrum_norm = spectrum / np.nansum(spectrum)

            arrays[component] =  spectrum_norm

        self.wvl = wvl
        self.arrays = arrays