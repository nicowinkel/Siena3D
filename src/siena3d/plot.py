"""
This file contains the plotting functions
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from astropy import units as u
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo

# linestyles

ls = {
    'loosely dotted': (0, (1, 10)),
    'dotted': (0, (1, 1)),
    'densely dotted': (0, (1, 1)),

    'loosely dashed': (0, (5, 10)),
    'dashed': (0, (5, 5)),
    'densely dashed': (0, (5, 1)),

    'loosely dashdotted': (0, (3, 10, 1, 10)),
    'dashdotted': (0, (3, 5, 1, 5)),
    'densely dashdotted': (0, (3, 1, 1, 1)),

    'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
    'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
    'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
}


def set_rc_params():
    """
    Specifies the runtime configuration settings for matplotlib.
    This function also defines a number of line styles.
    """

    mult = 1
    mpl.rcParams.update({'font.size': 15 * mult})
    mpl.rcParams['legend.fontsize'] = 15 * mult
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['xtick.labelsize'] = 15 * mult
    mpl.rcParams['ytick.labelsize'] = 15 * mult
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 1
    mpl.rcParams['ytick.major.width'] = 1
    mpl.rcParams['xtick.minor.size'] = 3
    mpl.rcParams['ytick.minor.size'] = 3
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.minor.width'] = 1
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.bottom'] = True
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.left'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['axes.labelsize'] = 15 * mult
    mpl.rcParams['text.usetex'] = True


def colorbar(cax, mappable, orientation="vertical", ticks=None, label=None, fontsize=14, format=None):
    fig = cax.figure
    cax.tick_params(length=5, width=1, labelsize=.8 * fontsize)

    cb = fig.colorbar(mappable, cax=cax, orientation=orientation, format=format)
    cb.set_label(label, labelpad=5, fontsize=fontsize)

    return cb


def scalebar(ax, astrometry, loc=(0.5, 0.5), c='k', distance=50):
    """
    Plots distance scale bar to given axis. Distance parameters cz, pxsize
    must be specified as attributes of astrometry object.

    Parameters
    ----------
    ax : `matplotlib.pyplot.axes`
        axis to which the scale bar will be added
    astrometry : `Siena3D.Astrometry` [optional]
        object from which the distance parameters will be adopted.
    loc : `tuple` [optional]
        location of the scale bar in fraction of axis coordinates.
    c : `matplotlib colour` [optional]
        scale bar color
    distance: 'float'
        distance in [kpc] that the scale bar resembles
    """

    arcsec_per_kpc = 1 / cosmo.kpc_proper_per_arcmin(astrometry.par.cz / 3e5).value * 60 * 1e3

    # compute width and hight in coordinates of the plot
    xextent = ax.get_xlim()[1] - ax.get_xlim()[0]
    yextent = ax.get_ylim()[1] - ax.get_ylim()[0]
    height = 3e-2 * yextent
    width = distance / 1e3 * arcsec_per_kpc / astrometry.par.sampling / 1e3

    xy = (loc[0] - width / xextent / 2, loc[1] - height / yextent / 2)

    rect = patches.Rectangle(xy, width / xextent, height / yextent, linewidth=1, edgecolor=c,
                             facecolor=c, transform=ax.transAxes)

    ax.add_patch(rect)

    tloc = (loc[0], xy[1] - 3 * (height / yextent))
    ax.text(*tloc, r'$%s\,$pc' % str(distance), c=c, fontsize=15, ha='center', va='center', transform=ax.transAxes)


# Functions for sspectrum class

def plot_AGNspec_model(spectrum, axes=None, savefig=True):
    set_rc_params()

    # get line parameters from compound model
    Hb_broad_fit = spectrum.bestfit_model[0](spectrum.wvl * u.Angstrom)
    FeII4924_broad_fit = spectrum.bestfit_model[1](spectrum.wvl * u.Angstrom)
    FeII5018_broad_fit = spectrum.bestfit_model[2](spectrum.wvl * u.Angstrom)

    Hb_medium_fit = spectrum.bestfit_model[3](spectrum.wvl * u.Angstrom)
    FeII4924_medium_fit = spectrum.bestfit_model[4](spectrum.wvl * u.Angstrom)
    FeII5018_medium_fit = spectrum.bestfit_model[5](spectrum.wvl * u.Angstrom)

    Hb_core_fit = spectrum.bestfit_model[6](spectrum.wvl * u.Angstrom)
    OIII4959_core_fit = spectrum.bestfit_model[7](spectrum.wvl * u.Angstrom)
    OIII5007_core_fit = spectrum.bestfit_model[8](spectrum.wvl * u.Angstrom)

    Hb_wing_fit = spectrum.bestfit_model[9](spectrum.wvl * u.Angstrom)
    OIII4959_wing_fit = spectrum.bestfit_model[10](spectrum.wvl * u.Angstrom)
    OIII5007_wing_fit = spectrum.bestfit_model[11](spectrum.wvl * u.Angstrom)

    # Plot
    fig, axes = plt.subplots(1, 1, figsize=(6, 4), dpi=200)

    plt.step(spectrum.wvl + 1.25 / 2, spectrum.AGN_spectrum, c='k', label='AGN')
    plt.plot(spectrum.wvl, spectrum.eline_model + spectrum.cont * u.Jy, '-', linewidth=1, c='red',
             label=r'Compound Model')

    plt.plot(spectrum.wvl, Hb_core_fit, '-.', linewidth=0.5, c='m', label=r'H$\beta$ core')
    plt.plot(spectrum.wvl, Hb_medium_fit, '--', linewidth=.5, c='m', label=r'H$\beta$ medium')
    plt.plot(spectrum.wvl, Hb_broad_fit, '-', linewidth=.5, c='m', label=r'H$\beta$ broad')
    plt.plot(spectrum.wvl, Hb_wing_fit, '.', markersize=.5, c='m', label=r'H$\beta$ wing')

    plt.plot(spectrum.wvl, OIII4959_core_fit, '-.', linewidth=.5, c='green', label=r'OIII4959 core')
    plt.plot(spectrum.wvl, OIII4959_wing_fit, '.', markersize=.5, c='green', label=r'OIII4959 wing')
    plt.plot(spectrum.wvl, OIII5007_core_fit, '-.', linewidth=.5, c='darkgreen', label=r'OIII5007 core')
    plt.plot(spectrum.wvl, OIII5007_wing_fit, '.', markersize=.5, c='darkgreen', label=r'OIII5007 wing')

    plt.plot(spectrum.wvl, FeII4924_medium_fit, '--', linewidth=.5, c='blue', label=r'FeII4924 medium')
    plt.plot(spectrum.wvl, FeII4924_broad_fit, '-', linewidth=.5, c='blue', label=r'FeII4924 broad')
    plt.plot(spectrum.wvl, FeII5018_medium_fit, '--', linewidth=.5, c='darkblue', label=r'FeII5018 medium')
    plt.plot(spectrum.wvl, FeII5018_broad_fit, '-', linewidth=.5, c='darkblue', label=r'FeII5018 broad')

    plt.legend(fontsize=6)
    # plt.xlim(spectrum.fit_range[0].value, spectrum.fit_range[1].value)
    # Axis labels for Spectrum Plots
    xlabel = r'rest-frame wavelength $\lambda \, [\rm{\AA}]$'
    ylabel = r'$F_\lambda \,\, [10^{-16}\, \rm{ergs}^{-1}\rm{cm}^{-2}\rm{A}^{-1}]$'

    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)

    if savefig:
        fig.savefig(spectrum.par.output_dir + '/' + spectrum.par.obj + '.agnspec_model.png', bbox_inches='tight')

    return None


# Functions for spectroastrometry class

def plotly_spectrum(spectrum, savefig=False):
    """ Generates an interactive html file that show the plot of the best fit model
    """

    # Open the best_fit_components file
    with fits.open(spectrum.par.output_dir + '/' + spectrum.par.obj + '.AGNspec_components.fits') as hdul:
        t = hdul[1].data  # FITS table data is stored on FITS extension 1
    # cols = [i.name for i in t.columns]

    # Create a figure with subplots
    fig = make_subplots(rows=2, cols=1, row_heights=(3, 1), vertical_spacing=0.05)

    # ax1: Data & Model
    tracename = "Data"
    fig.add_trace(go.Scatter(x=t["wvl"], y=t["data"], mode="lines",
                             line_shape='hvh',
                             line=go.scatter.Line(color="black", width=1.5), name=tracename, legendrank=1,
                             showlegend=True), row=1, col=1)
    tracename = "Model"
    fig.add_trace(go.Scatter(x=t["wvl"], y=t["eline_model"] + t["powerlaw"], mode="lines",
                             line=go.scatter.Line(color="red", width=1),
                             name=tracename, legendrank=2, showlegend=True), row=1, col=1)
    tracename = "Power-law"
    fig.add_trace(go.Scatter(x=t["wvl"], y=t["powerlaw"], mode="lines",
                             line=go.scatter.Line(color="red", width=1, dash="dash"), name=tracename,
                             legendrank=3, showlegend=True), row=1, col=1)

    # component colors: pick equidistant colors from colormap (excluding the start/end)
    cmap = mpl.cm.get_cmap('gist_earth')
    n = len(spectrum.components)
    colors = [mpl.colors.to_hex(cmap((idx + .5) / n)) for idx in range(n)]

    # plot individual emission line components
    for idx, comp in enumerate(spectrum.components):
        for eline in spectrum.components[comp]:
            fig.add_trace(go.Scatter(x=t["wvl"], y=t[eline], mode="lines",
                                     line=go.scatter.Line(color=colors[idx], width=1.5), name=eline.split('_')[0],
                                     legendgroup=comp, legendgrouptitle_text=comp, legendrank=11 + idx
                                     ),

                          row=1, col=1)

    fig.add_hline(y=0.0, line=dict(color="gray", width=2), row=1, col=1)

    # ax2: residuals
    bestfit = t["eline_model"] + t['powerlaw']
    residuals = (t['data'] - bestfit) / t['error']

    fig.add_trace(go.Scatter(x=t["wvl"], y=residuals, mode="lines",
                             line=go.scatter.Line(color="#FE00CE", width=1), name="Residuals",
                             showlegend=True, legendrank=100,
                             legendgroup='res', legendgrouptitle_text=' '),
                  row=2, col=1)

    # Figure layout, size, margins
    fig.update_layout(autosize=False,
                      width=1700,
                      height=800,
                      margin=dict(l=100, r=100, b=100, t=100, pad=1),
                      # title='None',
                      font_family="Times New Roman",
                      font_size=16,
                      # font_color="white",
                      # legend_title_text="Components",
                      # legend_bgcolor="black",
                      # paper_bgcolor="black",
                      # plot_bgcolor="black",
                      )

    # Update x-axis properties
    fig.update_xaxes(  # title=r"$\Large\lambda_{\rm{rest}}\;\left[Å\right]$",
        showticklabels=False,
        linewidth=0.5, linecolor="gray", mirror=True,
        gridwidth=1, gridcolor="#222A2A", zerolinewidth=2, zerolinecolor="#222A2A",
        row=1, col=1)
    fig.update_xaxes(title=r"$\Large\lambda_{\rm{rest}}\;\left[Å\right]$", linewidth=0.5, linecolor="gray", mirror=True,
                     gridwidth=1, gridcolor="#222A2A", zerolinewidth=2, zerolinecolor="#222A2A",
                     row=2, col=1)
    # Update y-axis properties
    fig.update_yaxes(title=r"$\Large f_\lambda\;\left[\rm{erg}\;\rm{cm}^{-2}\;\rm{s}^{-1}\;Å^{-1}\right]$",
                     linewidth=0.5, linecolor="gray", mirror=True,
                     gridwidth=1, gridcolor="#222A2A", zerolinewidth=2, zerolinecolor="#222A2A",
                     row=1, col=1)
    fig.update_yaxes(title=r"$\Large\Delta f_\lambda / f_\lambda ^{err}$", linewidth=0.5, linecolor="gray", mirror=True,
                     gridwidth=1, gridcolor="#222A2A", zerolinewidth=2, zerolinecolor="#222A2A",
                     # range=[-8, 8],
                     row=2, col=1)

    fig.update_xaxes(matches='x')
    # fig.update_yaxes(matches='y')

    fig.show()

    # write to figure to output directory
    if savefig:
        fig.write_html(spectrum.par.output_dir + '/' + spectrum.par.obj + '.agnspec_model.html', include_mathjax="cdn")
        #fig.write_image("Output/AGNspec_model.pdf")


class FinalPlot:
    """
    Class that generates plots for the  both spectra before/after fitting together with the
    surface brightness maps of the kinematic components

    Parameters
    ----------
    coor : `tuple`
        (x,y) coordinates from where the spectrum in the cube will be extracted

    plotmaps : `list`
        Kinematic components for which maps will be plotted. Must be a subset of the different
        components in the elines.par file.

    savefig : `boolean`
        writes figure if true.
    """

    def __init__(self):
        set_rc_params()

    def plot_spectrum(self, astrometry, gs=None, coor=(0, 0), savefig=False):

        """
        Plots a spectrum from the minicube

        Parameters
        ----------
        coor : `tuple`
            (x,y) coordinates from where the spectrum in the cube will be extracted
        gs : `GridSpecFromSubplotSpec` [optional]
            optional, existing GridSpec to which the plot will be added
        savefig : `boolean` [optional]
            saves plot as .png file
        """

        i, j = coor[1], coor[0]
        #         *** plotting***

        # component colors: pick equidistant colors from colormap (excluding the start/end)
        cmap = mpl.cm.get_cmap('magma')
        n = len(astrometry.spectrum.components)
        colors = [cmap((idx + .5) / n) for idx in range(n)]

        # Setup GridSpec & plot parameters
        xlabel = r'rest-frame wavelength $\lambda \, [\rm{\AA}]$'
        ylabel = r'$f_\lambda \,\, [10^{-16} \rm{erg/s/cm}^{2}/\rm{A}]$'
        rwindow = 5
        if gs == None:
            fig, axes = plt.subplots(figsize=(8, 8), dpi=100)
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0)
        gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0], hspace=0)
        gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[rwindow, 1], hspace=0)

        # ax0: AGN spectrum
        ax0 = plt.subplot(gs0[0])
        ax0.step(astrometry.wvl + .5 * 1.25, 3 * astrometry.cube.AGN_spectrum / np.nanmax(astrometry.cube.AGN_spectrum),
                 linewidth=1, color='k', label='AGN')
        for idx, comp in enumerate(astrometry.spectrum.components):
            spec_init = getattr(astrometry.basis, comp)
            spec_init_norm = spec_init / np.nanmax(spec_init)
            ax0.plot(astrometry.wvl + .5 * 1.25, spec_init_norm, linestyle='--', linewidth=.8, color=colors[idx],
                     label=comp)
        ax0.legend(fontsize=8)
        ax0.set_xlim(min(astrometry.wvl), max(astrometry.wvl))
        #ax0.set_ylim(np.nanmax(astrometry.cube.AGN_spectrum))
        print(np.nanmax(astrometry.cube.AGN_spectrum))
        ax0.set_yticklabels([])
        ax0.text(0.05, 0.85, 'init', fontsize=12, ha='left', color='white', transform=ax0.transAxes,
                 bbox=dict(facecolor='darkblue', alpha=.8, edgecolor='white', boxstyle='round,pad=.5'))

        # ax1: spaxel spectrum and best-fit components
        # Model spectrum from adding the best-fit components
        _, continuum_fit = astrometry.spectrum.subtract_continuum(astrometry.wvl, astrometry.cube.data[:, i, j])
        bestfit_spectrum = continuum_fit
        for idx, comp in enumerate(astrometry.spectrum.components):
            spec_fit = getattr(astrometry.fluxmap, comp)[i, j] * getattr(astrometry.basis, comp)
            bestfit_spectrum = np.nansum([bestfit_spectrum, spec_fit], axis=0)
        spec = astrometry.cube.data[:, i, j]
        err = astrometry.cube.error[:, i, j]
        res = spec - bestfit_spectrum

        ax1 = plt.subplot(gs1[0])
        ax1.plot(astrometry.wvl, bestfit_spectrum, linewidth=1, c='firebrick', label='model')
        ax1.step(astrometry.wvl + .5 * 1.25, spec, color='k', linewidth=1, label='data')
        for idx, comp in enumerate(astrometry.spectrum.components):
            spec_fit = getattr(astrometry.fluxmap, comp)[coor[0], coor[1]] * getattr(astrometry.basis, comp)
            ax1.fill_between(astrometry.wvl + .5 * 1.25, spec_fit, linewidth=1, color=colors[idx], label=comp)
        ax1.legend(fontsize=8)
        ax1.set_xlim(min(astrometry.wvl), max(astrometry.wvl))
        ax1.set_ylim(1e-4 * np.nanmax(astrometry.cube.AGN_spectrum))
        ax1.text(0.05, 0.85, 'best-fit', fontsize=12, ha='left', color='white', transform=ax1.transAxes,
                 bbox=dict(facecolor='darkblue', alpha=.8, edgecolor='white', boxstyle='round,pad=.5'))

        # Residuals
        ax2 = plt.subplot(gs1[1])
        ax2.step(astrometry.wvl + .5 * 1.25, res / err, color='k', linewidth=1)
        ax2.fill_between(astrometry.wvl + .5 * 1.25, -3, 3, color='firebrick', edgecolor='white', alpha=.2)
        ax2.set_xlim(min(astrometry.wvl), max(astrometry.wvl))
        ax2.set_ylim(-6, 6)

        # Adjust plot parameters
        # ticks
        ax0.tick_params(axis='both', labelbottom=False)
        ax1.tick_params(axis='both', labelbottom=False)

        # labels
        if gs == None:
            ax0.set_ylabel(ylabel)
            ax1.set_ylabel(ylabel)
        else:
            ax1.set_ylabel(ylabel)
            ax1.yaxis.set_label_coords(-.12, .95)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(r'$\frac{\rm residual}{\rm error}$')

        ax0.xaxis.set_minor_locator(AutoMinorLocator())
        ax0.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())

        if savefig:
            plt.savefig('Output/Spectroastrometry_spec.png', bbox_inches='tight', dpi=300)

    def plot_maps(self, astrometry, gs=None, plotmaps=['broad', 'core', 'wing'], savefig=False):
        """
            Plots maps of the kinematic components

            Parameters
            ----------
            gs : `GridSpecFromSubplotSpec` [optional]
                existing GridSpec to which the plot will be added
            savefig : `boolean` [optional]
                saves plot as .png file
        """

        extent = np.array([-astrometry.par.ncrop / 2, astrometry.par.ncrop / 2,
                           -astrometry.par.ncrop / 2, astrometry.par.ncrop / 2]
                          )

        # extent *= astrometry.par.sampling * 1e3  # implement cellsize in cube!

        if gs == None:
            fig = plt.figure(figsize=(9, 2 + 1.5 * len(astrometry.fluxmap.__dict__)), dpi=150)
            gs = gridspec.GridSpec(3, len(plotmaps) + 1, wspace=.07, hspace=.06)

        # Top row: flux maps
        cmap = mpl.cm.get_cmap('gist_earth_r')
        for idx, comp in enumerate(plotmaps):
            fluxmap = getattr(astrometry.fluxmap, comp)
            ax = plt.subplot(gs[0, idx])
            im = ax.imshow(fluxmap / np.nanmax(fluxmap), origin='lower', cmap=cmap,  # extent=extent,
                           norm=LogNorm(vmin=2e-2, vmax=1))
            scalebar(ax, astrometry, c='k', loc=(.5, .22), distance=50)

            # annotations
            ax.annotate(comp, xy=(0.9, 0.85), fontsize=14, ha='right', xycoords='axes fraction')
            ax.annotate('(' + chr(0 * len(plotmaps) + idx + 97) + ')',
                        ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')
            if idx == 0: ax.annotate(r'Data', xy=(0.9, .7), fontsize=14, ha='right', xycoords='axes fraction')

            # axes labels and ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='both', labelbottom=False)
            if idx == 0:
                ax.set_ylabel(r'$\Delta \,  \delta \,[{\rm px}]$', labelpad=-7)
            else:
                ax.tick_params(axis='both', labelleft=False)

            # colorbar
            if idx == len(plotmaps) - 1:
                ax = plt.subplot(gs[0, idx + 1])
                cbarlabel = r'$ \Sigma$'
                colorbar(ax, im, label=cbarlabel)

        # Row 2: Model light distribution
        for idx, comp in enumerate(plotmaps):
            fluxmap = getattr(astrometry.fluxmodel, comp)
            ax = plt.subplot(gs[1, idx])
            im = ax.imshow(fluxmap / np.nanmax(fluxmap), origin='lower', cmap=cmap,  # extent=extent
                           norm=LogNorm(vmin=2e-2, vmax=1))
            scalebar(ax, astrometry, c='k', loc=(.5, .22), distance=50)

            # annotations
            if idx == 0: ax.annotate('Model', ha='right', xy=(0.9, 0.85), fontsize=14, xycoords='axes fraction')
            ax.annotate('(' + chr(1 * len(plotmaps) + idx + 97) + ')',
                        ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')

            # centroid location
            # add the location of the component, which is measured in coordinates of the minicube pixels

            ax.scatter(astrometry.loc.broad[0],  # + extent[0],
                       astrometry.loc.broad[1],  # + extent[2],
                       marker='x', c='firebrick', s=40, label='AGN')
            if idx > 0:
                ax.scatter(getattr(astrometry.loc, comp)[0],  # + extent[0],
                           getattr(astrometry.loc, comp)[1],  # + extent[2],
                           marker='x', c='gold', s=40, label='centroid')

            # add legend
            if idx == len(plotmaps) - 1:
                legend = ax.legend(fontsize=8, bbox_to_anchor=(0.98, 0.98), loc='upper right', framealpha=.5)
                # legend.get_frame()._alpha(.4)

            # axes labels and ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='both', labelbottom=False)
            if idx == 0:
                ax.set_ylabel(r'$\Delta \,  \delta \,[{\rm px}]$', labelpad=-7)
            else:
                ax.tick_params(axis='both', labelleft=False)

            # colorbar
            if idx == len(plotmaps) - 1:
                ax = plt.subplot(gs[1, idx + 1])
                cbarlabel = r'$ \Sigma$'
                colorbar(ax, im, label=cbarlabel)

        # Row 3: Residual maps
        cmap = mpl.cm.get_cmap('seismic')
        for idx, comp in enumerate(plotmaps):

            # combine systematic error from PSF with statistical error from data cube noise
            # compute systematic error from the normalized map, scaled to the component flux map
            residuals = getattr(astrometry.fluxmap, comp) - getattr(astrometry.fluxmodel, comp)
            error = np.sqrt(getattr(astrometry.errmap, comp)**2 +
                            (getattr(astrometry, 'sysmap') * getattr(astrometry.fluxmap, comp))**2
                            )

            ax = plt.subplot(gs[2, idx])
            im = ax.imshow(residuals/error, origin='lower', cmap=cmap, vmin=-1, vmax=1)  # , extent=extent)
            scalebar(ax, astrometry, c='k', loc=(.5, .22), distance=50)

            # annotations
            if idx == 0: ax.annotate('Residual', ha='right', xy=(0.9, 0.85), fontsize=14, xycoords='axes fraction')
            ax.annotate('(' + chr(2 * len(plotmaps) + idx + 97) + ')',
                        ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')

            # labels and ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xlabel(r'$\Delta \,  \alpha \,[{\rm px}]$')
            if idx == 0:
                ax.set_ylabel(r'$\Delta \,  \delta \,[{\rm px}]$', labelpad=-7)
            else:
                ax.tick_params(axis='both', labelleft=False)

            # colorbar
            if idx == len(plotmaps) - 1:
                ax = plt.subplot(gs[2, idx + 1])
                cbarlabel = 'residual/error'
                colorbar(ax, im, label=cbarlabel)

        # *** interactive plot ***
        self.boxes = np.full([astrometry.par.ncrop, astrometry.par.ncrop], mpl.lines.Line2D)
        for i in range(astrometry.par.ncrop):
            for j in range(astrometry.par.ncrop):
                # plot all boxes
                xbox = np.array([i, i, i + 1, i + 1, i]) - .5
                ybox = np.array([j, j + 1, j + 1, j, j]) - .5
                self.boxes[i, j] = self.fig.axes[3].plot(xbox, ybox, color='r', linewidth=2)[0]
                self.boxes[i, j].set_visible(False)

        if savefig:
            plt.savefig(astrometry.par.output_dir + '/' + astrometry.par.obj +
                        '.spectroastrometry_maps.png', bbox_inches='tight')

    def show_box(self, coor):
        """ highlights pixel that is located at input the coor
        """

        for i in range(self.boxes.shape[0]):
            for j in range(self.boxes.shape[0]):
                self.boxes[i, j].set_visible(False)

        box = self.boxes[coor]
        isVisible = box.get_visible()
        box.set_visible(not isVisible)

        self.fig.canvas.draw()

    def show_line(self, coor):

        for i in range(self.allcomp.shape[0]):
            for j in range(self.allcomp.shape[0]):
                self.allcomp[i, j].set_visible(False)

        line = self.allcomp[coor]
        isVisible = line.get_visible()
        line.set_visible(not isVisible)

        self.fig.canvas.draw()

    def on_press(self, event):
        """ Function triggered by mouse clicking
        """

        coor = (round(event.xdata), round(event.ydata))
        self.show_box(coor)
        self.plot_spectrum(self.astrometry, coor=coor, gs=self.inner1, savefig=False)

    def plot_all(self, astrometry, coor=(2, 2), plotmaps=['broad', 'core', 'wing'], savefig=True):

        self.astrometry = astrometry
        self.fig = plt.figure(figsize=(18, 7), dpi=150)
        gs = gridspec.GridSpec(1, 2, wspace=0.2, width_ratios=[5, len(plotmaps) * 2 + .2])
        self.inner1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], height_ratios=[1, 1.2], hspace=0)
        self.inner2 = gridspec.GridSpecFromSubplotSpec(3, len(plotmaps) + 1, subplot_spec=gs[1], wspace=.07,
                                                       hspace=.06, width_ratios=np.append(np.ones(len(plotmaps)), .07)
                                                       )

        # initial plot based on input coordinates
        self.plot_spectrum(astrometry, gs=self.inner1, coor=coor, savefig=False)
        self.plot_maps(astrometry, gs=self.inner2, plotmaps=plotmaps, savefig=False)

        # interactive plot updating
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)

        if savefig:
            plt.savefig(astrometry.par.output_dir + '/' + astrometry.par.obj +
                        '.spectroastrometry.jpg', bbox_inches='tight')

        plt.show()

        return self.fig


def print_result(astrometry):
    """Print the spectroastrometry result
    """

    print('\n')
    for component in astrometry.spectrum.components:
        # [px]
        px, dpx = astrometry.get_offset(component)

        # [arcsec]
        arcsec = px * astrometry.par.sampling
        darcsec = dpx * astrometry.par.sampling

        # [pc]
        d_obj = cosmo.comoving_distance(astrometry.par.cz / 3e5)
        pc = (d_obj * arcsec / 206265).to(u.pc).value
        dpc = (d_obj * darcsec / 206265).to(u.pc).value

        print('%15s  ' % (component) +
              'd = (%.2f\u00B1%.2f) px ' % (px, dpx)
              + '= (%.2f\u00B1%.2f) mas' % (arcsec * 1e3, darcsec * 1e3)
              + '= (%.2f\u00B1%.2f) pc' % (pc, dpc)
              )

    # print flux

    print('\n')
    for component in astrometry.spectrum.components:
        print('%15s  F = (%2.2f \u00B1% 2.2f) x %15s' % (component,
                                                         np.nansum(getattr(astrometry.fluxmap, component)),
                                                         np.nansum(getattr(astrometry.errmap, component)),
                                                         astrometry.cube.getHdrValue(keyword='BUNIT')
                                                         )
              )
    print('\n')

