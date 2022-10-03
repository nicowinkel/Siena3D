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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


def scalebar(ax, cz, loc=(0.5, 0.5), c='k', distance='50pc'):
    xextent = ax.get_xlim()[1] - ax.get_xlim()[0]
    yextent = ax.get_ylim()[1] - ax.get_ylim()[0]

    arcsec_per_kpc = 1 / cosmo.kpc_proper_per_arcmin(cz / 3e5).value * 60 * 1e3
    size_vertical = 1e-2 * yextent

    height = 3e-2 * yextent

    if distance == '50pc':
        width = .05 * arcsec_per_kpc
        label = r'$50\,$pc'

    elif distance == '100pc':
        width = .1 * arcsec_per_kpc
        label = r'$100\,$pc'

    elif distance == '500pc':
        width = .5 * arcsec_per_kpc
        label = r'$500\,$pc'

    elif distance == '1kpc':
        width = 1 * arcsec_per_kpc
        label = r'$1\,$kpc'

    elif distance == '5kpc':
        width = 5 * arcsec_per_kpc
        label = r'$5\,$kpc'

    elif distance == '10kpc':
        width = 10 * arcsec_per_kpc
        label = r'$10\,$kpc'

    else:
        raise ValueError('Specify distance scale!')

    xy = (loc[0] - width / xextent / 2, loc[1] - height / yextent / 2)

    rect = patches.Rectangle(xy, width / xextent, height / yextent, linewidth=1, edgecolor=c,
                             facecolor=c, transform=ax.transAxes)

    ax.add_patch(rect)

    tloc = (loc[0], xy[1] - 3 * (height / yextent))
    ax.text(*tloc, label, c=c, fontsize=15, ha='center', va='center', transform=ax.transAxes)


# Functions for sspectrum class

def plot_AGNspec_model(spectrum, axes=None, savefig=True, path='Output/'):
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
    fig, axes = plt.subplots(1, 1, sharey=True, figsize=(6, 4), dpi=200)

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
        astrometry.makedir(path)
        fig.savefig(path + 'Output/AGNspec_model.png', bbox_inches='tight')

    return None


# Functions for spectroastrometry class

def plotly_spectrum(spectrum):
    """
    Generates an interactive HTML plot of the best fit model
    """

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # plt.style.use('dark_background')  # For cool tron-style dark plots

    # Open the best_fit_components file
    hdu = fits.open(os.path.join("Output", "best_model_components.fits"))
    tbdata = hdu[1].data  # FITS table data is stored on FITS extension 1
    cols = [i.name for i in tbdata.columns]
    hdu.close()

    # Create a figure with subplots
    fig = make_subplots(rows=2, cols=1, row_heights=(3, 1))
    # tracenames = []
    # Plot

    tracename = "Data"
    fig.add_trace(go.Scatter(x=tbdata["wvl"], y=tbdata["data"], mode="lines",
                             line=go.scatter.Line(color="black", width=1), name=tracename, legendrank=1,
                             showlegend=True), row=1, col=1)
    tracename = "Model"
    fig.add_trace(
        go.Scatter(x=tbdata["wvl"], y=tbdata["eline_model"] + tbdata["powerlaw"], mode="lines",
                   line=go.scatter.Line(color="red", width=1),
                   name=tracename, legendrank=2, showlegend=True), row=1, col=1)
    # tracename = "Noise"
    # fig.add_trace(go.Scatter(x=tbdata["wvl"], y=tbdata["error"], mode="lines",
    #                         line=go.scatter.Line(color="#FE00CE", width=1), name=tracename, legendrank=3,
    #                         showlegend=True), row=1, col=1)

    tracename = "Power-law"
    fig.add_trace(go.Scatter(x=tbdata["wvl"], y=tbdata["powerlaw"], mode="lines",
                             line=go.scatter.Line(color="red", width=1, dash="dash"), name=tracename,
                             legendrank=5, showlegend=True), row=1, col=1)

    # emission line components

    colors = ["#00B5F7", "#22FFA7", "#FC0080", "#DA16FF", "rgb(153,201,59)"]
    for idx, comp in enumerate(spectrum.components):
        for eline in spectrum.components[comp]:
            # tracename="narrow line"
            fig.add_trace(go.Scatter(x=tbdata["wvl"], y=tbdata[eline], mode="lines",
                                     line=go.scatter.Line(color=colors[idx], width=1), name=eline.split('_')[0],
                                     legendgroup=comp, legendgrouptitle_text=comp, legendrank=11 + idx),
                          row=1, col=1)

        # tracenames.append(tracename)
        '''
        if line_list[comp]["line_type"]=="br":
              # tracename="broad line"
            fig.add_trace(go.Scatter( x = tbdata["wvl"], y = tbdata[comp], mode="lines", line=go.scatter.Line(color="#22FFA7", width=1), name=comp, legendgroup="broad lines",legendgrouptitle_text="broad lines", legendrank=13,), row=1, col=1)
              # tracenames.append(tracename)
        if line_list[comp]["line_type"]=="out":
              # tracename="outflow line"
            fig.add_trace(go.Scatter( x = tbdata["wvl"], y = tbdata[comp], mode="lines", line=go.scatter.Line(color="#FC0080", width=1), name=comp, legendgroup="outflow lines",legendgrouptitle_text="outflow lines", legendrank=14,), row=1, col=1)
              # tracenames.append(tracename)
        if line_list[comp]["line_type"]=="abs":
              # tracename="absorption line"
            fig.add_trace(go.Scatter( x = tbdata["wvl"], y = tbdata[comp], mode="lines", line=go.scatter.Line(color="#DA16FF", width=1), name=comp, legendgroup="absorption lines",legendgrouptitle_text="absorption lines", legendrank=15,), row=1, col=1)
              # tracenames.append(tracename)
        if line_list[comp]["line_type"]=="user":
              # tracename="absorption line"
            fig.add_trace(go.Scatter( x = tbdata["wvl"], y = tbdata[comp], mode="lines", line=go.scatter.Line(color="rgb(153,201,59)", width=1), name=comp, legendgroup="user lines",legendgrouptitle_text="user lines", legendrank=16,), row=1, col=1)
              # tracenames.append(tracename)
        '''
    fig.add_hline(y=0.0, line=dict(color="gray", width=2), row=1, col=1)

    # Plot bad pixels
    # lam_gal = tbdata["wvl"]
    # ibad = [i for i in range(len(lam_gal)) if i not in fit_mask]
    # if (len(ibad)>0):# and (len(ibad[0])>1):
    # 	bad_wvl = [(lam_gal[m],lam_gal[m+1]) for m in ibad if ((m+1)<len(lam_gal))]
    # 	# ax1.axvspan(bad_wvl[0][0],bad_wvl[0][0],alpha=0.25,color='xkcd:lime green',label="bad pixels")
    # 	fig.add_vrect(
    # 					x0=bad_wvl[0][0], x1=bad_wvl[0][0],
    # 					fillcolor="rgb(179,222,105)", opacity=0.25,
    # 					layer="below", line_width=0,name="bad pixels",
    # 					),
    # 	for i in bad_wvl[1:]:
    # 		# ax1.axvspan(i[0],i[0],alpha=0.25,color='xkcd:lime green')
    # 		fig.add_vrect(
    # 						x0=i[0], x1=i[1],
    # 						fillcolor="rgb(179,222,105)", opacity=0.25,
    # 						layer="below", line_width=0,name="bad pixels",
    # 					),

    '''
    # Residuals
    fig.add_trace(go.Scatter( x = tbdata["wvl"], y = tbdata["RESID"], mode="lines", line=go.scatter.Line(color="white"  , width=1), name="Residuals", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter( x = tbdata["wvl"], y = tbdata["NOISE"], mode="lines", line=go.scatter.Line(color="#FE00CE"  , width=1), name="Noise", showlegend=False, legendrank=3,), row=2, col=1)
    # Figure layout, size, margins
    fig.update_layout(
        autosize=False,
        width=1700,
        height=800,
        margin=dict(
            l=100,
            r=100,
            b=100,
            t=100,
            pad=1
        ),
        title= objname,
        font_family="Times New Roman",
        font_size=16,
        font_color="white",
        legend_title_text="Components",
        legend_bgcolor="black",
        paper_bgcolor="black",
        plot_bgcolor="black",
    )
    '''
    # Update x-axis properties
    fig.update_xaxes(title=r"$\Large\lambda_{\rm{rest}}\;\left[Å\right]$", linewidth=0.5, linecolor="gray", mirror=True,
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
    fig.update_yaxes(title=r"$\Large\Delta f_\lambda$", linewidth=0.5, linecolor="gray", mirror=True,
                     gridwidth=1, gridcolor="#222A2A", zerolinewidth=2, zerolinecolor="#222A2A",
                     row=2, col=1)

    fig.update_xaxes(matches='x')
    # fig.update_yaxes(matches='y')
    # fig.show()

    # Write to HTML
    fig.write_html(os.path.join("Output", "AGNspec_model.html"), include_mathjax="cdn")
    # Write to PDF
    # fig.write_image(run_dir.joinpath("%s_bestfit.pdf" % objname))

    fig.show()
    return None


def plot_spectrum(astrometry, speccoor=[0,0], gs=None, savefig=False):
    """
        Plots a spectrum from the minicube

        Parameters
        ----------
        speccoor : `tuple`
            (x,y) coordinates from where the spectrum in the cube will be extracted
        gs : `matplotlib.gridspec.GridSpec` [optional]
            optional, existing GridSpec to which the plot will be added
        savefig : `boolean` [optional]
            saves plot as .png file
    """

    # plt.style.use('default')

    # get model spectrum by adding the best-fit components
    _, continuum_fit = astrometry.spectrum.subtract_continuum(astrometry.wvl, astrometry.cube.data[:, speccoor[0], speccoor[1]])
    bestfit_spectrum = continuum_fit
    for idx, comp in enumerate(astrometry.spectrum.components):
        spec_fit = getattr(astrometry.fluxmap, comp)[speccoor[0], speccoor[1]] * getattr(astrometry.basis, comp)
        bestfit_spectrum += spec_fit

    spec = astrometry.cube.data[:, speccoor[0], speccoor[1]]
    err = astrometry.cube.error[:, speccoor[0], speccoor[1]]
    res = spec - bestfit_spectrum

    #         *** plotting***

    colors = ["#00B5F7", "#22FFA7", "#FC0080", "#DA16FF", "rgb(153,201,59)"]
    xlabel = r'rest-frame wavelength $\lambda \, [\rm{\AA}]$'
    ylabel = r'$f_\lambda \,\, [10^{-16} \rm{erg/s/cm}^{2}/\rm{A}]$'
    rwindow = 5

    if gs == None:
        fig, axes = plt.subplots(figsize=(8, 8), dpi=100)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0)

    gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0], hspace=0)
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[rwindow, 1], hspace=0)

    ax0 = plt.subplot(gs0[0])
    ax0.step(astrometry.wvl + .5 * 1.25, astrometry.cube.AGN_spectrum, linewidth=1, color='k', label='AGN')
    for idx, comp in enumerate(astrometry.spectrum.components):
        spec_init = getattr(astrometry.basis, comp)
        spec_init_norm = spec_init / np.nanmax(spec_init)
        ax0.step(astrometry.wvl + .5 * 1.25, spec_init_norm, linewidth=1, color=colors[idx], label=comp)
    ax0.legend(fontsize=8)
    ax0.set_xlim(min(astrometry.wvl), max(astrometry.wvl))
    ax0.set_ylim(1e-4 * np.nanmax(astrometry.cube.AGN_spectrum))

    # fit result
    ax1 = plt.subplot(gs1[0])
    ax1.step(astrometry.wvl + .5 * 1.25, spec, color='k', linewidth=1, label='AGN')
    for idx, comp in enumerate(astrometry.spectrum.components):
        spec_fit = getattr(astrometry.fluxmap, comp)[speccoor[0], speccoor[1]] * getattr(astrometry.basis, comp)
        ax1.fill_between(astrometry.wvl + .5 * 1.25, spec_fit, linewidth=1, color=colors[idx], label=comp)
    ax1.plot(astrometry.wvl, bestfit_spectrum, linewidth=1, c='firebrick', label='model')
    ax1.legend(fontsize=8)
    ax1.set_xlim(min(astrometry.wvl), max(astrometry.wvl))
    ax1.set_ylim(1e-4 * np.nanmax(spec))

    # residuals
    ax2 = plt.subplot(gs1[1])
    ax2.step(astrometry.wvl + .5 * 1.25, res / err, color='k', linewidth=1)
    ax2.fill_between(astrometry.wvl + .5 * 1.25, -3, 3, color='firebrick', edgecolor='white', alpha=.2)
    ax2.set_xlim(min(astrometry.wvl), max(astrometry.wvl))
    ax2.set_ylim(-6, 6)

    # plot parameters

    # ticks
    ax0.tick_params(axis='both', labelbottom=False)
    ax1.tick_params(axis='both', labelbottom=False)

    # labels
    if gs == None:
        ax0.set_ylabel(r'$f_\lambda \,\, [10^{-16} \rm{erg/s/cm}^{2}/\rm{\AA}]$')
        ax1.set_ylabel(r'$f_\lambda \,\, [10^{-16} \rm{erg/s/cm}^{2}/\rm{\AA}]$')
    else:
        ax1.set_ylabel(r'$f_\lambda \,\, [10^{-16} \rm{erg/s/cm}^{2}/\rm{\AA}]$')
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


def plot_maps(astrometry, gs=None, mapcomp=['broad', 'core', 'wing'], savefig=False):
    """
        Plots maps of the kinematic components

        Parameters
        ----------
        gs : `matplotlib.gridspec.GridSpec` [optional]
            existing GridSpec to which the plot will be added
        savefig : `boolean` [optional]
            saves plot as .png file
    """

    extent = np.array([-(astrometry.cube.ncrop) / 2, (astrometry.cube.ncrop) / 2,
                       -(astrometry.cube.ncrop) / 2, (astrometry.cube.ncrop) / 2
                       ]
                      )

    extent *= astrometry.pxsize * 1e3  # implement cellsize in cube!

    if gs == None:
        fig = plt.figure(figsize=(9, 2 + 1.5 * len(astrometry.fluxmap.__dict__)), dpi=150)
        gs = gridspec.GridSpec(3, len(mapcomp)+1, wspace=.07, hspace=.06)
        print('paperlapapp')

    # Top row: flux maps
    cmap = mpl.cm.get_cmap('gist_earth_r')
    for idx, comp in enumerate(mapcomp):
        fluxmap = getattr(astrometry.fluxmap, comp)
        ax = plt.subplot(gs[0, idx])
        im = ax.imshow(fluxmap / np.nanmax(fluxmap), origin='lower', extent=extent, cmap=cmap,
                       norm=LogNorm(vmin=2e-2, vmax=1))
        scalebar(ax, astrometry.cz, c='k', loc=(.5, .22), distance='50pc')

        # annotations
        ax.annotate(comp, xy=(0.9, 0.85), fontsize=14, ha='right', xycoords='axes fraction')
        ax.annotate('(' + chr(0 * len(mapcomp) + idx+97) + ')',
                    ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')
        if idx == 0: ax.annotate(r'Data', xy=(0.9, .7), fontsize=14, ha='right', xycoords='axes fraction')

        # axes labels and ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', labelbottom=False)
        if idx == 0: ax.set_ylabel(r'$\Delta \,  \delta \,[{\rm mas}]$', labelpad=-7)
        else: ax.tick_params(axis='both', labelleft=False)

        # colorbar
        if idx == len(mapcomp)-1:
            ax = plt.subplot(gs[0, idx+1])
            cbarlabel = r'$ \Sigma$'
            colorbar(ax, im, label=cbarlabel)



    # Row 2: Model light distribution
    for idx, comp in enumerate(mapcomp):
        fluxmap = getattr(astrometry.fluxmodel, comp)
        ax = plt.subplot(gs[1, idx])
        im = ax.imshow(fluxmap / np.nanmax(fluxmap), origin='lower', extent=extent, cmap=cmap,
                       norm=LogNorm(vmin=2e-2, vmax=1))
        scalebar(ax, astrometry.cz, c='k', loc=(.5, .22), distance='50pc')
        if idx == len(mapcomp) - 1:
            cbarlabel = r'$ \Sigma$'
            # colorbar(im, label=cbarlabel)

        # annotations
        if idx == 0: ax.annotate('Model', ha='right', xy=(0.9, 0.85), fontsize=14, xycoords='axes fraction')
        ax.annotate('(' + chr(1 * len(mapcomp) + idx + 97) + ')',
                    ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')

        # centroid location
        # add the location of the component, which is measured relative to the PSF position,
        # to the PSF's offset from zero (i.e. half the image size)
        # flipped sign for y-axis due to origin=lower argument

        ax.scatter((astrometry.loc.broad[0] + (astrometry.PSFmodel.x_0.value - astrometry.cube.ncrop / 2)) \
                   * astrometry.pxsize * 1000,
                   (astrometry.loc.broad[1] + (astrometry.PSFmodel.y_0.value - astrometry.cube.ncrop / 2)) \
                   * astrometry.pxsize * 1000,
                   marker='x', c='firebrick', s=40, label='AGN')
        if idx > 0:
            ax.scatter((getattr(astrometry.loc, comp)[0] + (astrometry.PSFmodel.x_0.value - astrometry.cube.ncrop / 2)) \
                       * astrometry.pxsize * 1000,
                       (getattr(astrometry.loc, comp)[1] + (astrometry.PSFmodel.y_0.value - astrometry.cube.ncrop / 2)) \
                       * astrometry.pxsize * 1000,
                       marker='x', c='gold', s=40, label='centroid')

        # add legend
        if idx == len(mapcomp) - 1:
            legend = ax.legend(fontsize=8, bbox_to_anchor=(0.98, 0.98), loc='upper right', framealpha=.5)
            legend.get_frame().set_alpha(.4)

        # axes labels and ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', labelbottom=False)
        if idx == 0: ax.set_ylabel(r'$\Delta \,  \delta \,[{\rm mas}]$', labelpad=-7)
        else: ax.tick_params(axis='both', labelleft=False)

        # colorbar
        if idx == len(mapcomp) - 1:
            ax = plt.subplot(gs[1, idx + 1])
            cbarlabel = r'$ \Sigma$'
            colorbar(ax, im, label=cbarlabel)

    # Row 3: Residual maps
    cmap = mpl.cm.get_cmap('seismic')
    for idx, comp in enumerate(mapcomp):

        residuals = (getattr(astrometry.fluxmap, comp) - getattr(astrometry.fluxmodel, comp)
                     ) / getattr(astrometry.fluxmap, comp)

        ax = plt.subplot(gs[2, idx])
        im = ax.imshow(residuals, origin='lower', extent=extent, cmap=cmap, vmin=-1, vmax=1)
        scalebar(ax, astrometry.cz, c='k', loc=(.5, .22), distance='50pc')

        if idx == len(mapcomp) - 1:
            cbarlabel = 'residual/error'
            #colorbar(im, label=cbarlabel)

        # annotations
        if idx == 0: ax.annotate('Residual', ha='right', xy=(0.9, 0.85), fontsize=14, xycoords='axes fraction')
        ax.annotate('(' + chr(2 * len(mapcomp) + idx + 97) +')',
                    ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')

        # labels and ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel(r'$\Delta \,  \alpha \,[{\rm mas}]$')
        if idx == 0: ax.set_ylabel(r'$\Delta \,  \delta \,[{\rm mas}]$', labelpad=-7)
        else: ax.tick_params(axis='both', labelleft=False)

        # colorbar
        if idx == len(mapcomp) - 1:
            ax = plt.subplot(gs[2, idx + 1])
            cbarlabel = r'$ \Sigma$'
            colorbar(ax, im, label=cbarlabel)

    if savefig:
        astrometry.makedir(path)
        plt.savefig(path + 'Output/Spectroastrometry_maps.png', bbox_inches='tight')


def plot_all(astrometry, speccoor=[2, 2], mapcomp=['broad', 'core', 'wing'], savefig=True, path='.'):
    """
        Plots both spectra before/after fitting together with the
        surface brightness maps of the kinematic components

        Parameters
        ----------
        speccoor : `tuple`
            (x,y) coordinates from where the spectrum in the cube will be extracted

        mapcomp : `list`
            Kinematic components for which maps will be plotted. Must be a subset of the different
            components in the elines.par file.

        savefig : `boolean`
            writes figure if true.

        path : `boolean`
            path relative to working dir where the 'Ouput" directory will be created.
    """

    set_rc_params()

    fig = plt.figure(figsize=(18, 7), dpi=150)
    outer = gridspec.GridSpec(1, 2, wspace=0.2, width_ratios=[5, len(mapcomp)*2+.2])

    inner1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0],
                                              height_ratios=[1, 1.2], hspace=0)

    inner2 = gridspec.GridSpecFromSubplotSpec(3, len(mapcomp)+1,
                                              subplot_spec=outer[1],
                                              wspace=.07, hspace=.06,
                                              width_ratios=np.append(np.ones(len(mapcomp)), .07)
                                              )

    plot_spectrum(astrometry, speccoor=speccoor, gs=inner1, savefig=False)
    plot_maps(astrometry, gs=inner2, mapcomp=mapcomp, savefig=False)

    plt.show()

    if savefig:
        astrometry.makedir(path)
        plt.savefig(path + '/Output/Spectroastrometry.jpg', bbox_inches='tight')

    return fig


def print_result(astrometry):
    """
        Print the spectroastrometry result
    """

    print('\n')
    for component in astrometry.spectrum.components:
        # [px]
        px, dpx = astrometry.get_offset(component)

        # [arcsec]
        arcsec = px * astrometry.pxsize
        darcsec = dpx * astrometry.pxsize

        # [pc]
        d_obj = cosmo.comoving_distance(astrometry.cz / 3e5)
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
                                                         '1e-16 ergs-1cm-2'
                                                         )
              )
    print('\n')

    return None
