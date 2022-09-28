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

def colorbar(mappable, orientation="vertical", ticks=None, label=None, fontsize=14, format=None):

    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    if orientation == 'vertical':
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.tick_params(length=5, width=1, labelsize=.8 * fontsize)

        cb = fig.colorbar(mappable, cax=cax, orientation=orientation, format=format)
        cb.set_label(label, labelpad=5, fontsize=fontsize)

    elif orientation == 'horizontal':
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cax.xaxis.set_label_position('top')
        cax.xaxis.set_ticks_position('top')
        cax.tick_params(bottom=False, top=True, length=5, width=1, pad=-22, labelsize=.8 * fontsize)

        cb = fig.colorbar(mappable, cax=cax, orientation=orientation, format=format)
        cb.set_label(label, labelpad=-45, fontsize=fontsize)

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

def plot_AGNspec_model(spectrum,axes=None, savefig=True, path='Output/'):

    set_rc_params()

    # get line parameters from compound model
    Hb_broad_fit = spectrum.bestfit_model[0](spectrum.wvl*u.Angstrom)
    FeII4924_broad_fit = spectrum.bestfit_model[1](spectrum.wvl*u.Angstrom)
    FeII5018_broad_fit = spectrum.bestfit_model[2](spectrum.wvl * u.Angstrom)


    Hb_medium_fit = spectrum.bestfit_model[3](spectrum.wvl*u.Angstrom)
    FeII4924_medium_fit = spectrum.bestfit_model[4](spectrum.wvl*u.Angstrom)
    FeII5018_medium_fit = spectrum.bestfit_model[5](spectrum.wvl*u.Angstrom)

    Hb_core_fit = spectrum.bestfit_model[6](spectrum.wvl*u.Angstrom)
    OIII4959_core_fit = spectrum.bestfit_model[7](spectrum.wvl*u.Angstrom)
    OIII5007_core_fit = spectrum.bestfit_model[8](spectrum.wvl*u.Angstrom)

    Hb_wing_fit = spectrum.bestfit_model[9](spectrum.wvl*u.Angstrom)
    OIII4959_wing_fit = spectrum.bestfit_model[10](spectrum.wvl*u.Angstrom)
    OIII5007_wing_fit = spectrum.bestfit_model[11](spectrum.wvl*u.Angstrom)


    # Plot
    fig,axes=plt.subplots(1,1,sharey=True,figsize=(6,4), dpi=200)

    plt.step(spectrum.wvl+1.25/2, spectrum.AGN_spectrum, c='k', label='AGN')
    plt.plot(spectrum.wvl, spectrum.eline_model + spectrum.cont*u.Jy, '-', linewidth=1, c='red', label=r'Compound Model')

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
    #plt.xlim(spectrum.fit_range[0].value, spectrum.fit_range[1].value)
    # Axis labels for Spectrum Plots
    xlabel=r'rest-frame wavelength $\lambda \, [\rm{\AA}]$'
    ylabel=r'$F_\lambda \,\, [10^{-16}\, \rm{ergs}^{-1}\rm{cm}^{-2}\rm{A}^{-1}]$'

    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)

    if savefig:
        fig.savefig(path + 'AGNspec_model.png', bbox_inches='tight')

    return None


# Functions for spectroastrometry class

def plotly_spectrum(spectrum):

    """
    Generates an interactive HTML plot of the best fit model
    """

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plt.style.use('dark_background')  # For cool tron-style dark plots

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
    #tracename = "Noise"
    #fig.add_trace(go.Scatter(x=tbdata["wvl"], y=tbdata["error"], mode="lines",
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
                                     legendgroup=comp, legendgrouptitle_text=comp, legendrank=11+idx),
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

def plot_spectrum(astrometry, coor=None, gs=None, savefig=False):

    """
        Plots a spectrum from the minicube

        Parameters
        ----------
        coor : `tuple`
            (x,y) coordinates from where the spectrum in the cube will be extracted
        gs : `matplotlib.gridspec.GridSpec` [optional]
            optional, existing GridSpec to which the plot will be added
        savefig : `boolean` [optional]
            saves plot as .png file
    """

    # get base spectra (normalized to 1 erg-scm-2) and
    # approx. rescale to maximum flux density of the AGN spectrum
    broad_init = astrometry.basis.broad
    broad_init = broad_init / np.nanmax(astrometry.basis.broad) * .3 * np.nanmax(astrometry.cube.AGN_spectrum)
    core_init = astrometry.basis.core_OIII # + astrometry.basis.core_Hb
    core_init = core_init / np.nanmax(core_init) * .3 * np.max(astrometry.cube.AGN_spectrum)
    wing_init = astrometry.basis.wing_OIII  # astrometry.basis.wing_Hb+
    wing_init = wing_init / np.nanmax(wing_init) * .3 * np.max(astrometry.cube.AGN_spectrum)

    # fit result, scaled to AGN_spectrum
    broad_fit = astrometry.fluxmap.broad[coor[0], coor[1]] * astrometry.basis.broad
    core_fit = astrometry.fluxmap.core_OIII[coor[0], coor[1]] * astrometry.basis.core_OIII #\
                #+ astrometry.fluxmap.core_Hb[coor[0], coor[1]] * astrometry.basis.core_Hb
    wing_fit = astrometry.fluxmap.wing_OIII[coor[0], coor[1]] * astrometry.basis.wing_OIII
    # + astrometry.fluxmap.wing_Hb[coor[0],coor[1]]*astrometry.basis.wing_Hb \
    _, continuum_fit = astrometry.spectrum.subtract_continuum(astrometry.wvl, astrometry.cube.data[:, coor[0], coor[1]])
    model_fit = continuum_fit + broad_fit + core_fit + wing_fit

    spec = astrometry.cube.data[:, coor[0], coor[1]]
    err = astrometry.cube.error[:, coor[0], coor[1]]
    res = spec - model_fit

    #         *** plotting***

    # init model

    xlabel = r'rest-frame wavelength $\lambda \, [\rm{\AA}]$'
    ylabel = r'$f_\lambda \,\, [10^{-16} \rm{erg/s/cm}^{2}/\rm{A}]$'
    rwindow = 5

    if gs == None:
        fig, axes = plt.subplots(figsize=(8, 8), dpi=100)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0)

    gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0], hspace=0)
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[rwindow, 1], hspace=0)

    ax0 = plt.subplot(gs0[0])
    ax0.step(astrometry.wvl + .5 * 1.25, astrometry.cube.AGN_spectrum,
             linewidth=1, color='k', label='AGN')
    ax0.plot(astrometry.wvl, broad_init, color='cornflowerblue',
             linestyle=ls['densely dashed'], linewidth=.8, label='broad')
    ax0.plot(astrometry.wvl, core_init, color='lightcoral',
             linestyle=ls['densely dashdotted'], linewidth=.8, label='core')
    ax0.plot(astrometry.wvl, wing_init, color='limegreen',
             linestyle=ls['densely dashdotdotted'], linewidth=.8, label='wing')
    ax0.legend(fontsize=10)
    ax0.set_xlim(min(astrometry.wvl), max(astrometry.wvl))
    ax0.set_ylim(1e-4 * np.nanmax(astrometry.cube.AGN_spectrum))

    # fit result
    ax1 = plt.subplot(gs1[0])
    ax1.step(astrometry.wvl + .5 * 1.25, spec, color='k', linewidth=1, label='AGN')
    ax1.fill_between(astrometry.wvl, broad_fit, facecolor='cornflowerblue', label='broad')
    ax1.fill_between(astrometry.wvl, core_fit, facecolor='lightcoral', label='core')
    ax1.fill_between(astrometry.wvl, wing_fit, facecolor='limegreen', label='wing')
    ax1.plot(astrometry.wvl, model_fit, linewidth=1, c='firebrick', label='model')
    ax1.legend(fontsize=10)
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
        plt.savefig('Figures/spectroastrometry_spec.png', bbox_inches='tight', dpi=300)

def plot_maps(astrometry, gs=None, savefig=False):

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

    extent *= 0.025 * 1e3  # implement cellsize in cube!

    if gs == None:
        fig = plt.figure(figsize=(9, 6), dpi=150)
        gs = gridspec.GridSpec(3, 3, wspace=.07, hspace=.06, width_ratios=[1, 1, 1.1])

    # Flux maps

    component = 'broad'
    fluxmap = getattr(astrometry.fluxmap, component)
    ax00 = plt.subplot(gs[0, 0])
    cmap = mpl.cm.get_cmap('gist_earth_r')
    im = ax00.imshow(fluxmap / np.nanmax(fluxmap), origin='lower', extent=extent, cmap=cmap,
                     norm=LogNorm(vmin=2e-2, vmax=1))
    scalebar(ax00, astrometry.cz, c='k', loc=(.5, .22), distance='50pc')

    component = 'core'
    fluxmap = getattr(astrometry.fluxmap, component)
    ax01 = plt.subplot(gs[0, 1])
    cmap = mpl.cm.get_cmap('gist_earth_r')
    im = ax01.imshow(fluxmap / np.nanmax(fluxmap), origin='lower', extent=extent, cmap=cmap,
                     norm=LogNorm(vmin=2e-2, vmax=1))
    scalebar(ax01, astrometry.cz, c='k', loc=(.5, .22), distance='50pc')

    component = 'wing'
    fluxmap = getattr(astrometry.fluxmap, component)
    ax02 = plt.subplot(gs[0, 2])
    cmap = mpl.cm.get_cmap('gist_earth_r')
    im = ax02.imshow(fluxmap / np.nanmax(fluxmap), origin='lower', extent=extent, cmap=cmap,
                     norm=LogNorm(vmin=2e-2, vmax=1))
    # scalebar(ax02, astrometry.cz, c='k', loc=(.5,.22), distance='50pc')
    cbarlabel = r'$ \Sigma$'
    colorbar(im, label=cbarlabel)

    # PSF maps

    component = 'broad'
    fluxmap = getattr(astrometry.fluxmodel, component)
    ax10 = plt.subplot(gs[1, 0])
    cmap = mpl.cm.get_cmap('gist_earth_r')
    im = ax10.imshow(fluxmap / np.nanmax(fluxmap), origin='lower', extent=extent, cmap=cmap,
                     norm=LogNorm(vmin=2e-2, vmax=1))
    # scalebar(ax10, astrometry.cz, c='k', loc=(.5,.18), distance='50pc')

    component = 'core'
    fluxmap = getattr(astrometry.fluxmodel, component)
    ax11 = plt.subplot(gs[1, 1])
    cmap = mpl.cm.get_cmap('gist_earth_r')
    im = ax11.imshow(fluxmap / np.nanmax(fluxmap), origin='lower', extent=extent, cmap=cmap,
                     norm=LogNorm(vmin=2e-2, vmax=1))
    # scalebar(ax10, astrometry.cz, c='k', loc=(.5,.18), distance='50pc')

    component = 'wing'
    fluxmap = getattr(astrometry.fluxmodel, component)
    ax12 = plt.subplot(gs[1, 2])
    cmap = mpl.cm.get_cmap('gist_earth_r')
    im = ax12.imshow(fluxmap / np.nanmax(fluxmap), origin='lower', extent=extent, cmap=cmap,
                     norm=LogNorm(vmin=2e-2, vmax=1))
    # scalebar(ax12, astrometry.cz, c='k', loc=(.5,.18), distance='50pc')
    cbarlabel = r'$ \Sigma$'
    colorbar(im, label=cbarlabel)

    # Residual maps

    component = 'broad'
    residuals = (getattr(astrometry.fluxmap, component)
                 - getattr(astrometry.fluxmodel, component)
                 ) / getattr(astrometry.fluxmap, component)
    ax20 = plt.subplot(gs[2, 0])
    cmap = mpl.cm.get_cmap('seismic')
    im = ax20.imshow(residuals, origin='lower', extent=extent, cmap=cmap, vmin=-1, vmax=1)
    scalebar(ax20, astrometry.cz, c='k', loc=(.5, .22), distance='50pc')

    component = 'core'
    residuals = (getattr(astrometry.fluxmap, component)
                 - getattr(astrometry.fluxmodel, component)
                 ) / getattr(astrometry.fluxmap, component)
    ax21 = plt.subplot(gs[2, 1])
    cmap = mpl.cm.get_cmap('seismic')
    im = ax21.imshow(residuals, origin='lower', extent=extent, cmap=cmap, vmin=-1, vmax=1)
    scalebar(ax21, astrometry.cz, c='k', loc=(.5, .22), distance='50pc')

    component = 'wing'
    residuals = (getattr(astrometry.fluxmap, component)
                 - getattr(astrometry.fluxmodel, component)
                 ) / getattr(astrometry.fluxmap, component)
    ax22 = plt.subplot(gs[2, 2])
    cmap = mpl.cm.get_cmap('seismic')
    im = ax22.imshow(residuals, origin='lower', extent=extent, cmap=cmap, vmin=-1, vmax=1)
    scalebar(ax22, astrometry.cz, c='k', loc=(.5, .22), distance='50pc')
    cbarlabel = 'residual/error'
    colorbar(im, label=cbarlabel)

    # draw borad centroids
    ax00.scatter(*astrometry.loc.broad[:2] * 0.025e3, marker='x', c='firebrick', s=40)
    #ax01.scatter(*astrometry.loc.core_Hb[:2] * 0.025e3, marker='x', c='gold', s=40)
    ax01.scatter(*astrometry.loc.broad[:2] * 0.025e3, marker='x', c='firebrick', s=40)
    ax02.scatter(*astrometry.loc.wing_OIII[:2] * 0.025e3, marker='x', c='gold', s=40, label='centroid')
    ax02.scatter(*astrometry.loc.broad[:2] * 0.025e3, marker='x', c='firebrick', s=40, label='AGN')
    legend = ax02.legend(fontsize=8, bbox_to_anchor=(0.95, 0.3), framealpha=.5)
    legend.get_frame().set_alpha(.4)

    # annotations
    ax00.annotate('(a)', ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')

    ax00.annotate(r'H$\beta$ broad', xy=(0.9, 0.85), fontsize=14, ha='right', xycoords='axes fraction')
    ax00.annotate(r'Data', xy=(0.9, .7), fontsize=14, ha='right', xycoords='axes fraction')
    ax01.annotate('(b)', ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')
    ax01.annotate(r'[O$\:$III] core', xy=(0.9, 0.85), fontsize=14, ha='right', xycoords='axes fraction')
    ax02.annotate('(c)', ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')
    ax02.annotate(r'[O$\:$III] wing', xy=(0.9, 0.85), fontsize=14, ha='right', xycoords='axes fraction')
    ax10.annotate('(d)', ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')
    ax10.annotate('Model', ha='right', xy=(0.9, 0.85), fontsize=14, xycoords='axes fraction')
    ax11.annotate('(e)', ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')
    ax12.annotate('(f)', ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')
    ax20.annotate('(g)', ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')
    ax20.annotate('Residual', ha='right', xy=(0.9, 0.85), fontsize=14, xycoords='axes fraction')
    ax21.annotate('(h)', ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')
    ax22.annotate('(i)', ha='left', xy=(0.1, 0.85), fontsize=14, xycoords='axes fraction')

    # switch on minor ticks
    ax00.xaxis.set_minor_locator(AutoMinorLocator())
    ax00.yaxis.set_minor_locator(AutoMinorLocator())
    ax01.xaxis.set_minor_locator(AutoMinorLocator())
    ax01.yaxis.set_minor_locator(AutoMinorLocator())
    ax02.xaxis.set_minor_locator(AutoMinorLocator())
    ax02.yaxis.set_minor_locator(AutoMinorLocator())
    ax10.xaxis.set_minor_locator(AutoMinorLocator())
    ax10.yaxis.set_minor_locator(AutoMinorLocator())
    ax11.xaxis.set_minor_locator(AutoMinorLocator())
    ax11.yaxis.set_minor_locator(AutoMinorLocator())
    ax12.xaxis.set_minor_locator(AutoMinorLocator())
    ax12.yaxis.set_minor_locator(AutoMinorLocator())
    ax20.xaxis.set_minor_locator(AutoMinorLocator())
    ax20.yaxis.set_minor_locator(AutoMinorLocator())
    ax21.xaxis.set_minor_locator(AutoMinorLocator())
    ax21.yaxis.set_minor_locator(AutoMinorLocator())
    ax22.xaxis.set_minor_locator(AutoMinorLocator())
    ax22.yaxis.set_minor_locator(AutoMinorLocator())

    ax01.tick_params(axis='both', labelleft=False)
    ax02.tick_params(axis='both', labelleft=False)
    ax11.tick_params(axis='both', labelleft=False)
    ax12.tick_params(axis='both', labelleft=False)
    ax21.tick_params(axis='both', labelleft=False)
    ax22.tick_params(axis='both', labelleft=False)
    ax00.tick_params(axis='both', labelbottom=False)
    ax01.tick_params(axis='both', labelbottom=False)
    ax02.tick_params(axis='both', labelbottom=False)
    ax10.tick_params(axis='both', labelbottom=False)
    ax11.tick_params(axis='both', labelbottom=False)
    ax12.tick_params(axis='both', labelbottom=False)

    ax00.set_ylabel(r'$\Delta \,  \delta \,[{\rm mas}]$', labelpad=-7)
    ax10.set_ylabel(r'$\Delta \,  \delta \,[{\rm mas}]$', labelpad=-7)
    ax20.set_ylabel(r'$\Delta \,  \delta \,[{\rm mas}]$', labelpad=-7)
    ax20.set_xlabel(r'$\Delta \,  \alpha \,[{\rm mas}]$')
    ax21.set_xlabel(r'$\Delta \,  \alpha \,[{\rm mas}]$')
    ax22.set_xlabel(r'$\Delta \,  \alpha \,[{\rm mas}]$')

    if savefig:
        astrometry.makedir(path)
        plt.savefig(path+'spectroastrometry_maps.png', bbox_inches='tight')

def plot_all(astrometry, coor, savefig=True, path='.'):

    """
        Plots both spectra before/after fitting together with the
        surface brightness maps of the kinematic components

        Parameters
        ----------
        coor : `tuple`
            (x,y) coordinates from where the spectrum in the cube will be extracted
    """

    fig = plt.figure(figsize=(15, 7), dpi=150)
    outer = gridspec.GridSpec(1, 2, wspace=0.2, width_ratios=[2.5, 3])

    inner1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0],
                                              height_ratios=[1, 1.2], hspace=0)
    inner2 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[1],
                                              wspace=.07, hspace=.06, width_ratios=[1, 1, 1.1])

    plot_spectrum(astrometry, coor=[2, 2], gs=inner1, savefig=False)
    plot_maps(astrometry, gs=inner2, savefig=False)

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
        arcsec = px * 0.025
        darcsec = dpx * 0.025

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
