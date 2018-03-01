import matplotlib
import numpy as np

def plot_quantiles(bin_centers,perc_all,ax,**kwargs):
    """
    Plot output of stat_tool.get_quantiles as bar plots. 

    kwargs
    label : label of the data
    color : color of the bars and errorbars
    alpha : Transparency of the bar
    vlines : plot vertical lines between each bins

    """

    delta_bins = bin_centers[1] - bin_centers[0]

    ax.errorbar(bin_centers,perc_all['median'],yerr=[perc_all['median']-perc_all['10th'],perc_all['90th']-perc_all['median']],color=kwargs.get('color','black'),linestyle='',marker='o',label=kwargs.get('label',None))

    if (kwargs.get('vlines',False)):
        for pos in np.arange(bin_centers[1]-delta_bins/2.,bin_centers[-1]+delta_bins/2.,delta_bins):
            ax2.axvline(pos, color='k', linestyle=':')
            ax2.set_xlim(bin_centers[0],bin_centers[-1])

    for i in np.arange(bin_centers.size):
        rect = matplotlib.patches.Rectangle((bin_centers[i]-0.375*delta_bins,perc_all['25th'][i]),width=0.75*delta_bins,height=perc_all['75th'][i]-perc_all['25th'][i],color=kwargs.get('color','black'),alpha=kwargs.get('alpha',0.5),linewidth=1.)
        ax.add_patch(rect)


def create_GFED_canvas(label_x,label_y,dim=(4,4)):
    """
    Create a gridded canvas to plot data for each GFED regions. Return dictionary of Axes, referenced by GFED label.
    """

    label = ["BONA","TENA","CEAM","NHSA","SHSA","EURO","MIDE","NHAF","SHAF","BOAS","CEAS","SEAS","EQAS","AUST"]

    fig = matplotlib.pyplot.figure(figsize=(17,10))
    gs = matplotlib.gridspec.GridSpec(dim[0],dim[1])
    ax = {}

    for N_GFED in np.arange(0,14):
        pos_plot = [(N_GFED)%dim[0],(N_GFED)/dim[1]]
        ax[label[N_GFED]] = fig.add_subplot(gs[pos_plot[1],pos_plot[0]])

        if (pos_plot[0] == 0):
            ax[label[N_GFED]].set_ylabel(label_x,fontsize=10)
        if ((pos_plot[1] == 3) | (pos_plot[1] == 2) & (pos_plot[0] >= 2)):
            ax[label[N_GFED]].set_xlabel(label_y,fontsize=10)

    return ax
