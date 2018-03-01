from string import ascii_lowercase as al
import iminuit
import scipy
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def detrend(data_x,data_y,deg=3,bins=None,plot=None):
    """
    EXPERIMENTAL, DO NOT USE !!
    Detrend function, need to be check before using. 


    """

    if bins is None:
        bins_x = np.linspace(np.amin(data_x),np.amin(data_x),20)
        bins_y = np.linspace(np.amin(data_y),np.amin(data_y),20)
    else:
        bins_x = bins[0]
        bins_y = bins[1]

    mean, bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y/np.mean(data_y),statistic='mean',bins=bins_x)
    count, bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y/np.mean(data_y),statistic='count',bins=bins_x)
    poly = np.polyfit(bin_edges[:-1],mean,deg=3)
    p = np.poly1d(poly)

    if plot:
        h, x_edges,y_edges = np.histogram2d(data_x,data_y/np.mean(data_y),bins=[np.logspace(2.,5.,40),np.linspace(0.,4.,40)])
        X, Y = np.meshgrid(x_edges, y_edges)

        plt.figure()
        plt.pcolormesh(X, Y, h.T,norm=LogNorm(vmin=1, vmax=1e5))
        plt.xscale('log')
        plt.errorbar(bin_edges[:-1],mean,yerr=mean/(np.sqrt(count)),color='red')
        plt.plot(bin_edges[:-1],p(bin_edges[:-1]))
        plt.show()

    data_y_corrected = data_y/p(data_x)
    return data_y_corrected 



def quantiles(data_x,data_y,bins=None):
    """
    Get median and 10th, 25th 75th and 90th quantiles of data_y in each bin of data_x. This can be used with the plot in plot_tools to display custoized bar plots.

    
    data_x,data_y : N-dim ndarrays (automatically linearized by the routines)
    bins : ndarray of bin edges of data_x

    return : centers of bin and dictionnary containing the quantiles, median and number of count in each bins

    """

    data_x = np.ravel(data_x)
    data_y = np.ravel(data_y)

    if (bins is None):
        bins = [np.amin(data_x),np.amax(data_x),20]

    perc_all = {}
    perc_all['count'], bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic='count', bins=bins)
    perc_all['median'], bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic='median', bins=bins)
    perc_all['10th'], bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic=lambda y: np.percentile(y, 10), bins=bins)
    perc_all['90th'], bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic=lambda y: np.percentile(y, 90), bins=bins)
    perc_all['25th'], bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic=lambda y: np.percentile(y, 25), bins=bins)
    perc_all['75th'], bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic=lambda y: np.percentile(y, 75), bins=bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2.

    return bin_centers, perc_all


def profile(data_x,data_y,bins=None,circ=None):
    """
    Get mean, standard deviation and number of countd of data_y in each bin of data_x.

    
    data_x,data_y : N-dim ndarrays (automatically linearized by the routines)
    bins : ndarray of bin edges of data_x
    circ : if defined as tuple of two elements, the routine will compute a circular mean in the range [circ[0], circ[1]]

    return : centers of bin and dictionnary containing the mean, std, and number of count in each bins.

    """

    data_x = np.ravel(data_x)
    data_y = np.ravel(data_y)

    if (bins is None):
        bins = [np.amin(data_x),np.amax(data_x),20]

    mean_dict = {}
    data_x = np.ravel(data_x)
    data_y = np.ravel(data_y)
    if (circ is not None):
        mean_dict['mean'], bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic=lambda y:scipy.stats.circmean(y,low=circ[0],high=circ[1]), bins=bins)
    else:
        mean_dict['mean'], bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic='mean', bins=bins)

    mean_dict['count'], bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic='count', bins=bins)
    mean_dict['std'], bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic=lambda y: np.mean((y-np.mean(y))**2), bins=bins)

    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2.
    return bin_centers, mean_dict


def fit_lognorm(x_to_fit,y_to_fit,err=None,sigma=1):
    """
    Fit the pdf of log normal distribution to the given data.

    
    data_x,data_y : 1d ndarrays of the data to fit
    err : 1d ndarray of uncertainty on data_y

    return : One dictionary containing the adjusted values, one containing the uncertainty, the chi2 value, and a list to plot the resulting function. 

    """
    if (err is None):
        err = np.ones(x_to_fit.size)

    def chi2(shape, loc, scale,a,b):
        return np.sum((a*scipy.stats.lognorm.pdf(x_to_fit, shape,loc,scale) - y_to_fit)**2/(err**2))

    m = iminuit.Minuit(chi2, shape=0.9,loc=0,scale=50,a=1000,print_level=1)
    m.migrad()
    
    x = np.linspace(np.amin(x_to_fit),np.amax(x_to_fit),200)
    y = m.values['a']*scipy.stats.lognorm.pdf(x,m.values['shape'],m.values['loc'],m.values['scale'])


    return m.values,m.errors,m.fval/x_to_fit.size,[x,y]




def fit_skewnorm(x_to_fit,y_to_fit,err=None,sigma=1):
    """
    Fit the pdf of a skew normal distribution to the given data.

    
    data_x,data_y : 1d ndarrays of the data to fit
    err : 1d ndarray of uncertainty on data_y

    return : One dictionary containing the adjusted values, one containing the uncertainty, the chi2 value, and a list to plot the resulting function. 

    """
    if (err is None):
        err = np.ones(x_to_fit.size)

    def chi2(shape, loc, scale,a,b):
        return np.sum((a*scipy.stats.skewnorm.pdf(x_to_fit, shape,loc,scale) - y_to_fit)**2/(err**2))

    m = iminuit.Minuit(chi2, shape=1.,loc=0,scale=50,a=10000,print_level=0)
    m.migrad()
    
    x = np.linspace(np.amin(x_to_fit),np.amax(x_to_fit),200)
    y = m.values['a']*scipy.stats.skewnorm.pdf(x,m.values['shape'],m.values['loc'],m.values['scale'])


    return m.values,m.errors,m.fval/x_to_fit.size,[x,y]

    

def fit_poly(x_to_fit,y_to_fit,err=None,sigma=1,deg=4):
    """
    Fit a polynomial to the given data.
    
    data_x,data_y : 1d ndarrays of the data to fit
    err : 1d ndarray of uncertainty on data_y

    return : One dictionary containing the adjusted values, one containing the uncertainty, the chi2 value, and a list to plot the resulting function. 

    """

    if (deg==2):
        func = lambda x,a,b,c: a*x**2 + b*x + c
        chi2 = lambda a,b,c: np.sum((func(x_to_fit, a,b,c) - y_to_fit)**2/(err**2))
    if (deg==3):
        func = lambda x,a,b,c,d: a*x**3 + b*x**2 + c*x + d
        chi2 = lambda a,b,c,d: np.sum((func(x_to_fit, a,b,c,d) - y_to_fit)**2/(err**2))
    if (deg==4):
        func = lambda x,a,b,c,d,e: a*x**4 + b*x**3 + c*x**2 + d*x + e
        chi2 = lambda a,b,c,d,e: np.sum((func(x_to_fit, a,b,c,d,e) - y_to_fit)**2/(err**2))
    if (deg==5):
        func = lambda x,a,b,c,d,e,f: a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f
        chi2 = lambda a,b,c,d,e,f: np.sum((func(x_to_fit, a,b,c,d,e,f) - y_to_fit)**2/(err**2))
        
    if (err is None):
        err = np.ones(x_to_fit.size)

    m = iminuit.Minuit(chi2,print_level=0)
    m.migrad()    

    x = np.linspace(np.amin(x_to_fit),np.amax(x_to_fit),200)
    if (deg==2):
        y = func(x,m.values['a'],m.values['b'],m.values['c'])
    if (deg==3):
        y = func(x,m.values['a'],m.values['b'],m.values['c'],m.values['d'])
    if (deg==4):
        y = func(x,m.values['a'],m.values['b'],m.values['c'],m.values['d'],m.values['e'])    
    if (deg==5):
        y = func(x,m.values['a'],m.values['b'],m.values['c'],m.values['d'],m.values['e'],m.values['f'])
        

    return m.values,m.errors,m.fval/x_to_fit.size,[x,y]


def fit_poly_dict(x_to_fit,y_to_fit,err=None,sigma=1,deg=4):
    """
    Fit a polynomial to the given data.
    
    data_x,data_y : 1d ndarrays of the data to fit
    err : 1d ndarray of uncertainty on data_y

    return : One dictionary containing the adjusted values, one containing the uncertainty, the chi2 value, and a list to plot the resulting function. 

    """

    dict_coeff = {x:(1,i) for i, x in enumerate(al[:(deg+1)], 1)}

    print dict_coeff

    poly = lambda x,**dict_coeff:np.sum([coeff[0]*x**coeff[1] for coeff in dict_coeff.keys()])
    chi2 = lambda **dict_coeff:np.sum((poly(x_to_fit,dict_coeff) - y_to_fit)**2/(err**2))

    # def poly(x,dict_coeff):
    #     print 'In poly :', np.sum([coeff[0]*x**coeff[1] for coeff in dict_coeff.keys()])
    #     return np.sum([coeff[0]*x**coeff[1] for coeff in dict_coeff.keys()])

    # def chi2(**dict_coeff):
    #     print a,b,c
    #     print 'In chi2 :', np.sum((poly(x_to_fit,dict_coeff) - y_to_fit)**2/(err**2))
    #     return np.sum((poly(x_to_fit,dict_coeff) - y_to_fit)**2/(err**2))

    # func(**{'type':'Event'})

    # if (deg==2):
    #     func = lambda x,a,b,c: a*x**2 + b*x + c
    #     chi2 = lambda a,b,c: np.sum((func(x_to_fit, a,b,c) - y_to_fit)**2/(err**2))
    # if (deg==3):
    #     func = lambda x,a,b,c,d: a*x**3 + b*x**2 + c*x + d
    #     chi2 = lambda a,b,c,d: np.sum((func(x_to_fit, a,b,c,d) - y_to_fit)**2/(err**2))
    # if (deg==4):
    #     func = lambda x,a,b,c,d,e: a*x**4 + b*x**3 + c*x**2 + d*x + e
    #     chi2 = lambda a,b,c,d,e: np.sum((func(x_to_fit, a,b,c,d,e) - y_to_fit)**2/(err**2))
        
    if (err is None):
        err = np.ones(x_to_fit.size)

    m = iminuit.Minuit(chi2,print_level=0)
    m.migrad()    

    x = np.linspace(np.amin(x_to_fit),np.amax(x_to_fit),200)
    y = func(x,m.values['a'],m.values['b'],m.values['c'],m.values['d'],m.values['e'])


    return m.values,m.errors,m.fval/x_to_fit.size,[x,y]

    
