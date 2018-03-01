import subprocess
import time

def is_ready(max_nbJobsMe, max_nbJobsAll):
    commandQstatMe  = "qstat | grep plaurent | wc -l"
#    commandQstatAll  = "qstat | grep '*' | wc -l"

   ## Look if enough nod in the machine
    notDoProd = True
    while (notDoProd):
        nbJobsMe = int(subprocess.check_output(commandQstatMe, shell=True))
        nbJobsMax = int(subprocess.check_output(commandQstatAll, shell=True))
        if (nbJobsAll>0):
            nbJobsAll = nbJobsAll-2 
        if (nbJobsMe < max_nbJobsMe and nbJobsAll < max_nbJobsAll):
            notDoProd = False
        else:
            time.sleep(1)
    return notDoProd


def get_bar_stat(data_x,data_y,custom_bins=None,custom_label_x,range_y):
    median, bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic='median', bins=custom_bins)
    mean, bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic='mean', bins=custom_bins)
    count, bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic='count', bins=custom_bins)
    perc_10, bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic=lambda y: np.percentile(y, 10), bins=custom_bins)
    perc_90, bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic=lambda y: np.percentile(y, 90), bins=custom_bins)
    perc_25, bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic=lambda y: np.percentile(y, 25), bins=custom_bins)
    perc_75, bin_edges, binnumber = scipy.stats.binned_statistic(data_x,data_y,statistic=lambda y: np.percentile(y, 75), bins=custom_bins)

    return median,mean,count,perc_10,perc_25,perc_75,perc_100



