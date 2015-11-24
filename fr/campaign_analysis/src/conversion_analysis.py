from db_utils import query_lutetium, get_time_limits
from datetime import timedelta
from plot_utils import plot_df
import pandas as pd
from civi_utils import get_clicks

methods = ['amazon', 'paypal', 'cc']

def get_conversion_clicks(start, stop, campaign):

    """
    Gets all donation data within the time range start:stop
    Groups data by banner, campaign and number of impressions seen
    """

    fields = ['payment_method', 'donation', 'country', 'name']
    return get_clicks(start, stop, campaign_reg = campaign, aggregation = 'hour', fields = fields):



def plot_conversion_rate(d, regs, start = '2000', stop = '2050', hours = 1, index = None, ylabel = 'conversion_rate',title= ''):
    if d.shape[0] == 0:
        print ('No Conversion rate data for this device')
        return

    d = d[start:stop]
    if index is None:
        d.index = pd.Series(d.index).apply(lambda tm: tm - timedelta(hours=(24 * tm.day + tm.hour) % hours))
    else:
        d.index = d[index]

    d_plot = pd.DataFrame()
    for name, reg in regs.items():
        clicks = d.ix[d.name.str.match(reg).apply(bool)]
        if clicks.shape[0] == 0:
            continue
        for method in methods:
            clicks_by_method = clicks[clicks['payment_method'] == method]
            if clicks_by_method.shape[0] == 0:
                continue
            donations = clicks_by_method[clicks_by_method['donation'] == 1]['n']
            if donations.shape[0] == 0:
                continue
            donations = donations.groupby(donations.index).sum()
            clicks_by_method = clicks_by_method.groupby(clicks_by_method.index)['n'].sum()
            d_plot[name+' '+ method] = donations / clicks_by_method

    if d_plot.shape[0] < 3:
        print ('No Conversion rate data for this device')
        return
    return plot_df(d_plot, ylabel=ylabel, title=title, interactive = False)




def get_conversion_rate_breakdowns(d, regs):
    d_totals = pd.DataFrame()
    for name, reg in regs.items():
        clicks = d.ix[d.name.str.match(reg).apply(bool)]
        for method in methods:
            clicks_by_method = clicks[clicks['payment_method'] == method]
            donations = clicks_by_method[clicks_by_method['donation'] == 1]['n']
            if clicks_by_method.shape[0] != 0: 
                d_totals[name+' '+ method] = [donations.sum() / clicks_by_method['n'].sum()]
    d_totals.index = ['Conversion Rate']
    return d_totals

