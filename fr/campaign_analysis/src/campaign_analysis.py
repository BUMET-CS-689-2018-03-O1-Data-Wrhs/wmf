from db_utils import query_lutetium, get_time_limits
from datetime import timedelta
from plot_utils import plot_df
import pandas as pd
import collections
import copy
from db_utils import get_hive_timespan, query_hive_ssh

def get_regs(dsk_campaign, mob_campaign, ipd_campaign):
    #B1516_daymonth_langCOUNTRY_dsk_bannerphase_bannertype_variable
    #B1516_0727_jaJP_dsk_p1_lg_txt_cnt

    p1 = '.*p1.*'
    p2 = '.*p2.*'
    dsk_lg = p1 + dsk_campaign
    dsk_sm = p2 + dsk_campaign

    mob_lg = p1 + mob_campaign
    mob_sm = p2 + mob_campaign

    ipd_lg = p1 + ipd_campaign
    ipd_sm = p2 + ipd_campaign

    # set up regular expressions for grouping data
    dsk = '|'.join([dsk_lg, dsk_sm]) 
    mob = '|'.join([mob_lg, mob_sm])
    ipd = '|'.join([ipd_lg, ipd_sm])
    lg = '|'.join([dsk_lg, mob_lg, ipd_lg]) 
    sm = '|'.join([dsk_sm, mob_sm, ipd_sm]) 


    all_regs = collections.OrderedDict()
    all_regs['Desktop Large'] = dsk_lg
    all_regs['Desktop Small'] = dsk_sm
    all_regs['Mobile Large'] = mob_lg
    all_regs['Mobile Small'] = mob_sm
    all_regs['Ipad Large'] = ipd_lg
    all_regs['Ipad Samll'] = ipd_sm


    device_regs = collections.OrderedDict()
    device_regs['Desktop'] = dsk
    device_regs['Mobile'] = mob
    device_regs['Ipad'] = ipd

    size_regs = collections.OrderedDict()
    size_regs['Large'] = lg
    size_regs['Small'] = sm

    dsk_regs = collections.OrderedDict()
    dsk_regs['Large Desk'] = dsk_lg
    dsk_regs['Small Desk'] = dsk_sm

    mob_regs = collections.OrderedDict()
    mob_regs['Large Mob'] = mob_lg
    mob_regs['Small Mob'] = mob_sm

    ipd_regs = collections.OrderedDict()
    ipd_regs['Large Ipad'] = ipd_lg
    ipd_regs['Small Ipad'] = ipd_sm

    lg_regs = collections.OrderedDict()
    lg_regs['Large Desk'] = dsk_lg
    lg_regs['Large Mobile'] = mob_lg
    lg_regs['Large Ipad'] = ipd_lg


    sm_regs = collections.OrderedDict()
    sm_regs['Small Desk'] = dsk_sm
    sm_regs['Small Mobile'] = mob_sm
    sm_regs['Small Ipad'] = ipd_sm

    return all_regs, device_regs, size_regs, dsk_regs,mob_regs, ipd_regs, lg_regs, sm_regs



def get_pageviews(start, stop, country, project):
    
    query = """
    SELECT year, month, day, hour, SUM(view_count) as pageviews, access_method FROM wmf.projectview_hourly
    WHERE agent_type = 'user'
    AND %(time)s
    AND project = '%(project)s'
    AND country_code = '%(country)s'
    GROUP BY year, month, day, hour, access_method
    """
    
    params = {'country': country, 'project': project, 'time': get_hive_timespan(start, stop) }
    d = query_hive_ssh(query % params, 'pvquery' + country + project, priority = True, delete = True)
    dt = d["year"].map(str) + '-' + d["month"].map(str) + '-' + d["day"].map(str) + ' ' + d["hour"].map(str) + ':00'
    d.index = pd.to_datetime(dt)

    del d['year']
    del d['month']
    del d['day']
    del d['hour']
    return d


def plot_traffic(pv, imp, source, reg, start, stop, hours = 1):
    pv_s  = copy.copy(pv[pv['access_method'] == source])
    pv_s.index = pd.Series(pv_s.index).apply(lambda tm: tm - timedelta(hours=(24 * tm.day + tm.hour) % hours))
    pv_s = pv_s.groupby(pv_s.index).sum()
    pv_s.rename(columns={'pageviews': source + ' pageviews'}, inplace=True)

    imp_s = copy.copy(imp.ix[imp.name.str.match(reg).apply(bool)][['n']])
    imp_s.index = pd.Series(imp_s.index).apply(lambda tm: tm - timedelta(hours=(24 * tm.day + tm.hour) % hours))
    imp_s = imp_s.groupby(imp_s.index).sum()

    imp_s.rename(columns={'n': source + ' impressions'}, inplace=True)

    d = pv_s.merge(imp_s, how = 'left', left_index = True, right_index = True)[[source + ' pageviews', source + ' impressions']]
    return plot_df(d, 'count per %d hours' % hours)



def plot_by_time(d, regs, start = '2000', stop = '2050', hours = 1, amount = False, cum = False, normalize = False, ylabel = '', interactive = False, index = None, rotate=False):
    
    d = d[start:stop]

    if index is None:
        d.index = pd.Series(d.index).apply(lambda tm: tm - timedelta(hours=(24 * tm.day + tm.hour) % hours))
    else:
        d.index = d[index]

    d_plot = pd.DataFrame()
    for name, reg in regs.items():
        if amount:
            counts = d.ix[d.name.str.match(reg).apply(bool)]['dollars']
        else:
            counts = d.ix[d.name.str.match(reg).apply(bool)]['n']

        if normalize:
            counts = counts/counts.sum()

        if cum:
            d_plot[name] = counts.groupby(counts.index).sum().cumsum()
        else:
            d_plot[name] = counts.groupby(counts.index).sum()

        if d_plot[name].shape[0] < 3:
            del d_plot[name]
            
            

    if d_plot.shape[0] < 3:
        print('There is no data for this campaign or this kind of banners')
        return

    d_plot = d_plot.fillna(0)


    plot_df(d_plot, ylabel, interactive = interactive, rotate=rotate)



def plot_rate_by_time(don, imp, regs,  hours = 1, start = '2000', stop = '2050', ylabel = 'donation rate', interactive = False, index = None):
    

    don = don[start:stop]
    imp = imp[start:stop]

    if index is None:
        don.index = pd.Series(don.index).apply(lambda tm: tm - timedelta(hours=(24 * tm.day + tm.hour) % hours))
        imp.index = pd.Series(imp.index).apply(lambda tm: tm - timedelta(hours=(24 * tm.day + tm.hour) % hours))
    else:
        don.index = don[index]
        imp.index = imp[index]

    d_plot = pd.DataFrame()
    for name, reg in regs.items():
        dons = don.ix[don.name.str.match(reg).apply(bool)]['n']
        dons = dons.groupby(dons.index).sum()
        imps = imp.ix[imp.name.str.match(reg).apply(bool)]['n']
        imps = imps.groupby(imps.index).sum()

        d_rate = dons/imps

        if d_rate.shape[0] < 3:
            continue

        largest = d_rate.nlargest(2).ix[1]
        d_rate[d_rate > largest] = largest
        d_plot[name] = d_rate

        

    if d_plot.shape[0] < 3:
        print('There is no data for this campaign or this kind of banners')
        return
    
    return plot_df(d_plot, ylabel, interactive = interactive)


def get_dollar_break_downs(don, regs):
    d_totals = pd.DataFrame()
    for name, reg in regs.items():
        counts = don.ix[don.name.str.match(reg).apply(bool)]['dollars']
        if counts.shape[0] != 0: 
            d_totals[name] = [int(counts.sum())]
    d_totals.index = ['Dollars']
    return d_totals

def get_donation_number_break_downs(don, regs):
    d_totals = pd.DataFrame()
    for name, reg in regs.items():
        counts = don.ix[don.name.str.match(reg).apply(bool)]['n']
        if counts.shape[0] != 0: 
            d_totals[name] = [counts.sum()]
    d_totals.index = ['# Donations']
    return d_totals


def get_average_donation_break_downs(don, regs):
    d_totals = pd.DataFrame()
    for name, reg in regs.items():
        counts = don.ix[don.name.str.match(reg).apply(bool)]['n']
        amounts = don.ix[don.name.str.match(reg).apply(bool)]['dollars']
        if counts.shape[0] != 0: 
            d_totals[name] = ['%.2f' % (amounts.sum() / counts.sum())]
    d_totals.index = ['Average Donations Amount']
    return d_totals

def get_impression_break_downs(imp, regs):
    d_totals = pd.DataFrame()
    for name, reg in regs.items():
        counts = imp.ix[imp.name.str.match(reg).apply(bool)]['n']
        if counts.shape[0] != 0: 
            d_totals[name] = [counts.sum()]
    d_totals.index = ['# Impressions']
    return d_totals

def get_donation_rate_break_downs(don, imp, regs):
    d_totals = pd.DataFrame()
    for name, reg in regs.items():
        den = imp.ix[imp.name.str.match(reg).apply(bool)]['n']
        num = don.ix[don.name.str.match(reg).apply(bool)]['n']
        if den.shape[0] != 0: 
            d_totals[name] = ['%.5f' % (num.sum()/den.sum())]
    d_totals.index = ['Donation Rate']
    return d_totals
