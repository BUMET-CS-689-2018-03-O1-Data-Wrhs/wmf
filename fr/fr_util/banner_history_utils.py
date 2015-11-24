import os
from datetime import datetime
import dateutil.parser
from dateutil import relativedelta
import json
import json
from collections import defaultdict
from operator import add
import datetime
import re
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import pickle
import seaborn as sns
import numpy as np
import copy
from plot_utils import plot_df


"""
A pageview is either in the sample or out of the sample. 
If a pageview in the sample lead to a donation, the event is not resent on the donation.
If a pageview was not in the sample, and a donation occurs, the log is sent, but without an r parameter.
So, to get a sample of just donoions, look at events without r parameters.
To get an unbiased sample of impressions look at events with the r parameter.
Since most impressions don't lead to donations, you can use the unbiased sample as a proxy for non-donations.
You can link events to civi if the pageview led to a donation. Specifically you can get the log ids for donations.
This means you an clean up the sample and remove pageviews that lead to a donation and you can learn more about the donations!
"""

def load_rdd(sc, start, stop, hour = False):
    base = 'hdfs:/wmf/data/raw/eventlogging/eventlogging_CentralNoticeBannerHistory/hourly/'
    files = []
    start = dateutil.parser.parse(start)
    stop = dateutil.parser.parse(stop)

    while start <= stop:
        f = os.path.join(base, start.strftime('%Y/%m/%d') )

        if hour:
            f = os.path.join(base, start.strftime('%Y/%m/%d/%H') )
            start += relativedelta.relativedelta(hours=1)
        else:
            f = os.path.join(f, '*')
            start += relativedelta.relativedelta(days=1)
        files.append(f)

    file_str = ','.join(files)
    return sc.sequenceFile(file_str).map(lambda x: json.loads(x[1])['event'])



def get_donor_data(data):
    return data.filter(lambda x: 'r' not in x)


def get_sample_data(data):
    return data.filter(lambda x: 'r' in x)


def transform_datetimes(data):
    
    def f(x):
        for elem in x['l']:
         elem['t'] = datetime.datetime.fromtimestamp(elem['t']).strftime('%Y-%m-%d %H') #:%M:%S
        return x
    return data.map(f)


status_codes = {
 '0' : 'CAMPAIGN_NOT_CHOSEN',
 '1' : 'CAMPAIGN_CHOSEN',
 '2' : 'BANNER_CANCELED',
 '3' : 'NO_BANNER_AVAILABLE',
 '4' : 'BANNER_CHOSEN'    ,
 '5' : 'BANNER_LOADED_BUT_HIDDEN',
 '6' : 'BANNER_SHOWN',
}

reason_codes =  {
'0' : 'other',
'1' : 'close',
'2' : 'waitdate',
'3' : 'waitimps',
'4' : 'waiterr',
'5' : 'belowMinEdits',
'6' : 'viewLimit',
'7' : 'seen-fullscreen',
'8' : 'cookies-disabled',
'9' : 'donate'
}

def transform_reasons(data):

    def get_readable_reasons(h):
        for d in h['l']:
            status = d['s'].split('.')
            if len(status) == 1:
                status.append(None)
                  
            d['status'] = status_codes.get(status[0], 'UNKNOWN')
            d['reason'] = reason_codes.get(status[1], '')
        return h

    return data.map(get_readable_reasons)



def filter_dt(data, start, stop):
    def inlcude(x):
        last_elem = x['l'][-1]
        if last_elem['t'] < start:
            return False
        if last_elem['t'] > stop:
            return False

        return True

    return data.filter(inlcude)


def filter_campaign(data, campaign):
    def inlcude(x):
        last_elem = x['l'][-1]
        if 'c' in last_elem and last_elem['c'] == campaign:
            return True

        return False

    return data.filter(inlcude)


def filter_campaign_stop_gap(data, campaign, banner_reg):
    def inlcude(x):
        last_elem = x['l'][-1]
        if 'c' in last_elem and last_elem['c'] == campaign:
            return True
        if 'b' in last_elem and  re.match(banner_reg, last_elem['b']):
            return True

        return False

    return data.filter(inlcude)

def get_status_counts(data, hour = True):
    if hour:
        counts = data.map(lambda x: (x['l'][-1]['t'] + '|' + x['l'][-1]['status'] + '.' + x['l'][-1]['reason'], 1)).countByKey()
    else:
        counts = data.map(lambda x: (x['l'][-1]['status'] + '.' + x['l'][-1]['reason'], 1)).countByKey()
    return counts



def get_previous_impression_counts(data, mapping = lambda x: str(x), hour = True):
    impressions = data.filter(lambda x: x['l'][-1]['s'] == '6')
    if hour:
        counts = impressions.map(lambda x: (x['l'][-1]['t'] + '|%s' % mapping(len([e for e in x['l'] if e['s'] == '6'])) , 1)).countByKey()
    else:
        counts = impressions.map(lambda x: (mapping(len([e for e in x['l'] if e['s'] == '6'])), 1)).countByKey()
    return counts


def get_previous_pageview_counts(data, mapping = lambda x: str(x), hour = True):
    impressions = data.filter(lambda x: x['l'][-1]['s'] == '6')
    if hour:
        counts = impressions.map(lambda x: (x['l'][-1]['t'] + '|%s' % mapping(len(x['l'])), 1)).countByKey()
    else:
        counts = impressions.map(lambda x: (mapping(len(x['l'])), 1)).countByKey()
    return counts
    

def get_counts_df(counts,  hours, r):
    d = pd.DataFrame(list(counts.items()))
    d.columns = ['id', 'n']
    d['status'] = d['id'].apply(lambda x: x.split('|')[1])
    d['dt'] = d['id'].apply(lambda x: pd.to_datetime(x.split('|')[0] + ':00'))
    del d['id']
    d.sort('dt', inplace = True)
    d['dt'] = d['dt'].apply(lambda tm: tm - timedelta(hours=(24 * tm.day + tm.hour) % hours))
    d.index = pd.MultiIndex.from_tuples(list(zip(d['dt'], d['status'])))
    d = d[['n',]]
    d = d.groupby(d.index).sum()
    d.index = pd.MultiIndex.from_tuples(d.index)
    d = d.unstack(level=[1])
    d.columns = d.columns.droplevel(0)
    d = d * r
    return d


def plot_counts(counts, hours, r , normalize):
    d = get_counts_df(counts,  hours, r)
    
    ylabel = 'count'
    if normalize:
        d = d.div(d.sum(axis=1), axis=0)
        ylabel = 'proportion'

    fig = plt.figure(figsize=(12, 6), dpi=80)
    ax = fig.add_subplot(111)
    d.plot(ax =ax, kind='bar', stacked=True, legend=True, color = sns.color_palette("hls", 8))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel(ylabel)
    #plt.tick_params(labelright=True)
    if normalize:
        ax.set_yticks(np.arange(0, 1.1, 0.1))

def get_campaign_data(sc, campaign, banner_reg, start, stop):
    data = load_rdd(sc, start, stop, hour = True)
    data = transform_datetimes(data)
    data = filter_dt(data, start, stop)
    data = filter_campaign_stop_gap(data, campaign, banner_reg)
    data = transform_reasons(data)
    donor_data = get_donor_data(data)
    sample_data = get_sample_data(data)
    donor_data.persist()
    sample_data.persist()
    return donor_data, sample_data


def get_counts(campaign, banner_reg, start, stop, dry = False):
    cmd = """
    ssh stat1002.eqiad.wmnet "\
    spark-submit \
    --driver-memory 1g --master yarn --deploy-mode client \
    --num-executors 1 --executor-memory 10g --executor-cores 8 \
    --queue priority \
    /home/ellery/wmf/fr/campaign_analysis/src/get_banner_history_data.py \
    --campaign '%s' \
    --banner_reg '%s' \
    --start '%s' \
    --stop  '%s' \
    "
    """
    cmd =  cmd % (campaign, banner_reg, start, stop)
    if not dry:
        ret = os.system(cmd)
        assert(ret == 0)
    os.system("scp stat1002.eqiad.wmnet:./sample_status_counts_%s ." % campaign)
    os.system("scp stat1002.eqiad.wmnet:./sample_impression_counts_%s ." % campaign)
    os.system("scp stat1002.eqiad.wmnet:./donor_impression_counts_%s ." % campaign)

    sample_status_counts = pickle.load(open("sample_status_counts_%s" % campaign, 'rb'))
    sample_impression_counts = pickle.load(open("sample_impression_counts_%s" % campaign, 'rb'))
    donor_impression_counts = pickle.load(open("donor_impression_counts_%s" % campaign, 'rb'))

    return sample_status_counts, sample_impression_counts, donor_impression_counts


def plot_pv_fraction_in_campaign(pv, access_method, counts, sample_rate, hours = 1):
    # get pvs per hours for source
    pv  = copy.copy(pv[pv['access_method'] == access_method])
    pv.index = pd.Series(pv.index).apply(lambda tm: tm - timedelta(hours=(24 * tm.day + tm.hour) % hours))
    pv = pv.groupby(pv.index).sum()

    pv['pageviews_in_campaign'] = get_counts_df(counts, hours, sample_rate).sum(axis=1)
    pv['fraction_of_pageviews_in_campaign'] = pv['pageviews_in_campaign'] / pv['pageviews']
    plot_df(pv[['fraction_of_pageviews_in_campaign']], 'fraction per %d hours' % hours)
    return pv


