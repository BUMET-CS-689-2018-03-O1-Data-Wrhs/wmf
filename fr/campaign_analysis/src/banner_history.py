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
        f = os.path.join(base, str(start.year), str(start.month), str(start.day) )


        if hour:
            if start.hour < 10:
                h = '0' + str(start.hour)
            else:
                h = str(start.hour)
            f = os.path.join(f, h)
            start += relativedelta.relativedelta(hours=1)
        else:
            f = os.path.join(f, '*')
            start += relativedelta.relativedelta(days=1)
        files.append(f)

    return sc.sequenceFile(','.join(files)).map(lambda x: json.loads(x[1])['event'])



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



def get_previous_pageviews_counts(data):
    impressions = data.filter(lambda x: x['l'][-1]['s'] == '6')
    counts = impressions.map(lambda x: (len([1 for e in x['l'] if e['s'] == '6']), 1)).countByKey()
    return counts


def get_previous_impression_counts(data):
    impressions = data.filter(lambda x: x['l'][-1]['s'] == '6')
    counts = impressions.map(lambda x: (len(x['l']), 1)).countByKey()
    return counts
    


def plot_status(statcnts, hours, r , normalize):
    d = pd.DataFrame(list(statcnts.items()))
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
    if normalize:
        d = d.div(d.sum(axis=1), axis=0)

    fig = plt.figure(figsize=(24, 12), dpi=80)
    ax = fig.add_subplot(111)
    d.plot(ax =ax, kind='bar', stacked=True, legend=True)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
