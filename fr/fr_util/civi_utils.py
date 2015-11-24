from db_utils import query_lutetium, get_time_limits
import pandas as pd
from datetime import timedelta
import copy

def to_int(x):
    try:
        return int(x)
    except:
        return -1

clicks_field_mapping = {
    'campaign': 'utm_campaign',
    'banner': 'banner',
    'name': 'CONCAT_WS(\' \', banner, utm_campaign)',
    'timestamp': 'DATE_FORMAT(CAST(ts as datetime), %(time_format)s)',
    'country': 'country',
    'impressions_seen': 'CAST(ct.utm_key as int)',
    'payment_method': 'utm_source',
    'id': 'ct.id',
    'amount': 'co.total_amount',
    'recurring' : 'co.contribution_recur_id IS NOT NULL',
    'donation' : 'ct.contribution_id is NOT NULL',
    'dollars' : 'SUM(co.total_amount)',
    'n': 'COUNT(*)',
}


def craft_select(fields, field_mapping):
    return "SELECT\n" + ',\n'.join(['\t' + field_mapping[f] + ' AS ' + f for f in fields])

def craft_groupby(fields, field_mapping):
    return 'GROUP BY\n' + ',\n'.join([ '\t' + field_mapping[f] for f in fields])

def get_time_format(aggregation):
    if aggregation == 'hour':
        return '%Y-%m-%d %H:00:00'
    elif aggregation == 'minute':
        return '%Y-%m-%d %H:%i:00'
    elif aggregation == 'none':
        return '%Y-%m-%d %H:%i:%s'
    else:
        print ("invalid aggregation parameter")


def cast_fields(d):
    d.index = pd.to_datetime(d['timestamp'])
    del d['timestamp']

    if 'impressions_seen' in d.columns:
        d['impressions_seen'].apply(to_int)
    if 'payment_method' in d.columns:
        d['payment_method'] = d['payment_method'].apply(lambda x: x.split('.')[2])
    if 'n' in d.columns:
        d['n'] = d['n'].astype(int)
    if 'amount' in d.columns:
        d['amount'] = d['amount'].astype(float)
    if 'dollars' in d.columns:
        d['dollars'] = d['dollars'].astype(float)
    return d


def get_clicks_helper(start, stop, campaign_reg = '.*', banner_reg = '.*', aggregation = 'hour', select_fields = [], groupby_fields = []):

    """
    Gets all donation data within the time range start:stop
    """

    select_fields = set(select_fields)
    select_fields.add('timestamp')
    select_fields.add('n')

    groupby_fields = set(groupby_fields)
    groupby_fields.add('timestamp')


    if aggregation == 'none':
        select_fields.add('id')
        groupby_fields.add('id')


    params = get_time_limits(start, stop)
    params['campaign_reg'] = campaign_reg
    params['banner_reg'] = banner_reg
    params['time_format'] = get_time_format(aggregation)

    where_clause = """
    FROM drupal.contribution_tracking ct LEFT JOIN drupal.contribution_source cs
        ON (ct.id = cs.contribution_tracking_id)
        LEFT JOIN civicrm.civicrm_contribution co 
        ON (co.id = ct.contribution_id)
        WHERE ts BETWEEN %(start_ts)s AND %(stop_ts)s
        AND utm_medium = 'sitenotice'
        AND utm_campaign RLIKE %(campaign_reg)s
        AND banner RLIKE %(banner_reg)s
    """

    select_clause =  craft_select(select_fields, clicks_field_mapping)
    groupby_clause = craft_groupby(groupby_fields, clicks_field_mapping)
    query = select_clause + where_clause + groupby_clause
    
    d = query_lutetium(query, params)
    
    return cast_fields(d)


def get_clicks(start, stop, campaign_reg = '.*', banner_reg = '.*', aggregation = 'hour', fields = []):

    """
    Gets all donation data within the time range start:stop
    """

    d = get_clicks_helper(  start, \
                     stop, \
                    campaign_reg = campaign_reg, \
                    banner_reg = banner_reg, \
                    aggregation = aggregation, \
                    select_fields = copy.copy(fields), \
                    groupby_fields = copy.copy(fields), \
                    )
    return d



def get_donations(start, stop, campaign_reg = '.*', banner_reg = '.*', aggregation = 'hour', fields = [] ):

    """
    Gets all donation data within the time range start:stop
    """

    select_fields = copy.copy(fields)
    select_fields.append('donation')
    select_fields.append('dollars')

    groupby_fields = copy.copy(fields)
    groupby_fields.append('donation')

    d = get_clicks_helper(  start, \
                     stop, \
                    campaign_reg = campaign_reg, \
                    banner_reg = banner_reg, \
                    aggregation = aggregation, \
                    select_fields = select_fields, \
                    groupby_fields = groupby_fields, \
                    )
    d = d[d['donation'] == 1]
    del d['donation']
    return d



def get_impressions(start, stop, campaign_reg = '.*',  banner_reg = '.*', aggregation = 'hour'):

    """
    Gets all impression data within the time range start:stop
    """
    params = get_time_limits(start, stop)
    params['campaign_reg'] = campaign_reg
    params['banner_reg'] = banner_reg
    params['time_format'] = get_time_format(aggregation)


    query = """
    SELECT
        DATE_FORMAT(CAST(timestamp as datetime), %(time_format)s) AS timestamp,
        banner,
        campaign,
        CONCAT_WS(' ', banner, campaign) AS name,
        iso_code AS country, 
        SUM(count) AS n
    FROM pgehres.bannerimpressions imp JOIN pgehres.country c
        WHERE imp.country_id = c.id
        AND timestamp BETWEEN %(start)s AND %(stop)s 
        AND campaign RLIKE %(campaign_reg)s
        AND banner RLIKE %(banner_reg)s
    GROUP BY
        DATE_FORMAT(CAST(timestamp as datetime), %(time_format)s),
        banner,
        campaign,
        iso_code
    """
    
    d = query_lutetium(query, params)
    d.index = pd.to_datetime(d['timestamp'])
    del d['timestamp']
    d['n'] = d['n'].astype(int)
    
    return d


def get_pageviews(start, stop, country, project):

    """
    Get hourly pageview counts fro project from country
    """
    
    query = """
    SELECT
        year,
        month,
        day,
        hour,
        access_method,
        SUM(view_count) as pageviews,
    FROM wmf.projectview_hourly
        WHERE agent_type = 'user'
        AND %(time)s
        AND project = '%(project)s'
        AND country_code = '%(country)s'
    GROUP BY
        year,
        month,
        day,
        hour,
        access_method
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