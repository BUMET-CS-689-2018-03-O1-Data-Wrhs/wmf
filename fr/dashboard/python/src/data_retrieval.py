import os, sys
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
import multiprocessing as mp
import time
import re
from collections import defaultdict
import sys, traceback
from abc import ABCMeta, abstractmethod
from db_utils import *
import hashlib
import copy
import traceback




def query_lutetium_robust(query, params):
    # if the client does not have mysqldb, this wont work
    try:
        return query_lutetium(query, params)
    # so use ssh to query. This is not thread safe and absolutely rediculous
    # 
    except:
        print ("fetching data via ssh")
        ssh_params = copy.copy(params)

        #for k, v in ssh_params.items():
        #    if isinstance(v, basestring):
        #        ssh_params[k] = "'" + v + "'"
        query = query % ssh_params
        file_name = str(hashlib.md5(query.encode()).hexdigest())
        return query_lutetium_ssh(query, file_name);


def get_banner_data(retrieverclass, banner_names,  start, stop):
    d = {}
    for banner in banner_names:
        d[banner] = retrieverclass(banner, start, stop).get_all()
    return d



def get_banner_parrallel(retrieverclass, banner_names, start, stop, num_wokers = 6):

        def pool_wrapper(retriever):
            return (retriever.banner, retriever.get_all())

        p = mp.Pool(num_wokers)
        arguments = []
        for banner_name in banner_names:
            retriver_object = retrieverclass(banner_name, start, stop)
            arguments.append(retriver_object)
            
        results = p.map(pool_wrapper, arguments)

        return dict(results)



class BannerDataRetriever(object):
    __metaclass__ = ABCMeta
    def __init__(self, banner, start, stop):
        self.banner = banner
        self.params = self.get_param_dict(banner, start, stop)


    @abstractmethod
    def get_impressions(self):
        pass

    def get_clicks(self,):
        
        query = """
        SELECT
        ct.ts as timestamp, 
        CAST(ct.utm_key as int) as impressions_seen, 
        ct.utm_source 
        FROM drupal.contribution_tracking ct
        WHERE SUBSTRING_INDEX(ct.utm_source, '.', 1) = %(banner)s
        AND ct.ts BETWEEN %(start_ts)s AND %(stop_ts)s
        AND ct.utm_medium = 'sitenotice'
        ORDER BY ct.ts; 
        """

        d = query_lutetium_robust(query, self.params)
        d.index  = d['timestamp'].map(lambda t: pd.to_datetime(str(t)))
        del d['timestamp']
        d['impressions_seen'] = d['impressions_seen'].fillna(-1)
        d['impressions_seen'] = d['impressions_seen'].astype(int)
        try:
            #d['utm_source'] = d['utm_source'].apply(lambda x: x.decode('utf-8'))
            d['payment_method'] = d['utm_source'].apply(lambda x: x.split('.')[2])
        except:
            print(traceback.format_exc())
            print("error decoding payment method")


        return d


    def get_donations(self):
        
        query = """
        SELECT
        co.total_amount AS amount, 
        ct.ts as timestamp, 
        CAST(ct.utm_key as int) AS impressions_seen, 
        ct.utm_source,
        co.contribution_recur_id IS NOT NULL as recurring
        FROM civicrm.civicrm_contribution co, drupal.contribution_tracking ct
        WHERE co.id = ct.contribution_id
        AND ts BETWEEN %(start_ts)s AND %(stop_ts)s
        AND SUBSTRING_INDEX(ct.utm_source, '.', 1) = %(banner)s
        AND ct.utm_medium = 'sitenotice'
        order by ct.ts;
        """

        d = query_lutetium_robust(query, self.params)
        d.index = d['timestamp'].map(lambda t: pd.to_datetime(str(t)))
        del d['timestamp']
        d['amount'] = d['amount'].fillna(0.0)
        d['amount'] = d['amount'].astype(float)
        d['impressions_seen'] = d['impressions_seen'].fillna(-1)
        d['impressions_seen'] = d['impressions_seen'].astype(int)
        try:
            #d['utm_source'] = d['utm_source'].apply(lambda x: x.decode('utf-8'))
            d['payment_method'] = d['utm_source'].apply(lambda x: x.split('.')[2])
        except:
            print("error decoding payment method")

        return d 

    def get_all(self):
        d = {}
        d['clicks'] = self.get_clicks()
        d['donations'] = self.get_donations()
        d['impressions'] = self.get_impressions()

        return d


    def get_param_dict(self, banner, start, stop):
        params = get_time_limits(start, stop)
        params['banner'] = banner
        return params






#### Child TestData Retrieval Classes ####

class HiveBannerDataRetriever(BannerDataRetriever):

    def get_impressions(self):
        params = self.params.copy()
        params['start'] = params['start'].replace(' ', 'T')
        params['stop'] = params['stop'].replace(' ', 'T')
        query = """
        SELECT 
        sum(n) as count, minute as timestamp, result, reason, spider
        FROM ellery.oozie_impressions_v0_1 
        WHERE banner = '%(banner)s'
        AND minute BETWEEN '%(start)s' AND '%(stop)s' 
        AND year BETWEEN %(start_year)s AND %(stop_year)s \n
        """
        
        if params['start_year'] == params['stop_year']:
            query += "AND month BETWEEN %(start_month)s AND %(stop_month)s \n"
            if params['start_month'] == params['stop_month']:
                query += "AND day BETWEEN %(start_day)s AND %(stop_day)s \n"


        query += "GROUP BY minute, result, reason, spider;"
        query = query % params

        d = query_hive_ssh(query, 'hive_impressions_'+self.banner+".tsv")
        d.index = pd.to_datetime(d['timestamp'])
        d = d.sort()
        del d['timestamp']
        d['count'] = d['count'].fillna(0)
        d['count'] = d['count'].astype(int)
        d = d.fillna('na')
        return d


    def get_all(self):
        d = super(HiveBannerDataRetriever,self).get_all()
        dtemp = d['impressions']
        d['impressions'] = dtemp[(dtemp.result == 'show') & (dtemp.spider == False) ][['count']]
        d['traffic'] = dtemp
        return d

    

class OldBannerDataRetriever(BannerDataRetriever):

    def get_impressions(self):

        query = """
        SELECT SUM(count) as count, timestamp 
        FROM pgehres.bannerimpressions 
        WHERE banner = %(banner)s 
        AND timestamp BETWEEN %(start)s AND %(stop)s 
        GROUP BY timestamp 
        ORDER BY timestamp;
        """

        d = query_lutetium_robust(query, self.params)

        if d.shape[0] == 0:
            return pd.DataFrame(columns = ['count', 'timestamp'])

        d.index = pd.to_datetime(d['timestamp'])
        del d['timestamp']
        d['count'] = d['count'].fillna(0)
        d['count'] = d['count'].astype(int)

        return d

