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
from civi_utils import get_clicks, get_donations, get_impressions




def query_lutetium_robust(query, params):
    # if the client does not have mysqldb, this wont work
    try:
        return query_lutetium(query, params)
    # so use ssh to query. This is not thread safe and absolutely rediculous
    # 
    except:
        print ("fetching data via ssh")
        ssh_params = copy.copy(params)
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

        fields = ['payment_method']

        return get_clicks(self.params['start'], \
                          self.params['stop'],  \
                          banner_reg = self.params['banner'], \
                          aggregation = 'none', \
                          fields = fields, \
                          )


    def get_donations(self,):

        fields = ['recurring', 'impressions_seen', 'payment_method', 'amount']

        return get_donations(self.params['start'], \
                          self.params['stop'],  \
                          banner_reg = self.params['banner'], \
                          aggregation = 'none', \
                          fields = fields, \
                          )
        
    
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
        params['time_conditions'] = get_hive_timespan(params['start'], params['stop'])
        params['start'] = params['start'].replace(' ', 'T')
        params['stop'] = params['stop'].replace(' ', 'T')

        query = """
        SELECT 
        n, minute as timestamp
        FROM ellery.oozie_impressions_v0_2 
        WHERE banner = '%(banner)s'
        AND minute BETWEEN '%(start)s' AND '%(stop)s' 
        AND year BETWEEN %(start_year)s AND %(stop_year)s 
        AND %(time_conditions)s
        """
        
        query = query % params

        d = query_hive_ssh(query, 'hive_impressions_'+self.banner+".tsv")
        d.index = pd.to_datetime(d['timestamp'])
        d = d.sort()
        del d['timestamp']
        d['n'] = d['n'].fillna(0)
        d['n'] = d['n'].astype(int)
        d = d.fillna('na')
        return d



class OldBannerDataRetriever(BannerDataRetriever):

    def get_impressions(self):

        d =  get_impressions(self.params['start'], \
                          self.params['stop'],  \
                          banner_reg = self.params['banner'], \
                          aggregation = 'none', \
                          )

        # get_impressions does not allow picking fields yet, so need to group by banner and time
        d = d[['banner', 'n']].groupby(d.index).sum()
        return d

