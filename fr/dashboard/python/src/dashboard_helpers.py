import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from .data_retrieval import get_banner_data, HiveBannerDataRetriever, OldBannerDataRetriever
from datetime import timedelta
from stats_utils import *
import copy


"""
This module contains the Test class
"""
RECURRING_MONTHS = 14



class Test(object):
    def __init__(self, *args, **kwargs):
        self.names = list(set(args))
        self.start = kwargs.get('start', None)
        self.stop = kwargs.get('stop', None)
        self.num_workers = kwargs.get('num_workers', 1)
        self.hive = kwargs.get('hive', False)

        if self.hive:
            self.data = get_banner_data(HiveBannerDataRetriever, self.names, self.start, self.stop)
        else:
            self.data = get_banner_data(OldBannerDataRetriever, self.names, self.start, self.stop)

        for name in self.names:
            self.data[name]['clean_donations'] = self.get_clean_donations(self.data[name]['donations'])


    def get_clean_donations(self, donations):
        clean_donations =  donations[np.abs(donations.amount-donations.amount.mean()) <= (4*donations.amount.std())]
        return clean_donations


    def combine(self, names, combination_name):

        """
        Allows for the logical combination of data from banners
        Data from all banners in names will the combined
        The combination can be accessed by the given combination_name
        """

        if combination_name in self.names:
            print ("The combination_name is already in use")
            return

        if len(set(names).difference(self.names)) != 0:
            print ("One of the banners is not known to the test object")
            return

        #reduce set to be unique
        names = list(set(names))

        # add new data dict
        self.data[combination_name] = {}
        #combine donations
        combined_donations = pd.concat([self.data[name]['donations'] for name in names], axis=0)
        combined_donations = combined_donations.sort()
        self.data[combination_name]['donations'] = combined_donations

        #combine clicks
        combined_clicks = pd.concat([self.data[name]['clicks'] for name in names], axis=0)
        combined_clicks = combined_clicks.sort()
        self.data[combination_name]['clicks'] = combined_clicks

        #combine impressions
        combined_impressions = pd.concat([self.data[name]['impressions'] for name in names], axis=0)
        combined_impressions = combined_impressions.groupby(combined_impressions.index).sum()
        self.data[combination_name]['impressions'] = combined_impressions

        self.data[combination_name]['clean_donations'] = self.get_clean_donations(self.data[combination_name]['donations'])

        self.names.append(combination_name)


    def ecom(self, *args):

        """
        One might beinterested in combining data from different banners into one
        new synthetic banner. Say you ran a test where people who saw banner B1 
        on their first impression,saw banner B3 on their next impressions.
        You might want to combine data from these two banners to do analysis on
        the aggregate data from B1 and B3.
        The combine function takes a list of banners used in the initialization
        of the test object t and combines them under the new name combination_name.
        """

        # set up list of banner to process
        d = {}
        if len(args) == 0:
            names = self.names
        else:
            names = args

        # Step through metrics and compute them for each banner

       

        d['impressions'] = [self.data[name]['impressions']['n'].sum() for name in names]
        d['clicks'] = [self.data[name]['clicks'].shape[0] for name in names]
        d['amount'] = [self.data[name]['donations']['amount'].sum() for name in names]
        d['donations'] = [self.data[name]['donations'].shape[0] for name in names]
        d['recurring donations'] = [self.data[name]['donations']['recurring'].sum() for name in names]
        d['amount (no outliers)'] = [self.data[name]['clean_donations']['amount'].sum() for name in names]
        d['max'] = [self.data[name]['donations']['amount'].max() for name in names]
        d['median'] = [self.data[name]['donations']['amount'].median() for name in names]
        d['avg'] = [self.data[name]['donations']['amount'].mean() for name in names]
        d['avg (no outliers)'] = [self.data[name]['clean_donations']['amount'].mean() for name in names]

        d = pd.DataFrame(d)

    
        d.index = names

        # metrics computed from above metrics
        d['clicks/i'] = d['clicks'] / d['impressions']
        d['dons/i'] = d['donations'] / d['impressions']
        d['amount/i'] = d['amount'] / d['impressions']
        d['amount/i (no outliers)'] = d['amount (no outliers)'] / d['impressions']
        d['dons/clicks'] = d['donations'] / d['clicks']

         # Make numbers display nicely:
        integer_columns = ['impressions',
                            'clicks',
                            'amount',
                            'donations',
                            'amount (no outliers)',
                            'max']

        for c in integer_columns:
            d[c] = d[c].apply(lambda x: "%d" % x)

        precicion_2_columns = ['median',
                                'avg',
                                'avg (no outliers)',
                                ]

        for c in precicion_2_columns:
            d[c] = d[c].apply(lambda x: "%0.2f" % x)

        precicion_5_columns = ['dons/i',
                                'amount/i',
                                'clicks/i',
                                'amount/i (no outliers)',
                                'dons/clicks' ]

        for c in precicion_5_columns:
            d[c] = d[c].apply(lambda x: "%0.5f" % x)


        #Define the metrics in the order requested by Megan
        column_order = [
        'donations',
        'recurring donations',
        'impressions',
        'dons/i',
        'amount',
        'amount/i',
        'clicks',
        'clicks/i',
        'dons/clicks',
        'amount (no outliers)',
        'amount/i (no outliers)',
        'max',
        'median',
        'avg',
        'avg (no outliers)']

        # put hive traffic data into df if available
        #if self.hive:
        #    column_order.insert(1, 'traffic')
        d = d[column_order]

        return d.sort().transpose()


    def get_payment_method_details(self, *args):

        """
        A banner usually gives several payment options for users.
        This function returns a dataframe showing how many people clicked on each payment method, 
        how many successful donations came from each payment method,
        the percent of donations that came from each method,
        the total raised for each method,
        the average raised for reach method, where outliers where removed
        """

        # set up list of banner to process
        if len(args) == 0:
            names = self.names
        else:
            names = args


        ds = []

        #Define the metrics in the order requested by Megan
        column_order = [
        'name',
        'donations',
        'clicks',
        'conversion_rate',
        'percent clicked on',
        'percent donated on',
        'total_amount',
        'ave_amount_ro'
        ]
        # Step through metrics and compute them for each banner

        for name in names:

            clicks = self.data[name]['clicks']['payment_method'].value_counts()
            donations = self.data[name]['donations']['payment_method'].value_counts()
            donations_sum = self.data[name]['donations'].groupby(['payment_method']).apply(lambda x: x.amount.sum())
            ave = self.data[name]['clean_donations'].groupby(['payment_method']).apply(lambda x: x.amount.mean())

            df = pd.concat([donations, clicks, ave, donations_sum], axis=1)
            df.columns = ['donations', 'clicks', 'ave_amount_ro', 'total_amount']

            # metrics computed from above metrics
            df['conversion_rate'] = 100* df['donations'] / df['clicks']
            df['percent clicked on'] = 100*df['clicks'] / df['clicks'].sum()
            df['percent donated on'] = 100*df['donations'] / df['donations'].sum()
            df['name'] = name

            #Put the metrics in the order requested by Megan

            df = df[column_order]
            ds.append(df)


        df = pd.concat(ds)

        df.index = pd.MultiIndex.from_tuples(list(zip(df['name'], df.index)))
        del df['name']
        df = df.sort()


        precicion_2_columns = ['conversion_rate',
                                'percent clicked on',
                                'percent donated on',
                                'ave_amount_ro',
                                ]

        for c in precicion_2_columns:
            df[c] = df[c].apply(lambda x: "%0.2f" % x)

        return df



    def plot_donations_over_time(self, *args, **kwargs):
        # set up list of banner to process
        if len(args) == 0:
            names = self.names
        else:
            names = args

        #process keyword arguments
        window = kwargs.get('smooth', 10)
        start = kwargs.get('start', '2000')
        stop = kwargs.get('stop', '2050')
        amount = kwargs.get('amount', False)

        # helper function to join impression and donation data, very naive implementation
        def get_p_over_time(donations, impressions, window):
            donations['donation'] = 1
            d = donations.groupby(lambda x: (x.year, x.month, x.day, x.hour, x.minute)).sum()
            d.index = d.index.map(lambda t: pd.datetime(*t))
            d2 = impressions.join(d)
            d2 = d2.fillna(0)
            d2['d_window'] = 0
            d2['amount_window'] = 0
            m = d2.shape[0]

            for i in range(m):
                start = max(i-window, 0)
                end = min(i+window, m-1)
                d_window = d2.ix[start:end]
                p_window = float(d_window['donation'].sum())/d_window['count'].sum()
                d2['d_window'].ix[i] = p_window
                u_window = d_window['amount'].sum()/d_window['count'].sum()
                d2['amount_window'].ix[i] = u_window
            return d2

        # iterate over banners and generate plot
        fig = plt.figure(figsize=(10, 6), dpi=80)
        ax = fig.add_subplot(111)
        plt.xticks(rotation=70)
        formatter = DateFormatter('%Y-%m-%d %H:%M')

        plt.gcf().axes[0].xaxis.set_major_formatter(formatter)

        for name in names:
            d = get_p_over_time(self.data[name]['donations'], self.data[name]['impressions'], window)
            d = d[start:stop]
            if amount:
                ax.plot(d.index, d['amount_window'], label=name)
            else:
                ax.plot(d.index, d['d_window'], label=name)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if amount:
            plt.ylabel('amount per impression')
        else:
            plt.ylabel('donations per impression')
        plt.show()



    def plot_impressions(self, *args, **kwargs):

        """
        Plots impressions over the duration of the test
        Allow setting a time range
        And Smoothing by taking an avergae over a window of records
        """
        
        # set up list of banner to process
        if len(args) == 0:
            names = self.names
        else:
            names = args

        #process keyword arguments
        smooth = kwargs.get('smooth', 1)
        start = kwargs.get('start', '2000')
        stop = kwargs.get('stop', '2050')

        fig = plt.figure(figsize=(10, 6), dpi=80)
        ax = fig.add_subplot(111)
        plt.xticks(rotation=70)
        formatter = DateFormatter('%Y-%m-%d %H:%M')

        plt.gcf().axes[0].xaxis.set_major_formatter(formatter)

        for name in names:
            d = self.data[name]['impressions']['n']
            d = d[start:stop]        
            d = pd.rolling_mean(d, smooth)
            ax.plot(d.index, d.values, label=name)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel('impressions')
        plt.show()


    def plot_utm_key(self, *args, **kwargs):

        """
        utm_key is strangely named. It is the number of impressions
        donors saw before donating
        """
        
        # set up list of banner to process
        if len(args) == 0:
            names = self.names
        else:
            names = args

        #process keyword arguments
        max_key = kwargs.get('max_key', 30)
        normalize = kwargs.get('normalize', True)

        fig = plt.figure(figsize=(10, 6), dpi=80)
        ax = fig.add_subplot(111)
        
        for name in names:
            d = pd.DataFrame(self.data[name]['donations']['impressions_seen'].value_counts())
            d = d.sort()
            if normalize:
                d[0] = d[0]/d[0].sum()
            d1 = copy.deepcopy(d[:max_key])
            d1.loc[max_key+1] = d[max_key:].sum()

            ax.plot(d1.index, d1[0], marker='o', label=name)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('impressions seen before donating')
        ticks = [str(i) for i in range(0, max_key+1)]
        ticks.append(str(max_key+1)+"+")
        ax.set_xticks(range(0, max_key+2))

        ax.set_xlim([-0.5, max_key+1.5])
        ax.set_xticklabels(ticks)

        if normalize:
            plt.ylabel('fraction')
        else:
            plt.ylabel('counts')
        plt.show()    


    def compare_donation_amounts(self, a, b):

        """
        This one only operates in 2 banners.
        It gives very nice histogramms of donation amounts
        """
        a_cntr = Counter(np.floor(self.data[a]['donations']['amount']))
        b_cntr = Counter(np.floor(self.data[b]['donations']['amount']))

        print (a_cntr)

        keys = [int(s) for s in set(a_cntr.keys()).union(b_cntr.keys())]
        keys.sort()

        a_values = [a_cntr.get(k, 0) for k in keys]
        b_values = [b_cntr.get(k, 0) for k in keys]


        fig, ax = plt.subplots()
        fig.set_size_inches(15, 6)

        ind = 2.5*np.arange(len(keys))  # the x locations for the groups
        width = 1.2       # the width of the bars

        a_rects = ax.bar(ind, a_values, align='center', facecolor ='yellow', edgecolor='gray', label =a)
        b_rects = ax.bar(ind+width, b_values, align='center', facecolor ='blue', edgecolor='gray', label =b)

        ax.set_xticks(ind+width/2)
        ax.set_xticklabels(keys)
        ax.legend()


        def autolabel(rects):
            # attach some text labels
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                        ha='center', va='bottom')

        autolabel(a_rects)
        autolabel(b_rects)

        plt.show()



########### STATS FUNCTIONS #########
    def samples_per_banner(self, rate, mde=0.05, alpha=0.05, power=0.95):
        return int(samples_per_branch_calculator(rate, mde=mde, alpha=alpha, power=power))


    


    def classic_amount_stats(self, a, b, conf=95, rate='donations/impressions', remove_outliers=True, recurring = False):

        """
        Gives a confidence for difference in the dollars per 1000 impressions between banners a, b 

        a: string name of the A banner
        b: string name of the B banner
        conf: confidence level in [0, 100] for the confidence intervals.
        rate: there are two kinds of rates this function can handle:
            'donations/impressions': 
            'donations/clicks': donations per click
        remove_outliers: remove donations exceeding 3 standard deviations from the mean
        """

        t = rate.split('/')

        data = copy.deepcopy(self.data)

        # use donation data with outliers removed by defualt
        if remove_outliers:
            donations = 'clean_donations'
        else:
            donations = 'donations'
           
        a_event_values = data[a][donations]
        b_event_values = data[b][donations]

        # if recurring, add the expected life_time amount from a recurring donors
        if recurring:
            a_event_values = model_recurring2(a_event_values)
            b_event_values = model_recurring2(b_event_values)

            

        a_event_values = a_event_values['amount']
        b_event_values = b_event_values['amount']
            

        trial_type = t[1]

        

        if trial_type == 'clicks':
            a_num_trials = self.data[a]['clicks'].shape[0]
            b_num_trials = self.data[b]['clicks'].shape[0]
            amount_ci = difference_in_means_confidence_interval(a_event_values, a_num_trials, b_event_values, b_num_trials, alpha=(100 - conf)/200.0)
            print ("%s gives between $%0.4f and $%0.4f more $/clicks than %s" %(a, amount_ci[0], amount_ci[1], b))

        elif trial_type == 'impressions':
            a_num_trials = self.data[a]['impressions'].sum()
            b_num_trials = self.data[b]['impressions'].sum()
            amount_ci = difference_in_means_confidence_interval(a_event_values, a_num_trials, b_event_values, b_num_trials, alpha=(100 - conf)/200.0)
            print ("%s gives between $%0.4f and $%0.4f more $/1000 impressions than %s" %(a, 1000*amount_ci[0], 1000*amount_ci[1], b))

        else:
            print ("incorrect test argument")
            return



    def amount_stats(self, values, conf=95, rate='donations/impressions', plot = True):

        """
        Gives a confidence for difference in the dollars per 1000 impressions between banners a, b 

        values: a dictionary mapping from banner names to a cut-off point for most frequent donation amounts for that banner
        conf: confidence level in [0, 100] for the confidence intervals.
        rate: there are two kinds of rates this function can handle:
            'donations/impressions': 
            'donations/clicks': donations per click
        remove_outliers: remove donations exceeding 3 standard deviations from the mean
        """

        t = rate.split('/')
        trial_type = t[1]


        d = {}
        for name in values:
            num_donations = self.data[name]['donations']['amount'].shape[0]
            counts = self.data[name]['donations']['amount'].value_counts()
            counts.order()
            counts = counts.iloc[:values[name]]

            print ('Values for banner ', name, ':', list(counts.index))


            if trial_type == 'clicks':
                num_0s = int(self.data[name]['clicks'].shape[0]) - num_donations
            elif trial_type == 'impressions':
                num_0s = int(self.data[name]['impressions'].sum()) - num_donations
            else:
                print ("incorrect test argument")
                return

            counts =  counts.set_value(0.0, num_0s)
            d[name] =  get_multinomial_expectation_dist(counts)
        return print_stats(pd.DataFrame.from_dict(d), conf, plot)


    
    def rate_stats(self, *args, ** kwargs):


        """
        usage: t.rate_stats(B1, B2, ...BN, conf = 95, rate = 'donations/impressions', plot = True)

        Args:
            Bi: string name of the ith banner
            conf: confidence level in [0, 100] for the confidence intervals.
            rate: there are three kinds of rates this function can handle:
                ''donations/impressions'': donations per impression
                'clicks/impressions': clicks per impression
                'donations/clicks': donations per click
        plot: whether to plot the distributions over the CTRs


        This function computes:
        P(Bi is Best): probability that banner Bi gives more donations per impression than all other banners

        Winers Lift: a 'conf' percent confidence interval on the percent lift in rate the winning banenr  has over the others
        CI: a 'conf' percent confidence interval for the rate of Bi
        

        """
        conf = kwargs.get('conf', 95)
        rate = kwargs.get('rate', 'donations/impressions')
        plot = kwargs.get('plot', True)

        if len(args) == 0:
            names = self.names
        else:
            names = args

        if rate == 'donations/impressions':
            d = {}
            for name in names:
                num_heads = int(self.data[name]['donations'].shape[0])
                num_tails = int(self.data[name]['impressions'].sum() - num_heads)
                counts = pd.Series([num_tails , num_heads ], index = [0.0, 1.0])
                d[name] = get_multinomial_expectation_dist(counts)
            return print_stats(pd.DataFrame.from_dict(d), conf, plot)

        elif rate == 'clicks/impressions':
            d = {}
            for name in names:
                num_heads = int(self.data[name]['clicks'].shape[0])
                num_tails = int(self.data[name]['impressions'].sum() - num_heads)
                counts = pd.Series([num_tails , num_heads ], index = [0.0, 1.0])
                d[name] = get_multinomial_expectation_dist(counts)
            return print_stats(pd.DataFrame.from_dict(d), conf, plot)

            
        elif rate == 'donations/clicks':

            d = {}
            for name in names:
                num_heads = int(self.data[name]['donations'].shape[0])
                num_tails = int(self.data[name]['clicks'].shape[0] - num_heads)
                counts = pd.Series([num_tails , num_heads ], index = [0.0, 1.0])
                d[name] = get_multinomial_expectation_dist(counts)
            return print_stats(pd.DataFrame.from_dict(d), conf, plot)



def print_stats(dists, conf, plot):

    """
    Helper function to create a pandas datframe with rate statistics
    """

    if plot:
        plot_dist(dists)
    result_df = pd.DataFrame()

    def f(d):
        rci = bayesian_ci(d, conf)
        return "(%0.6f, %0.6f)" % (rci[0], rci[1])

    result_df['CI'] = dists.apply(f)

    def f(d):
        return d.idxmax()
    best = dists.apply(f, axis=1)
    result_df['P(Winner)'] = best.value_counts() / best.shape[0]
    result_df = result_df.sort('P(Winner)', ascending=False)

    def f(d):
        ref_d = dists[result_df.index[0]]
        lift_ci = bayesian_ci(100.0 * ((ref_d - d) / d), conf)
        return "(%0.2f%%, %0.2f%%)" % (lift_ci[0], lift_ci[1])

    result_df['Winners Lift'] = dists.apply(f)

    return result_df[['P(Winner)', 'Winners Lift', 'CI']]
    


def plot_dist(dists):
    """
    Helper function to plot the probability distribution over
    the donation rates (bayesian formalism)
    """
    fig, ax = plt.subplots(1, 1, figsize=(13, 3))

    bins = 50
    for name in dists.columns:
        ax.hist(dists[name], bins=bins, alpha=0.6, label=name, normed=True)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    



def custom_amount_stats(a_event_values, a_num_trials, b_event_values, b_num_trials, conf =95):
    amount_ci = difference_in_means_confidence_interval(a_event_values, a_num_trials, b_event_values, b_num_trials, alpha = (100 - conf)/200.0)
    print ("A gives between $%0.4f and $%0.4f more $/clicks than B" %(amount_ci[0], amount_ci[1]))


def custom_rate_stats(a_num_events, a_num_trials, b_num_events, b_num_trials, conf=95, plot =True):
    a_dist = get_beta_dist(a_num_events, a_num_trials)
    b_dist = get_beta_dist(b_num_events, b_num_trials)
    d = pd.DataFrame.from_dict({'A':a_dist, 'B':b_dist})
    return print_rate_stats(d, conf, plot)


def model_recurring1(donation_df):
    reccurring_df = copy.deepcopy(donation_df[donation_df['recurring'] == 1])
    for i in range(0, RECURRING_MONTHS -1):
        donation_df = donation_df.append(copy.copy(reccurring_df))
    return donation_df

def model_recurring2(donation_df):
    donation_df = copy.deepcopy(donation_df)
    donation_df['amount'] = donation_df['amount'] + donation_df['amount'] * donation_df['recurring'] * (RECURRING_MONTHS-1)
    return donation_df
