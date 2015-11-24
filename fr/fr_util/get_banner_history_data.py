from pyspark import SparkConf, SparkContext
import pandas as pd
import argparse
import json
from banner_history_utils import get_campaign_data, get_status_counts, get_previous_impression_counts
import pickle


"""
Usage: 
spark-submit \
--driver-memory 1g --master yarn --deploy-mode client \
--num-executors 1 --executor-memory 5g --executor-cores 8 \
--queue priority \
/home/ellery/wmf/fr/campaign_analysis/src/get_banner_history_data.py \
--campaign 'C1516_frFR_dsk_hi_FR' \
--banner_reg 'B1516.*frFR' \
--start '2015-10-22 12:00' \
--stop  '2015-10-23 12:00' 
"""




if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required = True, help='path to recommendation file' )
    parser.add_argument('--banner_reg', required = True, help='path to recommendation file' )
    parser.add_argument('--start', required = True, help='path to recommendation file' )
    parser.add_argument('--stop', required = True, help='path to recommendation file' )

    args = parser.parse_args()
    campaign = args.campaign
    banner_reg = args.banner_reg
    start = args.start
    stop = args.stop
    

    conf = SparkConf()
    conf.set("spark.app.name", 'banner history %s' % args.campaign)
    sc = SparkContext(conf=conf)

    donor_data, sample_data = get_campaign_data(sc, args.campaign, args.banner_reg, args.start, args.stop)
    sample_status_counts = get_status_counts(sample_data)

    def mapping(count):
        if count <=5:
            return '0' + str(count)
        elif count < 11:
            return '06-10'
        elif count < 16:
            return '11-15'
        else:
            return '16+'

    sample_impression_counts = get_previous_impression_counts(sample_data, mapping = mapping, hour = True)
    donor_impression_counts = get_previous_impression_counts(donor_data, mapping = mapping, hour = True)

    pickle.dump( sample_status_counts, open( "sample_status_counts_%s" % campaign, "wb" ) )
    pickle.dump( sample_impression_counts, open( "sample_impression_counts_%s" % campaign, "wb" ) )
    pickle.dump( donor_impression_counts, open( "donor_impression_counts_%s" % campaign, "wb" ) )






    

    

