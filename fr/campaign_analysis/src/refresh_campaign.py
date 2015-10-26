import argparse
import os, sys

"""
python /Users/ellerywulczyn/wmf/fr/campaign_analysis/refresh_campaign.py -c C1516_enJP
"""

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-c', required = True, help='name of ipynb you want to refresh' )
    args = parser.parse_args()

    ipynb_path = '/Users/ellerywulczyn/wmf/fr/campaign_analysis'
    drive_path = '/Users/ellerywulczyn/Google\ Drive/campaigns'
    nb = os.path.join(ipynb_path, args.c + '.ipynb')
    html = os.path.join(drive_path, args.c + '.html')
    run_nb_cmd = 'ipython nbconvert --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=3600 %s'
    os.system(run_nb_cmd % nb)
    os.system('mv ' + args.c + '.html ' + html)


