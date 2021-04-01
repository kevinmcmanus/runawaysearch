import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline

from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('./src')

from data_queries import  getClusterInfo, getGAIAKnownMembers
from gaiastars import gaiastars as gs
from  gaiastars import gaiadr2xdr3

import astropy.units as u

known_cluster_members, cluster_names = getGAIAKnownMembers()
print(cluster_names)

# just deal with Pleiades and alphaPer for now
cluster_names = ['Pleiades', 'alphaPer']
xmatches = {}
cluster_members={}
#for cl in cluster_names:
for cl in cluster_names:
    known_members_dr2 = list(known_cluster_members.query('Cluster == @cl').index)
    xmatches[cl] = gaiadr2xdr3(known_members_dr2)
    cluster_members[cl]  = gs(name = cl, description=f'{cl} sources from Table 1a records from Gaia archive')
    cluster_members[cl].from_source_idlist(list(xmatches[cl].dr3_source_id),schema='gaiaedr3', query_type='async')

#construct a dict mapping cluster name in Table1a to its name in Simabad
cluster_info = getClusterInfo()

cluster_names = ['Pleiades', 'alphaPer']
search_results = {}

from gaiastars import from_pickle
import time
import pickle

for cl in cluster_names:
    search_results[cl] = from_pickle(f'../data/search_results_{cl}.pkl')

#exclude the known members returned from the search
for cl in cluster_names:
    merged_fs = search_results[cl].merge(cluster_members[cl])
    print(f'------ {cl} -------')
    print(merged_fs.objs.which.value_counts())
    fs = merged_fs.query('which == \'{} cone search\''.format(cl))
    fs.name = 'Search Results, Known Members excluded'
    search_results[cl] = fs

for cl in cluster_names:
    #search_results[cl].objs = search_results[cl].objs[:1000]
    start = time.time()
    travel_time_df = search_results[cl].travel_time3d(cluster_info.loc[cl]['coords'],
            (0.8*10**cluster_info.loc[cl]['log_age']*u.year,
            1.2*10**cluster_info.loc[cl]['log_age']*u.year))
    end = time.time()
    print(f'Cluster: {cl}, Candidates: {travel_time_df.tt_3d_candidate.sum()} in {end-start} seconds')
    #print(travel_time_df)
    with open(f'../data/travel_time_{cl}.pkl', 'wb') as pkl:
        pickle.dump(travel_time_df, pkl)
