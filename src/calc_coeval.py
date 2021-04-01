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
from coeval import coeval

import astropy.units as u

def calc_coeval(star, center,times, rv):
    coeval_array = coeval(star, center, times, rv)
    coe_min_sep_i = np.unravel_index(coeval_array.argmin(),coeval_array.shape)
    return {"rv": rv[coe_min_sep_i[0]].value,
            "lookback_time": -1.0*times[coe_min_sep_i[1]].value,
            "min_sep": coeval_array[coe_min_sep_i].value}
    


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
    cluster_members[cl].from_source_idlist(list(xmatches[cl].dr3_source_id),schema='gaiaedr3', query_type='sync')

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

times = np.linspace(-500e6, -50e6, 1000)*u.year
rv = np.linspace(-100, 100, 1000)*u.km/u.second
coords = search_results['alphaPer'].get_coords()

cl='alphaPer'
source_id = 970348221586771840
coords = search_results[cl].get_coords()
star_i = search_results[cl].objs.index.get_loc(source_id)
print(calc_coeval(coords[star_i], cluster_info.loc[cl]['coords'], times, rv))

for cl in cluster_names:
    #search_results[cl].objs = search_results[cl].objs[:100]
    start = time.time()
    coords = search_results[cl].get_coords()
    coeval_df = pd.DataFrame([calc_coeval(s, cluster_info.loc[cl]['coords'], times, rv) for s in coords],
        index=pd.Index(search_results[cl].objs.index, name = search_results[cl].objs.index.name))
    end = time.time()
    print(f'-----------Cluster: {cl},  {end-start} seconds')
    with open(f'../data/coeval_{cl}.pkl', 'wb') as pkl:
        pickle.dump(coeval_df, pkl)