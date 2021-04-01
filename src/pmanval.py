import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline

from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('./src')

from data_queries import  getClusterInfo, getGAIAKnownMembers
from coeval import coeval
from gaiastars import gaiastars as gs,gaiadr2xdr3

import astropy.units as u
from astropy.coordinates import SkyCoord

known_cluster_members, cluster_names = getGAIAKnownMembers()
print(cluster_names)

# gaiadr2 to gaiaedr3 mapper
from  gaiastars import gaiadr2xdr3

errorcolumns = [
    'ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error','dr2_radial_velocity_error',
    'ra_dec_corr', 'ra_parallax_corr','ra_pmra_corr', 'ra_pmdec_corr',
    'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
    'parallax_pmra_corr', 'parallax_pmdec_corr',
    'pmra_pmdec_corr'
]
#add table to query to get the ruwe parameter
fixme = gs(name='fixme')
fixme.add_table_columns(errorcolumns)

# just deal with Hyades and alphaPer for now
cluster_names = [ 'Hyades']
xmatches = {}
cluster_members={}
#for cl in cluster_names:
for cl in cluster_names:
    known_members_dr2 = list(known_cluster_members.query('Cluster == @cl').index)
    xmatches[cl] = gaiadr2xdr3(known_members_dr2)
    cluster_members[cl]  = gs(name = cl, description=f'{cl} sources from Table 1a records from Gaia archive')
    cluster_members[cl].from_source_idlist(list(xmatches[cl].dr3_source_id),schema='gaiaedr3', query_type='sync')

print(cluster_members['Hyades'])

cluster_info = getClusterInfo()
cluster_names = ['Hyades']
search_results = {}

from gaiastars import from_pickle

for cl in cluster_names:
    search_results[cl] = from_pickle(f'./data/search_results_{cl}.pkl')

print(search_results[cl])

from perryman import perryman

c = cluster_info.loc['Hyades']['coords']

#pick 5 random known members to initialize
km = set(cluster_members['Hyades'].objs.index)
sr = set(search_results['Hyades'].objs.index)
common = [id for id in km.intersection(sr)]
n = len(common)
print(f'Number of common elements: {n}')
els = np.random.choice(n,5, replace=False)
elements = [common[el] for el in els]
print(f'Initial members: {elements}')

hyades = perryman(search_results['Hyades'].objs,
                  init_members=elements)
hyades.fit()