
import numpy as np
import pandas as pd

import sys
sys.path.append('./src')

from data_queries import querySIMBAD, formatSIMBADtoGAIA, getGAIAKnownMembers, getClusterInfo
from gaiastars import gaiastars as gs

import astropy.units as u

import os
if __name__ == "__main__":
    known_cluster_members, cluster_names = getGAIAKnownMembers()
    print(cluster_names)

    schoettler_constraints = [
        '(({schema}.gaia_source.astrometric_excess_noise < 1) or (({schema}.gaia_source.astrometric_excess_noise> 1) and ({schema}.gaia_source.astrometric_excess_noise_sig <2)))',
        '{schema}.gaia_source.ruwe < 1.3'
    ]

    #update class def to use new query constraints
    gs.set_gaia_source_constraints(gs,schoettler_constraints)

    #add table to query to get the ruwe parameter
    default_columns = gs.gaia_column_dict_gaiaedr3.copy() #save for later
    cd = default_columns.copy()
    cd['gaiaedr3.gaia_source'] += ['ruwe']
    gs.gaia_column_dict_gaiaedr3 = cd


    # gaiadr2 to gaiaedr3 mapper
    from  gaiastars import gaiadr2xdr3


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
    USEPICKLE = False
    from gaiastars import from_pickle

    for cl in cluster_names:
        ra_offset =  0*u.degree
        dec_offset = 0*u.degree
        search_radius = 25.0*u.degree

        ra = cluster_info.loc[cl]['coords'].ra + ra_offset
        dec = cluster_info.loc[cl]['coords'].dec + dec_offset

        dist = cluster_info.loc[cl]['coords'].distance.value
        minplx = 1000/(dist*1.25)
        maxplx = 1000/(dist*0.75)

        #grab some stars around the cluster center
        fs = gs(name=f'{cl} cone search', description=f'Conesearch around {cl}')

        fs.conesearch(ra, dec, search_radius, parallax=(minplx, maxplx), schema='gaiaedr3')

        search_results[cl]= fs

        print(fs)
    for cl in cluster_names:
        search_results[cl].to_pickle(f'./data/search_results_{cl}.pkl')
