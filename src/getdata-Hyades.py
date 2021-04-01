
import numpy as np
import pandas as pd

import sys
sys.path.append('./src')

from data_queries import getClusterInfo
from gaiastars import gaiastars as gs, from_pickle

import astropy.units as u

import os
if __name__ == "__main__":

    cluster_info = getClusterInfo()

    #open up the query
    query_constraints = ['{schema}.gaia_source.ruwe < 1.3'] 

    #update class def to use new query constraints
    gs.set_gaia_source_constraints(gs,query_constraints)

    #add table to query to get the ruwe parameter
    default_columns = gs.gaia_column_dict_gaiaedr3.copy() #save for later
    cd = default_columns.copy()
    #ruwe now part of the default columns so no need to add here
    #cd['gaiaedr3.gaia_source'] += ['ruwe']
    gs.gaia_column_dict_gaiaedr3 = cd
	
    #get some stars:
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
    # just deal with Pleiades and alphaPer for now
    cluster_names = ['Hyades', 'Pleiades', 'alphaPer']
    search_results = {}

    for cl in cluster_names:
        ra_offset =  0*u.degree
        dec_offset = 0*u.degree
        search_radius = 25.0*u.degree

        ra = cluster_info.loc[cl]['coords'].ra + ra_offset
        dec = cluster_info.loc[cl]['coords'].dec + dec_offset

        dist = cluster_info.loc[cl]['coords'].distance.value
        minplx = 1000/(dist*1.5)
        maxplx = 1000/(dist*0.5)

        #grab some stars around the cluster center
        fs = gs(name=f'{cl} cone search', description=f'Conesearch around {cl}')

        fs.conesearch(ra, dec, search_radius, parallax=(minplx, maxplx), schema='gaiaedr3', query_type='sync')

        search_results[cl]= fs

        print(fs)
        search_results[cl].to_pickle(f'./data/search_results_{cl}.pkl')
