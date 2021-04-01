
import numpy as np
import pandas as pd

import sys
sys.path.append('./src')

from data_queries import getClusterInfo
from gaiastars import gaiastars as gs, from_pickle

import astropy.units as u
from astropy.coordinates import SkyCoord

import os
if __name__ == "__main__":
    errorcolumns = [
        'ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error','dr2_radial_velocity_error',
        'ra_dec_corr', 'ra_parallax_corr','ra_pmra_corr', 'ra_pmdec_corr',
        'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
        'parallax_pmra_corr', 'parallax_pmdec_corr',
        'pmra_pmdec_corr'
    ]

    #open up the query
    query_constraints = ['{schema}.gaia_source.ruwe < 1.3'] 

    #update class def to use new query constraints
    gs.set_gaia_source_constraints(gs,query_constraints)

    #add table to query to get the ruwe parameter
    fixme = gs(name='fixme')
    fixme.add_table_columns(errorcolumns)

    #trumpler meta data
    trumpler_df = pd.DataFrame([
        ['Trumpler14', '10:43:55.4','-59:32:16', 2.37,0.15, 264],
        ['Trumpler15', '10:44:40.8', '-59:22:10', 2.36, 0.09, 320,],
        ['Trumpler16', '10:45:10.6', '-59:42:28', 2.32,0.12, 320]
    ], columns=['ClusterName','RA', 'Dec','Distance','DistErr','Radius']
    )
    trumpler_coords = SkyCoord(ra=trumpler_df.RA, dec=trumpler_df.Dec,
        unit=(u.hourangle, u.deg),
        distance = list(trumpler_df.Distance)*u.kpc)

    cluster_names =list(trumpler_df.ClusterName)
    search_results = {}
    for cli, cl in enumerate(cluster_names):
        ra_offset =  0*u.degree
        dec_offset = 0*u.degree
        search_radius = 1.0*u.degree

        ra = trumpler_coords[cli].ra + ra_offset
        dec = trumpler_coords[cli].dec + dec_offset

        dist = trumpler_coords[cli].distance.to_value(u.pc)
        #how many pc at that distance is the search radius?
        pc = np.tan(search_radius)*dist
        minplx = 1000/(dist+pc)
        maxplx = 1000/(dist-pc)

        print(f'Minplx: {minplx}, Maxplx: {maxplx}')

        #grab some stars around the cluster center
        fs = gs(name=f'{cl} cone search', description=f'Conesearch around {cl}')

        fs.conesearch(ra, dec, search_radius, parallax=(minplx, maxplx), schema='gaiaedr3', query_type='sync')

        search_results[cl]= fs

        print(fs)
        search_results[cl].to_pickle(f'./data/search_results_{cl}.pkl')
