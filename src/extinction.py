
import numpy as np
import pandas as pd
import pickle

import sys
sys.path.append('./src')

from data_queries import getClusterInfo
from gaiastars import gaiastars as gs, from_pickle

import astropy.units as u
from astropy.coordinates import SkyCoord, Angle

import os
if __name__ == "__main__":

    #1 degree cone search around center of Trumplers
    ra =Angle(161.16701439865315*u.degree)
    dec = Angle(-59.61962492584757*u.degree)
    search_radius = 1*u.degree
    minplx = 0.35
    maxplx = 0.55

    #fix up the query
    #extcolumns = ['a_g_val', 'e_bp_min_rp_val']
    #add table to query to get the ruwe parameter
    #fixme = gs(name='fixme')
    #fixme.add_table_columns(extcolumns, schema='gaiadr2')

    #open up the query
    query_constraints = ['{schema}.gaia_source.a_g_val is not NULL AND {schema}.gaia_source.e_bp_min_rp_val is not NULL']

    #update class def to use new query constraints
    gs.set_gaia_source_constraints(gs,query_constraints)

    fs = gs(name = 'deredden_query')

    fs.conesearch(ra, dec, search_radius, parallax=(minplx, maxplx), schema='gaiadr2', query_type='sync')

    print(fs)
    color_correction= {}
    from sklearn.neighbors import KNeighborsRegressor
    coords = fs.get_coords().galactic
    cart_coords = coords.cartesian.xyz.value.T

    a_g_val = np.array(fs.objs.a_g_val)
    color_correction['de_extinct'] = KNeighborsRegressor(n_neighbors=5)
    color_correction['de_extinct'].fit(cart_coords, a_g_val)

    e_bp_min_rp_val = np.array(fs.objs.e_bp_min_rp_val)
    color_correction['de_reddener'] = KNeighborsRegressor(n_neighbors=5)
    color_correction['de_reddener'].fit(cart_coords, e_bp_min_rp_val )
    
    color_correction['input'] = fs

    with open('./data/carina_color_correction.pkl','wb') as pkl:
        pickle.dump(color_correction, pkl)

    #get some stars:
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

    fs = gs(name = 'Search in vicinity of Carina')
    fs.conesearch(ra, dec, search_radius, parallax=(minplx, maxplx), schema='gaiaedr3', query_type='async')

    print(fs)
    print(fs.tap_query_string)

    print('correcting for reddening and extiction')
    coords = fs.get_coords().galactic
    cart_coords = coords.cartesian.xyz.value.T

    print('Dereddening')
    e_bp_min_rp_val_est = color_correction['de_reddener'].predict(cart_coords)
    fs.objs['e_bp_min_rp_val_est'] = e_bp_min_rp_val_est
    print('De-extincting')
    a_g_val_est = color_correction['de_extinct'].predict(cart_coords)
    fs.objs['a_g_val_est'] = a_g_val_est

    fs.to_pickle('./data/carina_search_results')

    print('All Done! Yay!')
    

