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

from gaiastars import from_pickle

carina_search_results = from_pickle(f'./data/carina_search_results')

from src.perryman import perryman

t15_info = {'ra':161.18, 'ra_error': np.nan,
            'dec':-59.367, 'dec_error': np.nan,
            'parallax': 0.38, 'parallax_error': 0.04,
            'pmra':-6.209, 'pmra_error':0.306,
            'pmdec':2.016, 'pmdec_error':0.216,
            'radial_velocity': -11.6, 'radial_velocity_error':5.8 }
t15_center = pd.Series(t15_info)

t15 = perryman(carina_search_results.objs, init_members=t15_center)

t15.fit(max_dist=50, maxiter=50)

import pickle
with open('./data/t15_fitted.pkl','wb') as pkl:
    pickle.dump(t15, pkl)