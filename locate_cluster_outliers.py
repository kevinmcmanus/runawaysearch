#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import necessary packages and files
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

from fieldstars import fieldstars as fs
import astropy.coordinates as coord
from astropy.table import QTable, Table, vstack
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astropy.visualization import quantity_support
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad

import pandas as pd
import numpy as np
import pickle
import os

#user will provide: cluster name

#everything in this file will be contained in an overall function that the user can call with their specific parameters (can also be used to test our functions)

#set up SIMBAD query for reference of known cluster members (should we use the GAIA associated members from that paper instead? or both like Kevin did in PleiadesGMM?)
#calculate cluster center from SIBAD/GAIA known reference stars

#do cone search of GAIA around cluster center (from fieldstars.py)

#CHECK 1 get proper motions of all new stars
#extrapolate pms "backwards" (need to figure this part out still) (will be written in extrapolate_proper_motions.py)
#if the extrapolated pm of a star is within a certain distance from the cluster center, keep it
#else, throw out stars that are no where near the cluster center when extrapolated pm

#CHECK 2 calculate color index of new stars kept (not sure if magnitude needs to be calculated or if it is already in DR2) (will be written within get_color_index.py)
#if color index of a star match the metric we decide, keep it (have to calculate what color is expected from known cluster members) (will be written within get_color_index.py)

#calculate radial velocities of stars that passed both Check 1 & 2 (this may be outside the scope of DR2)

