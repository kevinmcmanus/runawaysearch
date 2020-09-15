from locate_cluser_outliers.src.gaiastars import gaiastars as gs
from locate_cluster_outliers.src.data_quieries import *
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
import astropy.coordinates as coord
from astropy.table import QTable, Table, vstack

import pandas as pd
import numpy as np
import os
#need to calculate color index of stars in cluster based on their magnitudes in certain filters
#this function will get magnitudes of our known cluster members
def getMagsKnownMembers(known_members):
    #get known members
    #known_members = getGaiaKnownMembers(cluster_name) <--I realized I will pass this in 
    #get gaia data for the known members
    known_gaia = gs(cluster_name+': Known Members')
    known_gaia.from_source_idlist(known_members.index.to_list())
    #get magnitudes from gaia for known members for all filters and absolute magnitude
    distmod = 5*np.log10(known_gaia.objs.r_est)-5
    
    abs_mag = known_gaia.objs.phot_g_mean_mag - distmod 
    g_mag = known_gaia.objs.phot_g_mean_mag
    bp_map = known_gaia.objs.phot_bp_mean_mag
    rp_mag = known_gaia.objs.phot_rp_mean_mag
    
    return(g_mag, bp_mag, rp_mag, abs_mag)
#this function will get their absolute magnitudes from their filter magnitudes

#this function will get the color index of known members
def getColorExcessKnownMembers(known_members):
    #get known members
#     known_members = getGaiaKnownMembers(cluster_name) <- passing this in now
    #get gaia data for the known members
    known_gaia = gs(cluster_name+': Known Members')
    known_gaia.from_source_idlist(known_members.index.to_list())
    #get magnitudes from gaia for known members
    distmod = 5*np.log10(known_gaia.objs.r_est)-5
    bp_rp = known_gaia.objs.phot_bp_mean_mag - known_gaia.objs.phot_rp_mean_mag
    #do we want to do a different color index? 
    return(bp_rp)

#this function will convert filter types of gaia magnitudes to different ones because I think this is necessary?
# def convertFilters(mag_to_convert, new_filter):
    #gaia is UGRIZ I think? 
    #Cousins-Johnson?
    #what other fitler types should we include?


"""
    These next 2 functions will have to pass in the candidate members and 
    I need the stars after the proper motion trace back to do this, 
    also will the stars passed in be compatible with gaia_stars?
"""
# #this function will get the color index of candidate members 
# def getColorIndexCandidates(candidate_members):
#     
# #this function will get magnitudes of our candidate members 
# def getMagsCandidateMembers(candidate_members):
