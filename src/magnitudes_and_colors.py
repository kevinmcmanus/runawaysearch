from locate_cluser_outliers.src.gaiastars import gaiastars as gs
from locate_cluster_outliers.src.data_quieries import *
from locate_cluster_outliers.src.calculate_cluster_calues import *
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
import astropy.coordinates as coord
from astropy.table import QTable, Table, vstack

import pandas as pd
import numpy as np
import os
#need to calculate color index of stars in cluster based on their magnitudes in certain filters
#this function will get the known members absolute magnitudes of our known cluster members
def getMagsKnownMembers(known_members):
    '''Returns the color excess of the known_members
     using the distance to the sun the stars are and their gaia filter magnitudes
     ''' 
    #get gaia data for the known members
    known_gaia = gs(cluster_name+': Known Members')
    known_gaia.from_source_idlist(known_members.index.to_list())
    #get magnitudes from gaia for known members for all filters and absolute magnitude
    distmod = 5*np.log10(known_gaia.objs.r_est)-5
    
    abs_mag = known_gaia.objs.phot_g_mean_mag - distmod 
    #green?
    g_mag = known_gaia.objs.phot_g_mean_mag
    #blue?
    bp_map = known_gaia.objs.phot_bp_mean_mag
    #red?
    rp_mag = known_gaia.objs.phot_rp_mean_mag
    
    return(g_mag, bp_mag, rp_mag, abs_mag)


#this function will get the color index of known members
def getColorExcessKnownMembers(known_members):
    '''Returns the color excess of the known_members
     using the distance to the sun the stars are and their gaia filter magnitudes
     '''
    #get gaia data for the known members
    known_gaia = gs(cluster_name+': Known Members')
    known_gaia.from_source_idlist(known_members.index.to_list())
    #get magnitudes from gaia for known members
    distmod = 5*np.log10(known_gaia.objs.r_est)-5
    bp_rp = known_gaia.objs.phot_bp_mean_mag - known_gaia.objs.phot_rp_mean_mag
    #do we want to do a different color index? this is blue minus red I think
    return(bp_rp)


# #this function will get the color index of candidate members, necessary?
def getColorExcessCandidates(candidate_members):
    '''Returns the color excess of the candidate_members
     using the distance to the sun the stars are and their gaia filter magnitudes
    To update: specify how candidate_members are passed in and how it works with gaiastars format
    '''
    dist_mod = 5*np.log10(candidate_members.obj.d)-5
    bp_rp = candidate_members.objs.phot_bp_mean_mag - candidate_members.objs.phot_rp_mean_mag
    return(bp_rp)
    

def getAppMagsCandidateMembers(candidate_members):
    '''Returns the apparent magnitudes of the candidate members 
    using the distance and the magnitude of the members in the gaia red filter.
    To update: specify how candidate_members are passed in and how it works with gaiastars format
    '''
    #based on distance, get apparent mag of candidates
    distanceFromSun = candidate_members.obj.ObjDistCen
    #magnitude in gaia red filter? 
    rp = known_gaia.objs.phot_rp_mean_mag 
    #I am not sure if this is right 
    m = 5*np.log10(distance/10)+rp
    return(m)

#this function will convert filter types of gaia magnitudes to different ones, this may not be necessary.
# def convertFilters(mag_to_convert, new_filter):
    #gaia is UGRIZ I think? 
    #Cousins-Johnson?
    #what other fitler types should we include?