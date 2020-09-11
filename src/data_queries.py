from locate_cluser_outliers.src.gaiastars import gaiastars as gs
from astroquery.simbad import Simbad
from astropy.time import Time

import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astroquery.gaia import Gaia
import astropy.coordinates as coord
from astropy.table import QTable, Table, vstack

import pandas as pd
import numpy as np
import pickle
import os

#this is where functions for searching SIMBAD, GAIA known members, and SDSS or other catalogs
#clusters = ["Pleiades"]

#function for querying simbad with given cluster name
def querySIMBAD(cluster_name):
    #set up simbad search
    sim = Simbad()
    sim.add_votable_fields('parallax', 'pm','velocity','typed_id')
    #make table of data queried from given cluster name
    sim_table = vstack([sim.query_object(cluster_name)],join_type = 'exact')
    #turn into usable table
    clutser_data = Table(sim_table['TYPED_ID', 'PLX_VALUE', 'PLX_PREC','RA', 'RA_PREC', 'DEC', 'DEC_PREC', 
                                   'PMRA', 'PMDEC', 'RVZ_RADVEL', 'RVZ_ERROR'])
    return(cluster_data)

#the simbad table has different collumn names and coordinates than GAIA, 
#this function converts the SIMBAD data table into the GAIA format 
def formatSIMBADtoGAIA(clutser_name, simbad_table):
    
    simbad_table.rename_column('TYPED_ID','cluster')
    simbad_table.rename_column('PLX_VALUE','parallax')
    simbad_table.rename_column('PLX_PREC', 'parallax_error')
    simbad_table.rename_column('RA', 'ra')
    simbad_table.rename_column('RA_PREC','ra_error')
    simbad_table.rename_column('DEC', 'dec')
    simbad_table.rename_column('DEC_PREC','dec_error')
    simbad_table.rename_column('PMRA', 'pmra')
    simbad_table.rename_column('PMDEC', 'pmdec')
    simbad_table.rename_column('RVZ_RADVEL','radial_velocity')
    simbad_table.rename_column('RVZ_ERROR', 'rv_error')
    
    simbad_table = simbad_table.filled()
    #add cluster index
    simbad_table.add_index('cluster')
    
    simbad_table['coords'] = \
    SkyCoord(ra = simbad_table['ra'],
        dec = simbad_table['dec'], unit = (u.hour, u.deg),
        obstime = 'J2000',  #simbad returns J2000 coords
        distance = coord.Distance(parallax=Quantity(simbad_table['parallax'])),
        pm_ra_cosdec = simbad_table['pmra'],
        pm_dec = simbad_table['pmdec'],
        radial_velocity = simbad_table['radial_velocity']).apply_space_motion(new_obstime=Time(2015.5,format='decimalyear'))
    
    cluster_data = simbad_table
    return(cluster_data)

#function for getting data on the GAIA known cluster members as defined by this paper http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2018A%26A...616A..10G
def getGAIAKnownMembers(cluster_name):
    known_members = pd.read_csv('ftp://cdsarc.u-strasbg.fr/pub/cats/J/A+A/616/A10/tablea1a.dat',
                      delim_whitespace=True,
                      header=None, index_col=None,
                      names = ['SourceID', 'Cluster', 'RAdeg', 'DEdeg', 'Gmag', 'plx', 'e_plx'])
    known_members.set_index('SourceID', inplace=True)
    
    members.Cluster.unique()
    name_mapper = {'Pleiades': 'Pleiades'}
    members['SimbadCluster'] = members.Cluster.apply(lambda c:name_mapper[c])
    
    known_members['coords']=SkyCoord(ra = np.array(members.RAdeg)*u.degree,
        dec = np.array(members.DEdeg)*u.degree,
        obstime = Time(2015.5,format='decimalyear'),  #Gaia ref epoch is 2015.5
        distance = coord.Distance(parallax=Quantity(np.array(members.plx)*u.mas)))
    
    return(known_members)


#this function will add various distances to the known members table, plot is a boolean to know if the result should be converted into a dimentionless scalar
def getDistances(plot, cluster_data, known_members, cluster_name):
    #add column for distance of each member to cluster center
    known_members['dist_c3d']=[s[1].coords.separation_3d(cluster_data.loc[s[1].SimbadCluster]['coords']) for s in known_members.iterrows()]
    
    #if you want to plot you need it to be a dimentionless scalar
    if(plot == True):
        known_members['dist_c3d'] = Quantity(members.dist_c3d, unit=u.pc).value
    else:
        continue
    
    #get distance of each member from sun
    known_members['r_est'] = [s.distance.value for s in members.coords]
    return(known_members)
